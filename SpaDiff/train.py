import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .utils import mclust_R
from .model import HiGCN
import scanpy as sc
import numpy as np
import pandas as pd

class DEC(nn.Module):
    def __init__(self, X, HL, node_width, device,args,
                 opt="adam",trajectory=[],trajectory_interval=50,
                 ):
        super(DEC, self).__init__()
        
        self.X = X
        self.HL = HL
        self.device = device
        self.node_width = node_width
        self.opt = opt
        self.trajectory=trajectory
        self.trajectory_interval=trajectory_interval
        self.init_method = args.init_method 
        self.lr= args.lr
        self.epochs= args.epochs
        self.weight_decay=args.weight_decay
        self.n_clusters=args.n_clusters
        self.nhid=args.hidden
        self.update_interval=args.update_interval
        self.n_neighbors = args.n_neighbors
        self.alpha_dec = args.alpha_dec
        self.tol = args.tol_dec
        self.random_seed = args.random_seed
        self.model = HiGCN(self.node_width, self.nhid, args)


    def forward(self, x, hl):
        z = self.model(x, hl)
        q = 1.0 / ((1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha_dec) + 1e-8)
        q = q ** (self.alpha_dec+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q
    
    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss
    
    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p
    
    def fit(self, X, HL, pca_num = 30, res=0.5):    
        
        if self.opt=="sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.opt=="adam":
            self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr, weight_decay=self.weight_decay)
           
        features = self.model(X,HL)
        adata=sc.AnnData(features.cpu().detach().numpy())
        adata.obsm['emb'] = adata.X   

        if self.init_method=="kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_seed)
            y_pred= kmeans.fit_predict(adata.obsm['emb'])
        elif self.init_method=="mclust":
            y_pred = mclust_R(adata, self.n_clusters,used_obsm='emb')


        y_pred_last = y_pred 

        self.mu = nn.Parameter(torch.Tensor(self.n_clusters, self.nhid))
        self.trajectory.append(y_pred)
        
        features=pd.DataFrame(features.cpu().detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())        
        unique_labels = np.unique(y_pred)
        print(f"y_pred: {len(unique_labels)}")
        print(f"cluster_centers : {cluster_centers.shape}")

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()

        for epoch in range(self.epochs):
            if epoch%self.update_interval == 0:
                z, q = self.forward(X,HL)
                p = self.target_distribution(q).data
            self.optimizer.zero_grad()
            _,q = self.forward(X, HL)
            loss = self.loss_function(p, q)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
            
            if epoch%self.trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            #Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch>0 and (epoch-1)%self.update_interval == 0 and delta_label < self.tol:
                print('delta_label ', delta_label, '< tol ', self.tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break
            torch.cuda.empty_cache()
            
    def predict(self):
        z,q = self.forward(self.X,self.HL) 
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        z = z.detach().numpy()
        prob=q.detach().numpy()
        return y_pred, prob,z


class DEC_Multi(nn.Module):
    def __init__(self, n_clusters, hidden_dim, alpha=1.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mu = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
    
    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1) / 2)
        return q / torch.sum(q, dim=1, keepdim=True)

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        return p / torch.sum(p, dim=1, keepdim=True)

    def init_cluster(self, z):
        km = KMeans(self.n_clusters, n_init=10)
        y = km.fit_predict(z.detach().cpu().numpy())
        self.mu.data.copy_(
            torch.tensor(km.cluster_centers_, dtype=torch.float32, device=z.device)
        )
        return y
    
    def attention_entropy(self, attn_weight, eps=1e-8):
        return -torch.mean(
            torch.sum(attn_weight * torch.log(attn_weight + eps), dim=1)
        )


    def fit(self, model, inputs, HL, n_epochs=200, lr=1e-3, lambda_rec=1.0, device="cpu", verbose=True):
        model.train()
        self.train()
        X_rna, X_atac = inputs
        X_rna = X_rna.to(device)
        X_atac = X_atac.to(device)
        with torch.no_grad():
            out = model((X_rna, X_atac), HL)
            z = out["z"]

        self.init_cluster(z)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.parameters()),
            lr=lr
        )

        
        for epoch in range(n_epochs):
            out = model((X_rna, X_atac), HL)
            z = out["z"]
            weight = out["weight"]
            X_rna_hat = out["rna_rec"]
            X_atac_hat = out["atac_rec"]
            # DEC loss
            q = self.soft_assign(z)
            p = self.target_distribution(q).detach()
            loss_dec = F.kl_div(q.log(), p, reduction="batchmean")
            # reconstruction loss
            loss_rna = F.mse_loss(X_rna_hat, X_rna)
            loss_atac = F.mse_loss(X_atac_hat, X_atac)
            loss_rec = loss_rna + loss_atac
            # total loss
            loss = loss_dec + lambda_rec * loss_rec
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and epoch % 1 == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Loss={loss.item():.4f} | "
                    f"DEC={loss_dec.item():.4f} | "
                    f"REC={loss_rec.item():.4f}"
                )
        return z.detach(), q.detach()
    
    @torch.no_grad()
    def predict(self, model, inputs, HL, device="cpu"):
        model.eval()
        self.eval()

        X_rna, X_atac = inputs
        X_rna = X_rna.to(device)
        X_atac = X_atac.to(device)

        out = model((X_rna, X_atac), HL)
        z = out["z"]
        z_rna = out["z_rna"]  
        z_atac = out["z_atac"]

        q = self.soft_assign(z)
        y = q.argmax(dim=1)

        return y.cpu(), z.cpu(), z_rna.cpu(), z_atac.cpu()
    

def train_warmup(model, X_rna, X_atac, HL, args, verbose=True):

    warmup_epochs = args.warmup_epochs
    lr = args.lr_warmup
    lambda_atac = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    loss_history = {
        "total": [],
        "rna": [],
        "atac": []
    }

    for epoch in range(warmup_epochs):

        out = model((X_rna, X_atac), HL)

        loss_rna = F.mse_loss(out["rna_rec"], X_rna)
        loss_atac = F.mse_loss(out["atac_rec"], X_atac)
        loss = loss_rna + lambda_atac * loss_atac

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history["total"].append(loss.item())
        loss_history["rna"].append(loss_rna.item())
        loss_history["atac"].append(loss_atac.item())
        if verbose and epoch % 10 == 0:
            print(
                f"[Warm-up] Epoch {epoch:03d} | "
                f"Total={loss.item():.4f} | "
                f"RNA={loss_rna.item():.4f} | "
                f"ATAC={loss_atac.item():.4f}"
            )
    return loss_history
