import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from torch_sparse import matmul 
from torch_geometric.nn import MessagePassing 

class HiGCN_prop(MessagePassing):
    def __init__(self, K, alpha):
        super().__init__(aggr='add')
        self.K = K
        self.alpha = alpha
        self.fW = nn.Parameter(torch.Tensor(K + 1))
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K + 1):
            self.fW.data[k] = self.alpha * (1 - self.alpha) ** k

    def forward(self, x, HL):
        fW = F.softmax(self.fW, dim=0)
        out = fW[0] * x
        x_k = x
        for k in range(self.K):
            x_k = matmul(HL, x_k)
            out = out + fW[k + 1] * x_k
        return out

class HiGCN(nn.Module):
    """
    高阶图卷积网络
    """
    def __init__(self, num_features: int, num_out: int, args):
        super(HiGCN, self).__init__()
        self.Order = args.Order
        self.dprate = args.dprate 
        self.dropout = args.dropout 
        self.lin_in = nn.ModuleList()
        self.hgc = nn.ModuleList() 
        for i in range(self.Order):
            self.lin_in.append(Linear(num_features, args.hidden)) 
            self.hgc.append(HiGCN_prop(args.K, args.alpha)) 
        self.lin_out = Linear(args.hidden * self.Order, num_out) 

    def forward(self, x, HL) -> torch.Tensor:
        x_list = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.Order):
            H = self.lin_in[i](x)
            
            if self.dprate > 0.0:
                H = F.dropout(H, p=self.dprate, training=self.training)
            H = self.hgc[i](H, HL[i + 1])
            
            x_list.append(H)

        z = torch.cat(x_list, dim=1)
        # z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.lin_out(z)
        
        return z 


class MultiAutoEncoder(nn.Module):
    def __init__(self, rna_dim, atac_dim, args):
        super().__init__()

        self.rna_dim = rna_dim
        self.atac_dim = atac_dim
        self.hidden = args.hidden

        # Encoder
        self.enc_rna = HiGCN(self.rna_dim, args.hidden, args)
        self.enc_atac = HiGCN(self.atac_dim, args.hidden, args)

        # Decoder
        self.dec_rna = Decoder(
            latent_dim=args.hidden,
            out_dim=rna_dim,
            hidden_dim=args.h_decoder
        )

        self.dec_atac = Decoder(
            latent_dim=args.hidden,
            out_dim=atac_dim,
            hidden_dim=args.h_decoder
        )

        # Attention
        self.attn = nn.Linear(args.hidden * 2, 2)

    def forward(self, X_rna, X_atac, HL):
        # Encode
        z_rna = self.enc_rna(X_rna, HL)
        z_atac = self.enc_atac(X_atac, HL)

        # Attention
        attn_score = self.attn(torch.cat([z_rna, z_atac], dim=1))
        weight = torch.softmax(attn_score, dim=1)
        z = weight[:, 0:1] * z_rna + weight[:, 1:2] * z_atac

        # Decode
        X_rna_hat = self.dec_rna(z)
        X_atac_hat = self.dec_atac(z)

        return {
            "z": z,
            "z_rna": z_rna,
            "z_atac": z_atac,
            "weight": weight,
            "rna_rec": X_rna_hat,
            "atac_rec": X_atac_hat
        }
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, z):
        return self.net(z)

class AutoEncoder(nn.Module):
    def __init__(
        self,
        enc,
        dec
    ):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self, X, HL):
        z = self.encoder(X, HL)

        X_rec = self.decoder(z)

        return {
            "z": z,
            "X_rec": X_rec
        }