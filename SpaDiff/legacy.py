
"""Legacy DEC components disabled because no tutorial execution path imports them."""

# TUTORIAL-UNUSED: The original implementation is retained below as comments
# so it can be recovered deliberately without being imported at package startup.
# from __future__ import annotations
#
# from typing import Optional
#
# import numpy as np
# import torch
# from torch import nn
# import torch.nn.functional as F
#
# from .model import HiGCN
#
#
# class DEC(nn.Module):
#     def __init__(
#         self,
#         X,
#         HL,
#         node_width,
#         device,
#         args,
#         opt="adam",
#         trajectory: Optional[list] = None,
#         trajectory_interval=50,
#     ):
#         super().__init__()
#         self.X, self.HL, self.device = X, HL, device
#         self.node_width, self.opt = node_width, opt
#         # Fix for original DEC.__init__: never use trajectory=[] as a mutable default.
#         self.trajectory = [] if trajectory is None else trajectory
#         self.trajectory_interval = trajectory_interval
#         self.init_method = args.init_method
#         self.lr, self.epochs = args.lr, args.epochs
#         self.weight_decay = args.weight_decay
#         self.n_clusters, self.nhid = args.n_clusters, args.hidden
#         self.update_interval = args.update_interval
#         self.alpha_dec, self.tol = args.alpha_dec, args.tol_dec
#         self.random_seed = args.random_seed
#         self.model = HiGCN(node_width, self.nhid, args)
#         # Fix for original DEC.fit: mu was created after the optimizer, so the
#         # optimizer never owned the cluster-center parameter.
#         self.mu = nn.Parameter(torch.empty(self.n_clusters, self.nhid))
#         nn.init.xavier_uniform_(self.mu)
#
#     def forward(self, x, hl):
#         z = self.model(x, hl)
#         q = 1.0 / (
#             1.0 + torch.sum((z.unsqueeze(1) - self.mu).square(), dim=2) / self.alpha_dec
#         )
#         # Fix for original DEC.forward: parenthesized Student-t exponent.
#         q = q ** ((self.alpha_dec + 1.0) / 2.0)
#         q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
#         return z, q
#
#     @staticmethod
#     def loss_function(p, q):
#         return torch.mean(
#             torch.sum(p * torch.log(p.clamp_min(1e-12) / q.clamp_min(1e-12)), dim=1)
#         )
#
#     @staticmethod
#     def target_distribution(q):
#         p = q.square() / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
#         return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
#
#     def _initial_labels(self, features):
#         from sklearn.cluster import KMeans
#
#         array = features.detach().cpu().numpy()
#         if self.init_method == "kmeans":
#             estimator = KMeans(
#                 n_clusters=self.n_clusters, n_init=10, random_state=self.random_seed
#             )
#             return estimator.fit_predict(array), estimator.cluster_centers_
#         if self.init_method == "mclust":
#             import scanpy as sc
#             from .utils import mclust_R
#
#             adata = sc.AnnData(array)
#             adata.obsm["emb"] = array
#             labels = mclust_R(
#                 adata,
#                 self.n_clusters,
#                 used_obsm="emb",
#                 pca_num=min(30, array.shape[1]),
#                 random_seed=self.random_seed,
#             )
#             centers = np.vstack([array[labels == label].mean(0) for label in np.unique(labels)])
#             return labels, centers
#         raise ValueError("init_method must be 'kmeans' or 'mclust'")
#
#     def fit(self, X, HL, pca_num=30, res=0.5):
#         del pca_num, res  # retained for compatibility with the original API
#         with torch.no_grad():
#             features = self.model(X, HL)
#         labels, centers = self._initial_labels(features)
#         if centers.shape != tuple(self.mu.shape):
#             raise ValueError("initial clustering did not produce every requested cluster")
#         self.mu.data.copy_(torch.as_tensor(centers, device=X.device, dtype=X.dtype))
#         optimizer_cls = torch.optim.SGD if self.opt == "sgd" else torch.optim.Adam
#         kwargs = {"lr": self.lr, "weight_decay": self.weight_decay}
#         if self.opt == "sgd":
#             kwargs["momentum"] = 0.9
#         optimizer = optimizer_cls(self.parameters(), **kwargs)
#         previous = labels
#         self.trajectory.append(labels.copy())
#         self.train()
#         for epoch in range(self.epochs):
#             if epoch % self.update_interval == 0:
#                 with torch.no_grad():
#                     _, q = self(X, HL)
#                     p = self.target_distribution(q)
#             _, q = self(X, HL)
#             loss = self.loss_function(p, q)
#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()
#             if epoch % self.trajectory_interval == 0:
#                 self.trajectory.append(q.argmax(1).detach().cpu().numpy())
#             current = q.argmax(1).detach().cpu().numpy()
#             changed = np.mean(current != previous)
#             previous = current
#             if epoch > 0 and epoch % self.update_interval == 0 and changed < self.tol:
#                 break
#         return self
#
#     @torch.no_grad()
#     def predict(self):
#         self.eval()
#         z, q = self(self.X, self.HL)
#         # Fix for original DEC.predict: CUDA tensors must move to CPU first.
#         return (
#             q.argmax(1).cpu().numpy(),
#             q.cpu().numpy(),
#             z.cpu().numpy(),
#         )
#
#
# class DEC_Multi(nn.Module):
#     """Preserved multimodal DEC head with stable normalization."""
#
#     def __init__(self, n_clusters, hidden_dim, alpha=1.0):
#         super().__init__()
#         self.n_clusters, self.alpha = n_clusters, alpha
#         self.mu = nn.Parameter(torch.empty(n_clusters, hidden_dim))
#         nn.init.xavier_uniform_(self.mu)
#
#     def soft_assign(self, z):
#         q = 1.0 / (
#             1.0 + torch.sum((z.unsqueeze(1) - self.mu).square(), dim=2) / self.alpha
#         )
#         q = q ** ((self.alpha + 1.0) / 2.0)
#         return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
#
#     @staticmethod
#     def target_distribution(q):
#         p = q.square() / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
#         return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
#
#     def init_cluster(self, z, random_state=0):
#         from sklearn.cluster import KMeans
#
#         estimator = KMeans(self.n_clusters, n_init=10, random_state=random_state)
#         labels = estimator.fit_predict(z.detach().cpu().numpy())
#         self.mu.data.copy_(torch.as_tensor(estimator.cluster_centers_, device=z.device, dtype=z.dtype))
#         return labels
#
#     @staticmethod
#     def attention_entropy(attention_weight, eps=1e-8):
#         return -torch.mean(
#             torch.sum(attention_weight * torch.log(attention_weight + eps), dim=1)
#         )
#
#     def fit(
#         self,
#         model,
#         inputs,
#         HL,
#         n_epochs=200,
#         lr=1e-3,
#         lambda_rec=1.0,
#         device="cpu",
#         verbose=True,
#     ):
#         model.train()
#         self.train()
#         X_rna, X_atac = (tensor.to(device) for tensor in inputs)
#         with torch.no_grad():
#             z = model(X_rna, X_atac, HL)["z"]
#         self.init_cluster(z)
#         optimizer = torch.optim.Adam([*model.parameters(), *self.parameters()], lr=lr)
#         for epoch in range(n_epochs):
#             output = model(X_rna, X_atac, HL)
#             z = output["z"]
#             q = self.soft_assign(z)
#             p = self.target_distribution(q).detach()
#             loss_dec = F.kl_div(q.clamp_min(1e-12).log(), p, reduction="batchmean")
#             loss_rec = F.mse_loss(output["rna_rec"], X_rna) + F.mse_loss(
#                 output["atac_rec"], X_atac
#             )
#             loss = loss_dec + lambda_rec * loss_rec
#             optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             optimizer.step()
#             if verbose and epoch % 10 == 0:
#                 print(
#                     f"Epoch {epoch:03d} | Loss={loss.item():.4f} | "
#                     f"DEC={loss_dec.item():.4f} | REC={loss_rec.item():.4f}"
#                 )
#         return z.detach(), q.detach()
#
#     @torch.no_grad()
#     def predict(self, model, inputs, HL, device="cpu"):
#         model.eval()
#         self.eval()
#         X_rna, X_atac = (tensor.to(device) for tensor in inputs)
#         output = model(X_rna, X_atac, HL)
#         q = self.soft_assign(output["z"])
#         return (
#             q.argmax(1).cpu(),
#             output["z"].cpu(),
#             output["z_rna"].cpu(),
#             output["z_atac"].cpu(),
#         )
#
#
