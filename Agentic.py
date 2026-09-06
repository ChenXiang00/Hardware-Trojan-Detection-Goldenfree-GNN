import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_metric_learning import losses as pml_losses
#from torch_geometric.nn import GCNConv

class FeatureDetector(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.out_dim, 2)

    def encode(self, data):
        return self.encoder(data.x, data.edge_index, data.batch)

    def forward(self, data):
        z = self.encode(data)
        logits = self.head(z)
        return z, logits

def manual_ntxent(z1: torch.Tensor, z2: torch.Tensor, temperature: float):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    n = z1.size(0)

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    sim = sim.masked_fill(torch.eye(2 * n, device=z.device, dtype=torch.bool),-1e9)

    labels = torch.arange(n, device=z.device)
    labels = torch.cat([labels + n, labels], dim=0)

    return F.cross_entropy(sim, labels)

class ContrastiveWrapper:
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def __call__(self, z1, z2):
        return manual_ntxent(z1, z2, self.temperature)

class PrototypeBank(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(2, dim))
        self.register_buffer("initialized", torch.zeros(2, dtype=torch.bool))

    def separation_loss(self, z: torch.Tensor, y: torch.Tensor, margin: float = 0.0):
        # Normalize node embeddings
        z = F.normalize(z, dim=-1)

        # Split Benign / Trojan embeddings
        mask_b = (y == 0)
        mask_t = (y == 1)

        if not mask_b.any() or not mask_t.any():
            zero = z.sum() * 0.0
            return zero, torch.tensor(float("nan"), device=z.device)

        Z_B = z[mask_b]
        Z_T = z[mask_t]

        # Prototype P_B and P_T
        P_B = Z_B.mean(dim=0)
        P_T = Z_T.mean(dim=0)

        P_B = F.normalize(P_B, dim=0)
        P_T = F.normalize(P_T, dim=0)

        # Same-class attraction
        loss_B_same = (1.0 - torch.matmul(Z_B, P_B)).mean()
        loss_T_same = (1.0 - torch.matmul(Z_T, P_T)).mean()

        # Cross-class separation
        loss_B_cross = F.relu(torch.matmul(Z_B, P_T) - margin).mean()
        loss_T_cross = F.relu(torch.matmul(Z_T, P_B) - margin).mean()

        # Prototype separation
        loss_proto = F.relu(torch.dot(P_B, P_T) - margin)

        # Total separation loss
        loss = (loss_B_same+ loss_T_same+ loss_B_cross+ loss_T_cross+ loss_proto)
        dist = torch.norm(P_B - P_T, p=2)

        return loss, dist.detach()

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor):
        z = F.normalize(z.detach(), dim=-1)
        for c in [0, 1]:
            mask = (y == c)
            if not mask.any():
                continue

            p = z[mask].mean(dim=0)
            p = F.normalize(p, dim=0)

            if self.initialized[c]:
                new_p = self.momentum * self.prototypes[c] + (1.0 - self.momentum) * p
                self.prototypes[c].copy_(F.normalize(new_p, dim=0))
            else:
                self.prototypes[c].copy_(p)
                self.initialized[c] = True


class PrototypeSeparation(nn.Module):

    def __init__(self, dim:int, momentum:float=0.9):
        super().__init__()

        self.momentum = momentum
        self.register_buffer("prototypes", F.normalize(torch.randn(2, dim), dim=1))
        self.register_buffer("initialized",torch.zeros(2, dtype=torch.bool))

    @torch.no_grad()
    def update(self, z:torch.Tensor, y:torch.Tensor):

        z = F.normalize(z, dim=1)
        for cls in [0, 1]:
            mask = (y == cls)

            if mask.sum() == 0:
                continue

            mean_z = z[mask].mean(dim=0)
            mean_z = F.normalize(mean_z.unsqueeze(0), dim=1).squeeze(0)

            if not self.initialized[cls]:
                self.prototypes[cls] = mean_z
                self.initialized[cls] = True
            else:
                self.prototypes[cls] = (self.momentum * self.prototypes[cls]+ (1.0 - self.momentum) * mean_z)
                self.prototypes[cls] = F.normalize(self.prototypes[cls].unsqueeze(0),dim=1).squeeze(0)

    def separation_loss(self, z:torch.Tensor, y:torch.Tensor, margin:float=0.5):

        z = F.normalize(z, dim=1)

        P_B = self.prototypes[0]
        P_T = self.prototypes[1]

        loss = z.new_tensor(0.0)

        benign_mask = (y == 0)
        trojan_mask = (y == 1)

        if benign_mask.any():
            
            Z_B = z[benign_mask]
            loss += (1.0 - (Z_B @ P_B).mean())
            loss += F.relu((Z_B @ P_T).mean() - margin)

        if trojan_mask.any():
            
            Z_T = z[trojan_mask]
            loss += (1.0 -(Z_T @ P_T).mean())
            loss += F.relu((Z_T @ P_B).mean() - margin)

        proto_sim = torch.dot(P_B, P_T)

        loss += F.relu(proto_sim - margin)

        return loss, proto_sim