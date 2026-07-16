import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GINConv, GraphNorm, global_add_pool

#Encoder
class GINEncoderPlus(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int,
                 dropout: float = 0.5, use_gate: bool = True, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout = dropout
        self.use_gate = use_gate



        #1.Input Encoder
        self.input_encoder = nn.Linear(in_dim, hidden_dim)

        #2.Muti-GIN Layer
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(GraphNorm(hidden_dim)) #BatchNorm change to GraphNorm

        #3.Node-wise attention gate
        if use_gate:
            self.gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(), 
            )
        else:
            self.gate_nn = None
        #4.Embedding Layer 
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        #5.Output DIm
        self.out_dim = hidden_dim

        self.to(self.device)

    def forward(self, x, edge_index, batch):    #add an extra batch
        #to device
        x = x.to(self.device).float()
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)  #add

        #input encoder
        x = self.input_encoder(x)

        #Muti-GINLayers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x, batch)    #add
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Node-wise attention gate
        if self.gate_nn is not None:
            gate = self.gate_nn(x)          
            x = x * gate    

        # backend embedding
        x = self.embedding_layer(x)        
        return x


class ReadoutProject(nn.Module):
    def __init__(self, enc_out:int, proj_dim:int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(enc_out, enc_out), nn.ReLU(), nn.Linear(enc_out, proj_dim)
        )

    def forward(self, node_emb:Tensor, batch:Tensor)-> Tensor:
        g = global_add_pool(node_emb, batch)
        z = F.normalize(self.proj(g), dim = -1)
        return z
    
def nt_xent_multi(z:Tensor, pos_mask:Tensor, t:float)->Tensor:
    N = z.size(0)
    sim = (z @ z.t()) / t
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    exp_sim = torch.exp(sim)
    pos_sum = (exp_sim * pos_mask.float()).sum(dim = 1) + 1e-8
    all_sum = exp_sim.sum(dim = 1) +  1e-8
    loss = -torch.log(pos_sum / all_sum)
    return loss.mean()

class GraphClassifier(nn.Module):
    
    def __init__(self, encoder:nn.Module, in_dim:int, num_class:int = 2, hidden:int = 128, dropout: float = 0.5):

        super().__init__()
        self.encoder = encoder
        self.in_dim = in_dim
        self.attn = GlobalAttentionReadout(in_dim=in_dim, hidden=in_dim // 2, dropout=dropout)

        self.post_dim = in_dim * 4
        self.pre_norm = nn.LayerNorm(self.post_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.post_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        
        self.proj = nn.Linear(self.post_dim, hidden)
        self.classifier_w = nn.Parameter(torch.randn(num_class, hidden))
        nn.init.xavier_uniform_(self.classifier_w)
        self.scale = nn.Parameter(torch.tensor(7.0))

        #Extra
        #self.classifier = nn.Linear(hidden, num_class)
    
    def hybrid_readout(self, x, batch, diff_mask = None):
        
        mean = global_mean_pool(x, batch)
        
        max  = global_max_pool(x, batch)
        
        with torch.no_grad():
            B = int(batch.max().item()) + 1
        
        stds= []
        for b in range(B):
            mask = (batch == b)
            if mask.any():
                x_b = x[mask]
                stds.append(x_b.std(dim=0, unbiased=False, keepdim=True))
        std = torch.cat(stds, dim=0)
        
        att = self.attn(x, batch, diff_mask)
        
        return torch.cat([mean, max, std, att], dim=1)        
    
    def forward(self, batch):
        h = self.encoder(batch.x, batch.edge_index, batch = batch.batch)   
        diff_m = getattr(batch, "is_trojan", None)        
        g = self.hybrid_readout(h, batch.batch, diff_mask=diff_m)              
        g = self.pre_norm(g)
        z = self.mlp(g)                                       
        z = z + self.proj(g)
        
        z_n = F.normalize(z, dim=-1)
        w_n = F.normalize(self.classifier_w, dim=-1)
        logits = self.scale * (z_n @ w_n.t())                                   
        
        #logits = self.classifier(z)
        return logits, z

class GlobalAttentionReadout(nn.Module):
    
    def __init__(self, in_dim:int, hidden:int = 128, dropout:float = 0.5, bias:float = 1.2):
        super().__init__()
        self.gate =nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.bias = bias
    def forward(self, x, batch, diff_mask: Optional[Tensor] = None):
        
        score = self.gate(x)
        
        if diff_mask is not None:
            score = score + self.bias * diff_mask.view(-1, 1).float()
        
        with torch.no_grad():    
            B = int(batch.max().item()) + 1
        out = []
        for i in range(B):
            mask = (batch == i)
            if mask.any():
                x_b = x[mask]
                s_b = score[mask]
                a_b = torch.softmax(s_b, dim=0)
                out.append((a_b * x_b).sum(dim=0, keepdim=True))
                
        return torch.cat(out, dim=0)


class ReadoutProject(nn.Module):
    def __init__(self, enc_out:int, proj_dim:int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(enc_out, enc_out), nn.ReLU(), nn.Linear(enc_out, proj_dim)
        )

    def forward(self, node_emb:Tensor, batch:Tensor)-> Tensor:
        g = global_add_pool(node_emb, batch)
        z = F.normalize(self.proj(g), dim = -1)
        return z