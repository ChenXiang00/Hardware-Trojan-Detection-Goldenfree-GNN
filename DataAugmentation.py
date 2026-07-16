import os, sys
import copy
import glob
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dropout_edge


def _load_one(path: str):
    if path.endswith((".pt", ".pth")):
        return torch.load(path, map_location="cpu", weights_only=False)
    with open(path, "rb") as f:
        return pickle.load(f)

def _as_list(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, dict) and "graphs" in obj:
        return list(obj["graphs"])
    return [obj]

def normalize_graph(g: Data, benign_default: bool = False):
    if (not hasattr(g, "x")) or (g.x is None):
        return None
    if (not hasattr(g, "edge_index")) or (g.edge_index is None):
        return None

    g = copy.deepcopy(g)
    g.x = g.x.float()
    g.edge_index = g.edge_index.long()
    g.num_nodes = int(g.x.size(0))

    if hasattr(g, "y") and g.y is not None and int(g.y.numel()) == g.num_nodes:
        y = g.y.view(-1).long()
    elif hasattr(g, "loc_label") and g.loc_label is not None and int(g.loc_label.numel()) == g.num_nodes: 
        y = g.loc_label.view(-1).long()
    elif benign_default:
        y = torch.zeros(g.num_nodes, dtype=torch.long)
    else:
        return None

    g.y = y
    
    if hasattr(g, "loc_label"):
        del g.loc_label
    
    return Data(x=g.x, edge_index=g.edge_index, y=y)

def load_synthetic_graphs(path: str):
    out = []
    
    
    for obj in _as_list(_load_one(path)):
        g = normalize_graph(obj, benign_default=False)
        if g is not None:
            g.graph_source = torch.tensor([1], dtype=torch.long)
            out.append(g)
            
    
    return out

def load_benign_graphs(root_dir: str):
    out, files = [], []
    for ext in ("*.pkl", "*.pickle", "*.pt", "*.pth"):
        files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    for fp in sorted(set(files)):
        for obj in _as_list(_load_one(fp)):
            g = normalize_graph(obj, benign_default=True)
            if g is not None:
                g = ensure_x_2d_64(g)
                g.graph_source = torch.tensor([0], dtype=torch.long)
                out.append(g)
    return out


def ensure_x_2d_64(g):


    if g.x.dim() == 1:
        g.x = F.one_hot(g.x.long(), num_classes=64).float()
    elif g.x.dim() == 2 and g.x.size(1) == 1:
        g.x = F.one_hot(g.x.view(-1).long(), num_classes=64).float()
    else:
        g.x = g.x.float()
    return g

def augment_graph(data: Data, feature_mask: float, edge_drop: float):
    aug = copy.deepcopy(data)

    # 保守 feature masking：同一張圖共享一個 feature-dim mask
    x = aug.x.clone()
    if feature_mask > 0:
        feat_keep = (torch.rand(x.size(1)) > feature_mask).to(x.device)
        x = x * feat_keep.unsqueeze(0).float()

    # 保守 edge dropout：只刪邊，不刪 node
    edge_index, _ = dropout_edge(aug.edge_index, p=edge_drop, training=True)

    aug.x = x
    aug.edge_index = edge_index
    aug.y = data.y.clone()
    aug.num_nodes = int(data.num_nodes)
    return aug

def build_two_views(batch, feature_mask: float, edge_drop: float, device):

    batch = batch.to(device)

    view1 = copy.copy(batch)
    view2 = copy.copy(batch)

    # feature masking
    x1 = batch.x.clone()
    x2 = batch.x.clone()

    if feature_mask > 0:
        
        mask1 = (torch.rand(x1.size(1), device=device) > feature_mask).float()

        mask2 = (torch.rand(x2.size(1), device=device) > feature_mask).float()

        x1 = x1 * mask1.unsqueeze(0)
        x2 = x2 * mask2.unsqueeze(0)

    # edge dropout
    edge_index1, _ = dropout_edge(batch.edge_index, p=edge_drop, training=True)

    edge_index2, _ = dropout_edge(batch.edge_index, p=edge_drop, training=True)

    view1.x = x1
    view1.edge_index = edge_index1

    view2.x = x2
    view2.edge_index = edge_index2

    return view1, view2
