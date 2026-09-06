import os, sys, time
import json
import random
import argparse
from xml.parsers.expat import model
import torch
import gc 
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from Encoder import GINEncoderPlus
from Agentic import FeatureDetector, ContrastiveWrapper, PrototypeBank
from DataAugmentation import load_synthetic_graphs, load_benign_graphs, build_two_views


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def split_graphs(benign_graphs, infected_graphs, val_ratio=0.2, seed=42):
    benign_tr, benign_va = train_test_split(
        benign_graphs, test_size=val_ratio, random_state=seed, shuffle=True
    )
    infected_tr, infected_va = train_test_split(
        infected_graphs, test_size=val_ratio, random_state=seed, shuffle=True
    )
    
    
    return benign_tr + infected_tr, benign_va + infected_va

def compute_class_weights(graphs):
    counts = torch.zeros(2, dtype=torch.float)
    for g in graphs:
        counts += torch.bincount(g.y.view(-1), minlength=2).float()
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return weights

def make_train_sampler(train_graphs, balance_graphs: bool):
    if not balance_graphs:
        return None
    src = torch.tensor([int(g.graph_source.item()) for g in train_graphs], dtype=torch.long)
    counts = torch.bincount(src, minlength=2).float().clamp_min(1.0)
    weights = 1.0 / counts[src]
    return WeightedRandomSampler(weights=weights, num_samples=len(train_graphs), replacement=True)

@torch.no_grad()
def evaluate(model, loader, ce_criterion, device):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    total_ce, total_nodes = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        z, logits = model(batch)
        y = batch.y.view(-1).long()

        ce = ce_criterion(logits, y)
        total_ce += float(ce.item()) * y.numel()
        total_nodes += y.numel()

        prob = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)

        all_y.append(y.cpu())
        all_pred.append(pred.cpu())
        all_prob.append(prob.cpu())

    y_true = torch.cat(all_y).numpy()
    y_pred = torch.cat(all_pred).numpy()
    y_prob = torch.cat(all_prob).numpy()

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")

    return {
        "ce": total_ce / max(total_nodes, 1),
        "precision_0": float(p[0]),
        "recall_0": float(r[0]),
        "f1_0": float(f1[0]),
        "support_0": int(support[0]),
        "precision_1": float(p[1]),
        "recall_1": float(r[1]),
        "f1_1": float(f1[1]),
        "support_1": int(support[1]),
        "macro_f1": float((f1[0] + f1[1]) / 2.0),
        "auroc": auroc,
    }

def train_one_epoch(model, loader, optimizer, ce_criterion, ntxent, proto_bank, args, device):
    model.train()
    total_total, total_ce, total_ntx, total_proto = 0.0, 0.0, 0.0, 0.0
    total_nodes = 0
    proto_dist_hist = []

    for batch in loader:
        view1, view2 = build_two_views(batch, args.feature_mask, args.edge_drop, device)
        y = view1.y.view(-1).long()

        z1, logits1 = model(view1)
        z2, logits2 = model(view2)

        ce_loss = 0.5 * (ce_criterion(logits1, y) + ce_criterion(logits2, y))
        ntx_loss = ntxent(z1, z2)

        z_cat = torch.cat([z1, z2], dim=0)
        y_cat = torch.cat([y, y], dim=0)
        proto_loss, proto_dist = proto_bank.separation_loss(
            z_cat, y_cat, margin=args.proto_margin
        )

        loss = ce_loss + ntx_loss + args.lambda_proto * proto_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        proto_bank.update(z_cat, y_cat)

        n = y.numel()
        total_total += float(loss.item()) * n
        total_ce += float(ce_loss.item()) * n
        total_ntx += float(ntx_loss.item()) * n
        total_proto += float(proto_loss.item()) * n
        total_nodes += n

        if not torch.isnan(proto_dist):
            proto_dist_hist.append(float(proto_dist.item()))
        

    denom = max(total_nodes, 1)
    return {
        "total": total_total / denom,
        "ce": total_ce / denom,
        "ntx": total_ntx / denom,
        "proto": total_proto / denom,
        "proto_dist": float(sum(proto_dist_hist) / len(proto_dist_hist)) if proto_dist_hist else float("nan"),
    }


def train_one_epoch_without_separation(model, loader, optimizer, ce_criterion, ntxent, args, device):
    model.train()
    total_total, total_ce, total_ntx, total_proto = 0.0, 0.0, 0.0, 0.0
    total_nodes = 0
    proto_dist_hist = []

    for batch in loader:
        view1, view2 = build_two_views(batch, args.feature_mask, args.edge_drop, device)
        y = view1.y.view(-1).long()

        z1, logits1 = model(view1)
        z2, logits2 = model(view2)

        ce_loss = 0.5 * (ce_criterion(logits1, y) + ce_criterion(logits2, y))
        ntx_loss = ntxent(z1, z2)

        z_cat = torch.cat([z1, z2], dim=0)
        y_cat = torch.cat([y, y], dim=0)
        """
        proto_loss, proto_dist = proto_bank.separation_loss(
            z_cat, y_cat, margin=args.proto_margin
        )
        """
        loss = ce_loss + ntx_loss # + args.lambda_proto * proto_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        n = y.numel()
        total_total += float(loss.item()) * n
        total_ce += float(ce_loss.item()) * n
        total_ntx += float(ntx_loss.item()) * n
        #total_proto += float(proto_loss.item()) * n
        total_nodes += n


    denom = max(total_nodes, 1)
    return {
        "total": total_total / denom,
        "ce": total_ce / denom,
        "ntx": total_ntx / denom

        #"proto": total_proto / denom,
        #"proto_dist": float(sum(proto_dist_hist) / len(proto_dist_hist)) if proto_dist_hist else float("nan"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    #DataSet
    p.add_argument("--synthetic-ht", type=str, default="./DataSet/Synthetic_Dataset/synthetic_ht.pkl")
    p.add_argument("--benign-dir", type=str, default="./DataSet/Traditional_Dataset")

    p.add_argument("--out-dir", type=str, default="./runs/Encoder")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--feature-mask", type=float, default=0.10)
    p.add_argument("--edge-drop", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lambda-proto", type=float, default=0.2)

    p.add_argument("--proto-margin", type=float, default=0.5)
    p.add_argument("--proto-momentum", type=float, default=0.9)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=2028)
    p.add_argument("--balance-graphs", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")

    synthetic = load_synthetic_graphs(args.synthetic_ht)
    benign = load_benign_graphs(args.benign_dir)
    
    #extra_AES =  load_synthetic_graphs(args.extra_AES_ht)
    #extra_RS232 = load_synthetic_graphs(args.extra_RS232_ht)
    #extra_PIC = load_synthetic_graphs(args.extra_PIC_ht)

    train_graphs, val_graphs = split_graphs(benign, synthetic, val_ratio=args.val_ratio, seed=args.seed)
    #train_graphs +=  extra_PIC
    #train_graphs += extra_AES 
    #train_graphs +=  extra_RS232


    
    
    in_dim = int(train_graphs[0].x.size(1))
    encoder = GINEncoderPlus(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        layers=args.num_layers,
        dropout=args.dropout,
        use_gate=True,
        device=str(device),
    )
    model = FeatureDetector(encoder).to(device)

    class_weights = compute_class_weights(train_graphs).to(device)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ntxent = ContrastiveWrapper(temperature=args.temperature)
    proto_bank = PrototypeBank(dim=encoder.out_dim, momentum=args.proto_momentum).to(device)

    sampler = make_train_sampler(train_graphs, balance_graphs=args.balance_graphs)
    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        shuffle=(sampler is None),
        sampler=sampler,
    )

    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    history = {
        "epoch": [],
        "train_total": [], "train_ce": [], "train_ntx": [], 
        "train_proto": [],
        "proto_dist": [],
        "val_ce": [], "val_f1_0": [], "val_f1_1": [], "val_macro_f1": [], "val_auroc": [],
        "macro_precision": [], "macro_recall": []
    }

    best_score = -1.0
    best_ckpt = None
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, ce_criterion, ntxent, proto_bank, args, device
        )
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        """
        train_stats = train_one_epoch_without_separation(
            model, train_loader, optimizer, ce_criterion, ntxent, args, device
        )
        """
        val_stats = evaluate(model, val_loader, ce_criterion, device)

        
        
        history["epoch"].append(epoch)
        history["train_total"].append(train_stats["total"])
        history["train_ce"].append(train_stats["ce"])
        history["train_ntx"].append(train_stats["ntx"])
        history["train_proto"].append(train_stats["proto"])
        history["proto_dist"].append(train_stats["proto_dist"])
        history["val_ce"].append(val_stats["ce"])
        history["val_f1_0"].append(val_stats["f1_0"])
        history["val_f1_1"].append(val_stats["f1_1"])
        history["val_macro_f1"].append(val_stats["macro_f1"])
        history["val_auroc"].append(val_stats["auroc"])
        
        macro_precision = (val_stats["precision_0"] + val_stats["precision_1"]) / 2
        macro_recall = (val_stats["recall_0"] + val_stats["recall_1"]) / 2
        
        history["macro_precision"].append(macro_precision)
        history["macro_recall"].append(macro_recall)

        if val_stats["macro_f1"] > best_score:
            best_score = val_stats["macro_f1"]
            best_ckpt = {
                "epoch": epoch,
                "encoder_state_dict": model.encoder.state_dict(),
                "head_state_dict": model.head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "prototypes": proto_bank.prototypes.detach().cpu(),
                "prototype_initialized": proto_bank.initialized.detach().cpu(),
                "history": history,
                "args": vars(args),
                "val_stats": val_stats,
            }
            torch.save(best_ckpt, os.path.join(args.out_dir, "best_checkpoint.pt"))
            torch.save(model.encoder.state_dict(), os.path.join(args.out_dir, "encoder_best.pt"))
            torch.save(model.head.state_dict(), os.path.join(args.out_dir, "node_head_best.pt"))
            
            torch.save(
                {
                    "prototypes": proto_bank.prototypes.detach().cpu(),
                    "initialized": proto_bank.initialized.detach().cpu(),
                },
                os.path.join(args.out_dir, "prototypes.pt"),
            )
            
        """Memory cleanup"""
        print(
            f"[MEM] alloc={torch.cuda.memory_allocated()/1024**3:.2f}GB "
            f"reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB "
            f"max={torch.cuda.max_memory_allocated()/1024**3:.2f}GB"
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        

        print(
            f"[Epoch {epoch:03d}] "
            f"train_total={train_stats['total']:.4f} "
            f"train_ce={train_stats['ce']:.4f} "
            f"train_ntx={train_stats['ntx']:.4f} "
            f"train_proto={train_stats['proto']:.4f} "
            f"val_p={macro_precision:.4f} "
            f"val_r={macro_recall:.4f} "
            f"val_f1={val_stats['macro_f1']:.4f} "
            f"val_auroc={val_stats['auroc']:.4f}"
        )
    print(f"Avg Epoch Time = {np.mean(epoch_times):.2f}s")
    print(f"Total Time = {sum(epoch_times):.2f}s")  

    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(best_ckpt["val_stats"], f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
