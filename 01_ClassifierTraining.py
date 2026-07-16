# 02_GraphClassifierTraining.py
import os, sys, time
import json
import random
import argparse
import pickle
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch_geometric.loader import DataLoader
from Encoder import GINEncoderPlus, GraphClassifier
from DataAugmentation import load_synthetic_graphs, load_benign_graphs
from DataAugmentation import ensure_x_2d_64

# Utils
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def num_nodes(graphs):
    for g in graphs:
        num_nodes = g.x.size(0)
        g.num_nodes = num_nodes
    
    return graphs



def split_graphs(benign_graphs, infected_graphs, val_ratio=0.2, seed=42):

    benign_tr, benign_va = train_test_split(
        benign_graphs,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True
    )

    infected_tr, infected_va = train_test_split(
        infected_graphs,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True
    )

    return (benign_tr + infected_tr, benign_va + infected_va)


# Graph Label
def convert_to_graph_label(graphs):

    for g in graphs:
        if g.y.numel() > 1:
            if (g.y == 1).any():
                g.y = torch.tensor([1], dtype=torch.long)
            else:
                g.y = torch.tensor([0], dtype=torch.long)

    return graphs


# Evaluation
@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0
    ys = []
    preds = []
    probs = []
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        y = batch.y.view(-1)
        loss = criterion(logits, y)

        total_loss += (loss.item() * y.size(0))
        total_graphs += y.size(0)

        prob = torch.softmax(logits, dim=-1)[:,1]
        pred = logits.argmax(dim=1)

        ys.extend(y.cpu().tolist())
        preds.extend(pred.cpu().tolist())
        probs.extend(prob.cpu().tolist())

    p,r,f1,_ = precision_recall_fscore_support(ys, preds, average="binary", zero_division=0)

    try:
        auroc = roc_auc_score(ys, probs)
    except:
        auroc = float("nan")

    return {
        "loss" : total_loss / total_graphs,
        "precision" : p,
        "recall": r,
        "f1": f1,
        "auroc": auroc
    }


# Train One Epoch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        y = batch.y.view(-1)

        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += (loss.item() * y.size(0))
        total_graphs += y.size(0)

    return total_loss / total_graphs


# Args
def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--synthetic-ht", type=str, default="synthetic_ht.pkl")
    p.add_argument("--benign-dir", type=str, default="/home/chen/Desktop/NewHT/Separated_DataSet/train/B")
    p.add_argument("--encoder-ckpt", type=str, default="./runs/node_detector/encoder_best.pt")
    p.add_argument("--out-dir", type=str, default="./runs/graph_classifier")
    p.add_argument("--extra_pkl_1",type=str, default="/home/chen/Desktop/NewHT/HTDetection/Inference/AES/AES.pkl")
    p.add_argument("--extra_pkl_2",type=str, default="/home/chen/Desktop/NewHT/HTDetection/Inference/PIC/PIC.pkl")
    p.add_argument("--extra_pkl_3",type=str, default="/home/chen/Desktop/NewHT/HTDetection/Inference/RS232/RS232.pkl")
    #p.add_argument("--extra_pkl_4",type=str, default="/home/chen/Desktop/NewHT/HTDetection/Inference/GPT/GPT_Dataset.pkl")
    p.add_argument("--extra_pkl_4",type=str, default="/home/chen/Desktop/NewHT/HTDetection/Inference/Gemma/Gemma_Data.pkl")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim",type=int,default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2028)
    p.add_argument("--val-ratio",type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda")

    return p.parse_args()

def main():

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load Graph
    synthetic = load_synthetic_graphs(args.synthetic_ht)
    benign = load_benign_graphs(args.benign_dir)

    train_graphs, val_graphs = split_graphs(benign, synthetic, val_ratio=args.val_ratio, seed=args.seed)
    train_graphs = num_nodes(train_graphs)


    # Graph Label
    train_graphs = convert_to_graph_label(train_graphs)
    val_graphs = convert_to_graph_label(val_graphs)

    # Encoder
    in_dim = int(train_graphs[0].x.size(1))

    encoder = GINEncoderPlus(in_dim=in_dim, hidden_dim=args.hidden_dim, layers=args.num_layers,
        dropout=args.dropout, use_gate=True, device=str(device))

    #encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device))

    for p in encoder.parameters():
        p.requires_grad = True

    # Model
    model = GraphClassifier(encoder=encoder, in_dim=args.hidden_dim, hidden=args.hidden_dim, num_class=2, dropout=args.dropout).to(device)

    # Loader
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters()
         if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-3
    )

    weights = torch.tensor([6.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Training
    best_f1 = -1
    best_loss = 100
    for epoch in range(1, args.epochs + 1):
        

        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - start_time
        val_stats = evaluate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_f1={val_stats['f1']:.4f} "
            f"val_p={val_stats['precision']:.4f} "
            f"val_r={val_stats['recall']:.4f} "
            f"val_auroc={val_stats['auroc']:.4f}"
        )

        if best_f1 < val_stats['f1']:
            best_f1 = val_stats['f1']
            torch.save(
                model.state_dict(),
                os.path.join(args.out_dir, "best_classifier.pt")
            )
            print("[SAVE] best_classifier.pt")
    



if __name__ == "__main__":
    main()