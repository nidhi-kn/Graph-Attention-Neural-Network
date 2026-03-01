#!/usr/bin/env python3

import argparse
import sys
import time
import numpy as np
import pandas as pd
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import WeightedRandomSampler

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Re-use your enhanced pipeline + ENGAT (Enhanced GAT) model
import enhanced_wnn


def _resolve_csv_path(name_or_path: Optional[str]):
    if not name_or_path:
        return None
    s = str(name_or_path)
    if s.lower().endswith('.csv') and (':' in s or s.startswith('\\') or s.startswith('/') or s.startswith('.')):
        return s
    if s.lower().startswith('data') and s.lower().endswith('.csv'):
        return f"fdia_project/data/{s}"
    if s.lower().startswith('data') and s[4:].isdigit():
        return f"fdia_project/data/{s}.csv"
    return f"fdia_project/data/{s}"


def _ensure_utf8_stdout():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def safe_stratified_split(X, y, seed=42):
    y = np.asarray(y)
    if np.unique(y).size < 2:
        raise ValueError(
            'Only one class present in labels; enable attack augmentation or provide attack samples.'
        )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_graph_dataset(X, y, edge_index, features_per_bus, num_nodes=None):
    data_list = []
    if num_nodes is None:
        if X.shape[1] % features_per_bus != 0:
            raise ValueError('X feature dimension is not divisible by features_per_bus')
        num_nodes = X.shape[1] // features_per_bus
    for i in range(len(X)):
        node_features = X[i].reshape(num_nodes, features_per_bus)
        data_list.append(
            Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                y=torch.tensor([int(y[i])], dtype=torch.long),
                edge_index=edge_index,
            )
        )
    return data_list


def _make_weighted_sampler(y):
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    class_weight = {int(c): 1.0 / float(n) for c, n in zip(classes, counts)}
    sample_weights = np.array([class_weight[int(yi)] for yi in y], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden)
        self.c2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden=64, heads=4, dropout=0.2):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout)
        self.c2 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout)
        self.lin = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.elu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.c2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.2):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.lin = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.c1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.c2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


def train_graph_model(model, train_loader, val_loader, class_weights, epochs=30, lr=1e-3, patience=6):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best_state = None
    best_val = -1
    bad = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                p = out.argmax(dim=1)
                preds.extend(p.cpu().numpy())
                labels.extend(batch.y.cpu().numpy())

        val_bal = balanced_accuracy_score(labels, preds)
        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_mlp(model, train_loader, val_loader, class_weights, epochs=30, lr=1e-3, patience=6):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    best_state = None
    best_val = -1
    bad = 0

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                p = out.argmax(dim=1)
                preds.extend(p.cpu().numpy())
                labels.extend(y.numpy())

        val_bal = balanced_accuracy_score(labels, preds)
        if val_bal > best_val:
            best_val = val_bal
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _find_best_threshold_for_balanced_acc(labels, probs, grid_size=201):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0:
        return 0.5, float('nan')

    thresholds = np.linspace(0.0, 1.0, int(grid_size))
    best_t = 0.5
    best_bal = -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        bal = balanced_accuracy_score(labels, preds)
        if bal > best_bal:
            best_bal = bal
            best_t = float(t)
    return best_t, float(best_bal)


def _predict_probs_graph(model, loader):
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pr = F.softmax(out, dim=1)[:, 1]
            labels.extend(batch.y.cpu().numpy())
            probs.extend(pr.cpu().numpy())
    return np.asarray(labels).astype(int), np.asarray(probs, dtype=np.float64)


def _predict_probs_mlp(model, loader):
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            pr = F.softmax(out, dim=1)[:, 1]
            labels.extend(y.numpy())
            probs.extend(pr.cpu().numpy())
    return np.asarray(labels).astype(int), np.asarray(probs, dtype=np.float64)


def _compute_metrics_from_probs(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= float(threshold)).astype(int)
    bal = balanced_accuracy_score(labels, preds)
    nr = recall_score(labels, preds, pos_label=0, zero_division=0.0)
    ar = recall_score(labels, preds, pos_label=1, zero_division=0.0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float('nan')
    return float(bal), float(nr), float(ar), float(auc)


def evaluate_graph_model(model, test_loader, threshold=0.5):
    labels, probs = _predict_probs_graph(model, test_loader)
    return _compute_metrics_from_probs(labels, probs, threshold=threshold)


def evaluate_mlp(model, test_loader, threshold=0.5):
    labels, probs = _predict_probs_mlp(model, test_loader)
    return _compute_metrics_from_probs(labels, probs, threshold=threshold)


def run(
    seed=42,
    quick=False,
    full_wnn=False,
    wnn_variant='enhanced',
    train_csv=None,
    test_csv=None,
    wnn_hidden_dim=None,
    wnn_num_layers=None,
    wnn_dropout=None,
    wnn_epochs=None,
    wnn_lr=None,
    wnn_weight_decay=None,
    wnn_label_smoothing=None,
    wnn_patience=None,
):
    # Data: data1.csv is attack-majority; generating more synthetic attacks would
    # usually hurt balanced accuracy. We balance training via sampling instead.
    train_csv = _resolve_csv_path(train_csv) or "fdia_project/data/data1.csv"
    X, y, edge_index, features_per_bus, _ = enhanced_wnn.load_fdia_data_enhanced(
        use_feature_selection=(full_wnn),
        augment_attacks=False,
        ga_population_size=(16 if quick else 30),
        ga_generations=(6 if quick else 15),
        ga_mutation_rate=(0.12 if quick else 0.1),
        csv_path=train_csv,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = safe_stratified_split(X, y, seed=seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

    num_nodes = X_train.shape[1] // features_per_bus

    # Caching datasets per seed
    train_graph = create_graph_dataset(X_train, y_train, edge_index, features_per_bus, num_nodes=num_nodes)
    val_graph = create_graph_dataset(X_val, y_val, edge_index, features_per_bus, num_nodes=num_nodes)
    test_graph = create_graph_dataset(X_test, y_test, edge_index, features_per_bus, num_nodes=num_nodes)

    batch_size = 32 if quick else 48
    graph_sampler = _make_weighted_sampler(y_train)
    train_loader = DataLoader(train_graph, batch_size=batch_size, sampler=graph_sampler)
    val_loader = DataLoader(val_graph, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graph, batch_size=batch_size, shuffle=False)

    # MLP uses flattened engineered features
    mlp_sampler = _make_weighted_sampler(y_train)
    mlp_train = TorchDataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        sampler=mlp_sampler,
    )
    mlp_val = TorchDataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    mlp_test = TorchDataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )

    results = []

    # ENGAT (Enhanced GAT) - train directly using the already-created loaders.
    # This avoids huge overhead from re-loading + re-feature-engineering.
    wnn_start = time.time()
    wnn_hidden_dim = int(wnn_hidden_dim) if wnn_hidden_dim is not None else None
    wnn_num_layers = int(wnn_num_layers) if wnn_num_layers is not None else None
    wnn_dropout = float(wnn_dropout) if wnn_dropout is not None else None
    wnn_epochs = int(wnn_epochs) if wnn_epochs is not None else None
    wnn_lr = float(wnn_lr) if wnn_lr is not None else None
    wnn_weight_decay = float(wnn_weight_decay) if wnn_weight_decay is not None else None
    wnn_label_smoothing = float(wnn_label_smoothing) if wnn_label_smoothing is not None else None
    wnn_patience = int(wnn_patience) if wnn_patience is not None else None

    if str(wnn_variant).lower() == 'graphon':
        wnn_model = enhanced_wnn.GraphonWNN(
            input_dim=features_per_bus,
            hidden_dim=(wnn_hidden_dim if wnn_hidden_dim is not None else (256 if quick else 320)),
            dropout=(wnn_dropout if wnn_dropout is not None else 0.2),
            num_layers=(wnn_num_layers if wnn_num_layers is not None else (3 if quick else 4)),
        )
        wnn_name = 'GraphonWNN'
    else:
        wnn_model = enhanced_wnn.EnhancedWNN(
            input_dim=features_per_bus,
            hidden_dim=(wnn_hidden_dim if wnn_hidden_dim is not None else 384),
            dropout=(wnn_dropout if wnn_dropout is not None else 0.3),
            num_heads=8,
        )
        wnn_name = 'ENGAT'
    wnn_model = enhanced_wnn.train_enhanced_model(
        wnn_model,
        train_loader,
        val_loader,
        class_weights,
        epochs=(wnn_epochs if wnn_epochs is not None else (20 if quick else 40)),
        lr=(wnn_lr if wnn_lr is not None else 1e-3),
        weight_decay=(wnn_weight_decay if wnn_weight_decay is not None else 1e-4),
        label_smoothing=(wnn_label_smoothing if wnn_label_smoothing is not None else 0.1),
        patience=(wnn_patience if wnn_patience is not None else (6 if quick else 12)),
    )
    y_val_prob, p_val_prob = _predict_probs_graph(wnn_model, val_loader)
    wnn_thr, _ = _find_best_threshold_for_balanced_acc(y_val_prob, p_val_prob)

    if test_csv:
        test_csv = _resolve_csv_path(test_csv)
        Xt, yt, edge_t, fpb_t, _ = enhanced_wnn.load_fdia_data_enhanced(
            use_feature_selection=False,
            augment_attacks=False,
            ga_population_size=1,
            ga_generations=1,
            ga_mutation_rate=0.0,
            csv_path=test_csv,
        )
        if int(fpb_t) != int(features_per_bus):
            raise ValueError('Train/test features_per_bus mismatch; datasets have different schema.')
        Xt = scaler.transform(Xt)
        num_nodes_t = Xt.shape[1] // features_per_bus
        test_graph_t = create_graph_dataset(Xt, yt, edge_t, features_per_bus, num_nodes=num_nodes_t)
        test_loader_t = DataLoader(test_graph_t, batch_size=batch_size, shuffle=False)
        y_test_prob, p_test_prob = _predict_probs_graph(wnn_model, test_loader_t)
        bal, nr, ar, auc = _compute_metrics_from_probs(y_test_prob, p_test_prob, threshold=wnn_thr)
    else:
        y_test_prob, p_test_prob = _predict_probs_graph(wnn_model, test_loader)
        bal, nr, ar, auc = _compute_metrics_from_probs(y_test_prob, p_test_prob, threshold=wnn_thr)
    wnn_time = (time.time() - wnn_start) * 1000.0
    results.append(
        {
            'model': wnn_name,
            'balanced_acc': bal,
            'normal_recall': nr,
            'attack_recall': ar,
            'roc_auc': auc,
            'train_time_ms': wnn_time,
        }
    )

    # Baselines
    epochs = 20 if quick else 35
    patience = 5 if quick else 7

    mlp = MLP(input_dim=X_train.shape[1], hidden=(192 if quick else 256), dropout=0.2)
    t0 = time.time()
    mlp = train_mlp(mlp, mlp_train, mlp_val, class_weights, epochs=epochs, lr=1e-3, patience=patience)
    mlp_t = (time.time() - t0) * 1000.0
    y_val_prob, p_val_prob = _predict_probs_mlp(mlp, mlp_val)
    mlp_thr, _ = _find_best_threshold_for_balanced_acc(y_val_prob, p_val_prob)
    bal, nr, ar, auc = evaluate_mlp(mlp, mlp_test, threshold=mlp_thr)
    results.append(
        {'model': 'MLP', 'balanced_acc': bal, 'normal_recall': nr, 'attack_recall': ar, 'roc_auc': auc, 'train_time_ms': mlp_t}
    )

    gcn = GCN(in_dim=features_per_bus, hidden=(48 if quick else 64), dropout=0.2)
    t0 = time.time()
    gcn = train_graph_model(gcn, train_loader, val_loader, class_weights, epochs=epochs, lr=1e-3, patience=patience)
    gcn_t = (time.time() - t0) * 1000.0
    y_val_prob, p_val_prob = _predict_probs_graph(gcn, val_loader)
    gcn_thr, _ = _find_best_threshold_for_balanced_acc(y_val_prob, p_val_prob)
    bal, nr, ar, auc = evaluate_graph_model(gcn, test_loader, threshold=gcn_thr)
    results.append(
        {'model': 'GCN', 'balanced_acc': bal, 'normal_recall': nr, 'attack_recall': ar, 'roc_auc': auc, 'train_time_ms': gcn_t}
    )

    gat = GAT(in_dim=features_per_bus, hidden=(48 if quick else 64), heads=4, dropout=0.2)
    t0 = time.time()
    gat = train_graph_model(gat, train_loader, val_loader, class_weights, epochs=epochs, lr=1e-3, patience=patience)
    gat_t = (time.time() - t0) * 1000.0
    y_val_prob, p_val_prob = _predict_probs_graph(gat, val_loader)
    gat_thr, _ = _find_best_threshold_for_balanced_acc(y_val_prob, p_val_prob)
    bal, nr, ar, auc = evaluate_graph_model(gat, test_loader, threshold=gat_thr)
    results.append(
        {'model': 'GAT', 'balanced_acc': bal, 'normal_recall': nr, 'attack_recall': ar, 'roc_auc': auc, 'train_time_ms': gat_t}
    )

    sage = GraphSAGE(in_dim=features_per_bus, hidden=(48 if quick else 64), dropout=0.2)
    t0 = time.time()
    sage = train_graph_model(sage, train_loader, val_loader, class_weights, epochs=epochs, lr=1e-3, patience=patience)
    sage_t = (time.time() - t0) * 1000.0
    y_val_prob, p_val_prob = _predict_probs_graph(sage, val_loader)
    sage_thr, _ = _find_best_threshold_for_balanced_acc(y_val_prob, p_val_prob)
    bal, nr, ar, auc = evaluate_graph_model(sage, test_loader, threshold=sage_thr)
    results.append(
        {'model': 'GraphSAGE', 'balanced_acc': bal, 'normal_recall': nr, 'attack_recall': ar, 'roc_auc': auc, 'train_time_ms': sage_t}
    )

    return results


def main():
    _ensure_utf8_stdout()

    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--quick', action='store_true')
    p.add_argument('--full-wnn', action='store_true', help='Enable GA (and PSO unless --quick) for ENGAT. Slower.')
    p.add_argument('--wnn-variant', type=str, default='enhanced', choices=['enhanced', 'graphon'])
    p.add_argument('--train-data', type=str, default='data1.csv', help='Training CSV (e.g. data1.csv or full path).')
    p.add_argument('--test-data', type=str, default='', help='Optional test CSV for transfer (e.g. data2.csv).')
    p.add_argument('--wnn-hidden-dim', type=int, default=0, help='Override ENGAT hidden dim (0 = default).')
    p.add_argument('--wnn-num-layers', type=int, default=0, help='Override GraphonWNN num_layers (0 = default).')
    p.add_argument('--wnn-dropout', type=float, default=-1.0, help='Override ENGAT dropout (-1 = default).')
    p.add_argument('--wnn-epochs', type=int, default=0, help='Override ENGAT training epochs (0 = default).')
    p.add_argument('--wnn-lr', type=float, default=0.0, help='Override ENGAT learning rate (0 = default).')
    p.add_argument('--wnn-weight-decay', type=float, default=-1.0, help='Override ENGAT weight decay (-1 = default).')
    p.add_argument('--wnn-label-smoothing', type=float, default=-1.0, help='Override label smoothing (-1 = default).')
    p.add_argument('--wnn-patience', type=int, default=0, help='Override early stopping patience (0 = default).')
    p.add_argument('--out', type=str, default='baseline_comparison.csv')
    args = p.parse_args()

    all_rows = []
    for seed in args.seeds:
        set_seed(seed)
        print('\n' + '=' * 80)
        print(f'Running baseline comparison (seed={seed})')
        print('=' * 80)

        rows = run(
            seed=seed,
            quick=args.quick,
            full_wnn=args.full_wnn,
            wnn_variant=args.wnn_variant,
            train_csv=args.train_data,
            test_csv=(args.test_data or None),
            wnn_hidden_dim=(args.wnn_hidden_dim if args.wnn_hidden_dim and args.wnn_hidden_dim > 0 else None),
            wnn_num_layers=(args.wnn_num_layers if args.wnn_num_layers and args.wnn_num_layers > 0 else None),
            wnn_dropout=(args.wnn_dropout if args.wnn_dropout is not None and args.wnn_dropout >= 0.0 else None),
            wnn_epochs=(args.wnn_epochs if args.wnn_epochs and args.wnn_epochs > 0 else None),
            wnn_lr=(args.wnn_lr if args.wnn_lr and args.wnn_lr > 0.0 else None),
            wnn_weight_decay=(args.wnn_weight_decay if args.wnn_weight_decay is not None and args.wnn_weight_decay >= 0.0 else None),
            wnn_label_smoothing=(args.wnn_label_smoothing if args.wnn_label_smoothing is not None and args.wnn_label_smoothing >= 0.0 else None),
            wnn_patience=(args.wnn_patience if args.wnn_patience and args.wnn_patience > 0 else None),
        )
        for r in rows:
            r['seed'] = seed
            all_rows.append(r)

    df = pd.DataFrame(all_rows)

    # Aggregate mean/std
    agg = df.groupby('model').agg(
        balanced_acc_mean=('balanced_acc', 'mean'),
        balanced_acc_std=('balanced_acc', 'std'),
        normal_recall_mean=('normal_recall', 'mean'),
        normal_recall_std=('normal_recall', 'std'),
        attack_recall_mean=('attack_recall', 'mean'),
        attack_recall_std=('attack_recall', 'std'),
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        train_time_ms_mean=('train_time_ms', 'mean'),
        train_time_ms_std=('train_time_ms', 'std'),
    ).reset_index()

    # Pretty columns
    def pm(mean, std, digits=3):
        if np.isnan(std):
            std = 0.0
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

    table = pd.DataFrame(
        {
            'Model': agg['model'],
            'Balanced Acc': [pm(m, s, 3) for m, s in zip(agg['balanced_acc_mean'], agg['balanced_acc_std'])],
            'Normal Recall': [pm(m, s, 3) for m, s in zip(agg['normal_recall_mean'], agg['normal_recall_std'])],
            'Attack Recall': [pm(m, s, 3) for m, s in zip(agg['attack_recall_mean'], agg['attack_recall_std'])],
            'ROC AUC': [pm(m, s, 3) for m, s in zip(agg['roc_auc_mean'], agg['roc_auc_std'])],
            'Train Time (ms)': [pm(m, s, 1) for m, s in zip(agg['train_time_ms_mean'], agg['train_time_ms_std'])],
        }
    )

    df.to_csv(args.out.replace('.csv', '_raw.csv'), index=False)
    table.to_csv(args.out, index=False)

    print('\n' + '=' * 80)
    print('Baseline comparison summary (mean ± std)')
    print(table.to_string(index=False))
    print('=' * 80)
    print(f"Saved summary: {args.out}")
    print(f"Saved raw: {args.out.replace('.csv', '_raw.csv')}")


if __name__ == '__main__':
    main()
