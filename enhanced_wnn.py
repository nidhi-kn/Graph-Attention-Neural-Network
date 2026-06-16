#!/usr/bin/env python3
"""
Enhanced WNN with Advanced Training Techniques
- Enhanced Feature Engineering
- Improved WNN Architecture
- Advanced Training Techniques (Label Smoothing, Cosine Annealing)
"""

import os
import time
import numpy as np
import pandas as pd
import h5py
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import argparse
import warnings
warnings.filterwarnings('ignore')

# Ensure UTF-8 output on Windows terminals (prevents UnicodeEncodeError on emoji prints)
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


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Enhanced WNN with Advanced Training Techniques")
print(f"Using device: {device}")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


# ============================================================================
# PART 1: RANDOM ATTACK GENERATION
# ============================================================================

class RandomAttackGenerator:
    """Generate random FDIA attacks on power grid measurements"""
    
    def __init__(self, num_buses=118, attack_types=['random', 'targeted', 'coordinated', 'stealthy']):
        self.num_buses = num_buses
        self.attack_types = attack_types
        
    def generate_random_attack(self, clean_measurements, attack_type='random', severity=0.1):
        """
        Generate random attack on measurements
        
        Args:
            clean_measurements: Original measurements (N, num_buses)
            attack_type: Type of attack
            severity: Attack magnitude (0.0 to 1.0)
        
        Returns:
            attacked_measurements, attack_mask
        """
        attacked = clean_measurements.copy()
        attack_mask = np.zeros(self.num_buses)
        
        if attack_type == 'random':
            # Random bus attacks (10-30% of buses)
            num_attacked_buses = np.random.randint(int(0.1 * self.num_buses), int(0.3 * self.num_buses))
            attacked_buses = np.random.choice(self.num_buses, num_attacked_buses, replace=False)
            
            for bus in attacked_buses:
                # Add random noise
                noise = np.random.uniform(-severity, severity) * np.std(clean_measurements[:, bus])
                attacked[:, bus] += noise
                attack_mask[bus] = 1
                
        elif attack_type == 'targeted':
            # Target critical buses (high connectivity)
            num_attacked_buses = np.random.randint(5, 15)
            # Simulate targeting high-variance buses (more critical)
            bus_importance = np.std(clean_measurements, axis=0)
            critical_buses = np.argsort(bus_importance)[-num_attacked_buses:]
            
            for bus in critical_buses:
                # Larger attack on critical buses
                noise = np.random.uniform(-severity * 2, severity * 2) * np.std(clean_measurements[:, bus])
                attacked[:, bus] += noise
                attack_mask[bus] = 1
                
        elif attack_type == 'coordinated':
            # Attack connected buses together
            num_clusters = np.random.randint(2, 5)
            cluster_size = np.random.randint(5, 10)
            
            for _ in range(num_clusters):
                start_bus = np.random.randint(0, self.num_buses - cluster_size)
                cluster_buses = range(start_bus, start_bus + cluster_size)
                
                # Coordinated noise
                common_noise = np.random.uniform(-severity, severity)
                for bus in cluster_buses:
                    attacked[:, bus] += common_noise * np.std(clean_measurements[:, bus])
                    attack_mask[bus] = 1
                    
        elif attack_type == 'stealthy':
            # Small perturbations on many buses
            num_attacked_buses = np.random.randint(int(0.4 * self.num_buses), int(0.6 * self.num_buses))
            attacked_buses = np.random.choice(self.num_buses, num_attacked_buses, replace=False)
            
            for bus in attacked_buses:
                # Very small noise (harder to detect)
                noise = np.random.uniform(-severity * 0.5, severity * 0.5) * np.std(clean_measurements[:, bus])
                attacked[:, bus] += noise
                attack_mask[bus] = 1
        
        return attacked, attack_mask
    
    def augment_dataset(self, X, y, augmentation_factor=0.5):
        """
        Augment dataset with random attacks
        
        Args:
            X: Original features
            y: Original labels
            augmentation_factor: Fraction of new attack samples to add
        
        Returns:
            X_augmented, y_augmented
        """
        # Find normal samples
        normal_indices = np.where(y == 0)[0]
        num_new_attacks = int(len(normal_indices) * augmentation_factor)
        
        print(f"🎲 Generating {num_new_attacks} random attack samples...")
        
        X_new_attacks = []
        y_new_attacks = []
        
        for _ in range(num_new_attacks):
            # Randomly select a normal sample
            idx = np.random.choice(normal_indices)
            clean_sample = X[idx].reshape(1, -1)
            
            # Generate random attack
            attack_type = np.random.choice(self.attack_types)
            severity = np.random.uniform(0.05, 0.3)
            
            attacked_sample, _ = self.generate_random_attack(
                clean_sample, attack_type=attack_type, severity=severity
            )
            
            X_new_attacks.append(attacked_sample.flatten())
            y_new_attacks.append(1)  # Attack label
        
        # Combine original and augmented data
        X_augmented = np.vstack([X, np.array(X_new_attacks)])
        y_augmented = np.concatenate([y, np.array(y_new_attacks)])
        
        print(f"✅ Dataset augmented: {len(X)} → {len(X_augmented)} samples")
        print(f"   Normal: {np.sum(y_augmented == 0)}, Attack: {np.sum(y_augmented == 1)}")
        
        return X_augmented, y_augmented


# ============================================================================
# PART 2: ENHANCED DATA LOADING WITH FEATURE ENGINEERING
# ============================================================================

def load_fdia_data_enhanced(
    use_feature_selection=True,
    augment_attacks=True,
    ga_population_size=30,
    ga_generations=15,
    ga_mutation_rate=0.1,
    csv_path=None,
):
    """Load FDIA data with ALL enhancements"""
    print("📂 Loading FDIA data with enhancements...")
    
    CSV_PATH = str(csv_path) if csv_path else "fdia_project/data/data1.csv"
    MAT_PATH = "fdia_project/data/IEEE_118_system.mat"
    
    # Load CSV data
    # Prefer header-based parsing when available (data1.csv has named columns and a 'marker' label).
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Handle labels
    if 'marker' in df.columns:
        labels_raw = df['marker'].values
        df_features = df.drop(columns=['marker'])
    else:
        # Fallback for legacy no-header format
        labels_raw = df.iloc[:, -1].values
        df_features = df.iloc[:, :-1]

    if labels_raw.dtype == 'object':
        labels_str = pd.Series(labels_raw).astype(str).str.strip().str.lower().values
        uniq = set(np.unique(labels_str))
        if {'natural', 'attack'}.issubset(uniq):
            labels = (labels_str == 'attack').astype(int)
        else:
            unique_labels = np.unique(labels_raw)
            labels = np.where(labels_raw == unique_labels[0], 0, 1)
    else:
        labels = (labels_raw > 0).astype(int)

    # Convert features to numeric
    features = df_features.apply(pd.to_numeric, errors='coerce').fillna(0).values
    features = np.where(np.isinf(features), 0, features)
    features = np.where(np.isnan(features), 0, features)

    # Detect relay-based schema: columns like R1-..., R2-..., ... and log columns.
    relay_cols = []
    log_cols = []
    relay_ids = set()
    for c in df_features.columns:
        if isinstance(c, str) and c.startswith('R') and (':' in c or '-' in c):
            # Extract numeric relay id if present
            j = 1
            while j < len(c) and c[j].isdigit():
                j += 1
            if j > 1:
                relay_ids.add(int(c[1:j]))
                relay_cols.append(c)
            else:
                log_cols.append(c)
        else:
            log_cols.append(c)

    relay_ids = sorted(relay_ids)
    is_relay_schema = (len(relay_ids) > 0 and max(relay_ids) <= 10 and len(relay_ids) <= 10)

    if is_relay_schema:
        num_nodes = len(relay_ids)
        # Define per-relay feature order: group by relay id, keep stable column order.
        relay_cols_by_id = {rid: [] for rid in relay_ids}
        for c in relay_cols:
            j = 1
            while j < len(c) and c[j].isdigit():
                j += 1
            rid = int(c[1:j])
            if rid in relay_cols_by_id:
                relay_cols_by_id[rid].append(c)

        relay_feat_dim = len(relay_cols_by_id[relay_ids[0]]) if relay_ids else 0
        for rid in relay_ids:
            if len(relay_cols_by_id[rid]) != relay_feat_dim:
                raise ValueError("Relay feature columns are not consistent across relays")

        log_feat_dim = len(log_cols)
        features_per_bus = relay_feat_dim + log_feat_dim

        # Build dense per-node features: [N, num_nodes * features_per_node]
        enhanced_features = np.zeros((features.shape[0], num_nodes * features_per_bus), dtype=np.float32)
        print(f"Engineering relay features ({num_nodes} relays, {features_per_bus} features per relay)...")

        df_feat = df_features
        for i in range(df_feat.shape[0]):
            for node_idx, rid in enumerate(relay_ids):
                base = node_idx * features_per_bus
                # Relay-specific measurements
                v = df_feat.loc[df_feat.index[i], relay_cols_by_id[rid]].to_numpy(dtype=np.float32, copy=False)
                enhanced_features[i, base:base + relay_feat_dim] = v
                # Replicated global/log features per node
                if log_feat_dim:
                    lv = df_feat.loc[df_feat.index[i], log_cols].to_numpy(dtype=np.float32, copy=False)
                    enhanced_features[i, base + relay_feat_dim:base + features_per_bus] = lv

        # Use a simple fully-connected undirected relay graph (no self loops)
        src, dst = [], []
        for a in range(num_nodes):
            for b in range(num_nodes):
                if a != b:
                    src.append(a)
                    dst.append(b)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        # Load topology (118-bus case)
        with h5py.File(MAT_PATH, 'r') as f:
            branch = np.array(f['mpc_ori']['branch']).T
            src = branch[:, 0].astype(int) - 1
            dst = branch[:, 1].astype(int) - 1

        edge_index = torch.tensor(
            np.vstack([
                np.concatenate([src, dst]),
                np.concatenate([dst, src])
            ]), dtype=torch.long
        )

        # ENHANCED FEATURE ENGINEERING - 10 features per bus
        num_nodes = 118
        features_per_bus = 10

        enhanced_features = np.zeros((features.shape[0], num_nodes * features_per_bus))

        print("Engineering enhanced features (10 per bus)...")

        for i in range(features.shape[0]):
            sample = features[i]

            # Global statistics
            global_mean = np.mean(sample)
            global_std = np.std(sample) if np.std(sample) > 0 else 1.0
            global_max = np.max(sample)
            global_min = np.min(sample)
            global_median = np.median(sample)

            for bus in range(num_nodes):
                base_idx = bus * features_per_bus

                # Original per-bus value
                raw_val = sample[bus] if bus < len(sample) else 0.0

                # Feature 1: Raw value
                enhanced_features[i, base_idx] = raw_val

                # Feature 2: Z-score normalized
                enhanced_features[i, base_idx + 1] = (raw_val - global_mean) / global_std

                # Feature 3: Min-max normalized
                if (global_max - global_min) > 0:
                    enhanced_features[i, base_idx + 2] = (raw_val - global_min) / (global_max - global_min)
                else:
                    enhanced_features[i, base_idx + 2] = 0.5

                # Feature 4: Deviation from median
                enhanced_features[i, base_idx + 3] = (raw_val - global_median) / (global_std + 1e-6)

                # Feature 5: Position encoding (sin)
                enhanced_features[i, base_idx + 4] = np.sin(2 * np.pi * bus / num_nodes)

                # Feature 6: Position encoding (cos)
                enhanced_features[i, base_idx + 5] = np.cos(2 * np.pi * bus / num_nodes)

                # Feature 7: Anomaly indicator (2-sigma)
                enhanced_features[i, base_idx + 6] = 1.0 if abs(raw_val - global_mean) > 2 * global_std else 0.0

                # Feature 8: Anomaly indicator (3-sigma)
                enhanced_features[i, base_idx + 7] = 1.0 if abs(raw_val - global_mean) > 3 * global_std else 0.0

                # Feature 9: Squared value (capture non-linearity)
                enhanced_features[i, base_idx + 8] = (raw_val ** 2) / (global_std ** 2 + 1e-6)

                # Feature 10: Relative position in distribution
                enhanced_features[i, base_idx + 9] = (raw_val < global_median) * 1.0
    
    # Final sanitization (prevents scaler overflow / inf propagation)
    enhanced_features = np.nan_to_num(
        enhanced_features.astype(np.float64, copy=False),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    enhanced_features = np.clip(enhanced_features, -1e9, 1e9).astype(np.float32, copy=False)

    print(f"Enhanced features created: {enhanced_features.shape}")
    print(f"   Original class distribution - Normal: {np.sum(labels==0)}, Attack: {np.sum(labels==1)}")
    
    # RANDOM ATTACK AUGMENTATION
    if augment_attacks:
        # Use full feature dimension here so the attack generator
        # operates over all enhanced features without index errors.
        # (Each \"bus\" in RandomAttackGenerator now corresponds to one
        # feature dimension of the enhanced feature vector.)
        attack_gen = RandomAttackGenerator(num_buses=enhanced_features.shape[1])
        enhanced_features, labels = attack_gen.augment_dataset(
            enhanced_features, labels, augmentation_factor=0.3
        )
    
    return enhanced_features, labels, edge_index, features_per_bus, None


class EnhancedWNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=384, dropout=0.3, num_heads=8):
        super().__init__()
        self.dropout = float(dropout)

        self.in_lin = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.gat1 = GATConv(hidden_dim, hidden_dim // 2, heads=2, dropout=self.dropout)
        self.gat2 = GATConv((hidden_dim // 2) * 2, hidden_dim, heads=1, dropout=self.dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=self.dropout)
        self.gat4 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=self.dropout)
        self.gat_norm = nn.LayerNorm(hidden_dim)

        self.graphon_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=self.dropout, batch_first=True
        )

        self.post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, batch):
        x = self.in_lin(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat3(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat4(x, edge_index))
        x = self.gat_norm(x)

        x_dense, mask = to_dense_batch(x, batch)
        key_padding_mask = ~mask
        att_x, _ = self.graphon_attention(
            x_dense,
            x_dense,
            x_dense,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x_att = att_x[mask]
        x_att = self.post(x_att)

        pooled = global_mean_pool(x_att, batch)
        return self.classifier(pooled)


def _build_node_u(num_nodes: int, device=None):
    u = torch.linspace(0.0, 1.0, steps=int(num_nodes), dtype=torch.float32)
    u = u.view(-1, 1)
    if device is not None:
        u = u.to(device)
    return u


class GraphonKernelLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.kernel_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

    def forward(self, x_dense, u_dense, mask):
        bsz, n, h = x_dense.shape
        ui = u_dense.unsqueeze(2).expand(bsz, n, n, 1)
        uj = u_dense.unsqueeze(1).expand(bsz, n, n, 1)
        du = torch.abs(ui - uj)
        k_in = torch.cat([ui, uj, du], dim=-1)
        w = self.kernel_mlp(k_in).squeeze(-1)

        valid = mask
        valid_i = valid.unsqueeze(2)
        valid_j = valid.unsqueeze(1)
        w = w.masked_fill(~(valid_i & valid_j), float('-inf'))
        w = F.softmax(w, dim=-1)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        x_next = torch.bmm(w, x_dense)
        x_next = self.out(x_next)
        return x_next


class GraphonWNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.inp = nn.Sequential(
            nn.Linear(int(input_dim), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.layers = nn.ModuleList([
            GraphonKernelLayer(self.hidden_dim, dropout=self.dropout) for _ in range(int(num_layers))
        ])
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, batch, u=None):
        x = self.inp(x)
        x_dense, mask = to_dense_batch(x, batch)

        if u is None:
            num_nodes = int(x_dense.shape[1])
            u = _build_node_u(num_nodes, device=x_dense.device)
            u_dense = u.view(1, num_nodes, 1).expand(x_dense.shape[0], num_nodes, 1)
        else:
            u = u.to(x_dense.device)
            if u.dim() == 2 and u.size(0) == x.size(0):
                u_dense, _ = to_dense_batch(u, batch)
            elif u.dim() == 2 and u.size(0) == x_dense.size(1):
                u_dense = u.view(1, u.size(0), 1).expand(x_dense.shape[0], u.size(0), 1)
            else:
                u_dense = u

        for layer in self.layers:
            x_dense = x_dense + layer(x_dense, u_dense, mask)

        x_dense = self.norm(x_dense)
        x_out = x_dense[mask]
        pooled = global_mean_pool(x_out, batch)
        return self.classifier(pooled)


def run_ablation(seed=42, quick=False):
    set_seed(seed)
    print('Ablation mode is not configured in this simplified version.')
    return None


def create_graph_dataset(X, y, edge_index, features_per_bus, num_nodes=None):
    """Create PyTorch Geometric dataset"""
    data_list = []
    if num_nodes is None:
        if X.shape[1] % features_per_bus != 0:
            raise ValueError("X feature dimension is not divisible by features_per_bus")
        num_nodes = X.shape[1] // features_per_bus
    
    for i in range(len(X)):
        node_features = X[i].reshape(num_nodes, features_per_bus)
        data = Data(
            x=torch.FloatTensor(node_features),
            y=torch.LongTensor([y[i]]),
            edge_index=edge_index
        )
        data_list.append(data)
    
    return data_list


def train_enhanced_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    epochs=40,
    lr=0.001,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=12,
):
    """Enhanced training with better techniques"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Weighted loss with label smoothing
    weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(label_smoothing))
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    best_val_acc = 0
    patience = int(patience)
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        val_acc = balanced_accuracy_score(all_labels, all_preds)
        
        scheduler.step()
        
        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                  f"Val Acc = {val_acc:.4f}")
    
    return model


def evaluate_model_enhanced(model, test_loader):
    """Baseline evaluation with argmax decision rule (kept for compatibility)."""
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    normal_recall = recall_score(all_labels, all_preds, pos_label=0, zero_division=0.0)
    attack_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0.0)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        roc_auc = 0.5
    
    return balanced_acc, normal_recall, attack_recall, roc_auc, f1


def find_best_threshold_enhanced(model, val_loader):
    """Find probability threshold that maximizes balanced accuracy on validation data."""
    model.eval()
    model = model.to(device)
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    if not all_labels:
        return 0.5
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_t = 0.5
    best_bal_acc = 0.0
    
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (all_probs >= t).astype(int)
        bal_acc = balanced_accuracy_score(all_labels, preds)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_t = t
    
    return best_t


def evaluate_model_with_threshold_enhanced(model, test_loader, threshold=0.5):
    """Evaluate model using a tunable decision threshold on attack probability."""
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(out, dim=1)
            attack_prob = probs[:, 1]
            preds = (attack_prob >= threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    normal_recall = recall_score(all_labels, all_preds, pos_label=0, zero_division=0.0)
    attack_recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0.0)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        roc_auc = 0.5
    
    return balanced_acc, normal_recall, attack_recall, roc_auc, f1


# ============================================================================
# PART 6: MAIN EXECUTION WITH ALL IMPROVEMENTS
# ============================================================================

def main():
    """Main execution with all enhancements"""
    
    print("="*80)
    print("🚀 ENHANCED WNN WITH ADVANCED TRAINING TECHNIQUES")
    print("="*80)
    print("\nEnhancements:")
    print("✅ 1. Random Attack Generation & Data Augmentation")
    print("✅ 2. Enhanced Feature Engineering")
    print("✅ 3. Improved WNN Architecture with Residual Connections")
    print("✅ 4. Advanced Training Techniques (Label Smoothing, Cosine Annealing)")
    print("="*80)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-aug', action='store_true')
    args = parser.parse_args()

    if args.ablation:
        run_ablation(seed=args.seed, quick=args.quick)
        return None

    metrics = run_single_experiment(
        seed=args.seed,
        augment_attacks=(not args.no_aug),
        use_feature_selection=False,
        use_pso=False,
        tune_threshold=True,
        quick=args.quick,
        return_model=True,
    )

    bal_acc = metrics['balanced_acc']
    normal_recall = metrics['normal_recall']
    attack_recall = metrics['attack_recall']
    roc_auc = metrics['roc_auc']
    f1 = metrics['f1']
    trained_model = metrics.get('model', None)
    
    # Print results
    print("\n" + "="*80)
    print("🏆 ENHANCED WNN FINAL RESULTS")
    print("="*80)
    print(f"Balanced Accuracy:  {bal_acc:.4f} ({bal_acc*100:.2f}%)")
    print(f"Normal Recall:      {normal_recall:.4f} ({normal_recall*100:.2f}%)")
    print(f"Attack Recall:      {attack_recall:.4f} ({attack_recall*100:.2f}%)")
    print(f"ROC AUC Score:      {roc_auc:.4f}")
    print(f"F1 Score (Weighted):{f1:.4f}")
    print("="*80)
    
    # Save results
    results_df = pd.DataFrame({
        'Metric': ['Balanced Accuracy', 'Normal Recall', 'Attack Recall', 'ROC AUC', 'F1 Score'],
        'Value': [bal_acc, normal_recall, attack_recall, roc_auc, f1]
    })
    
    results_df.to_csv('enhanced_wnn_results.csv', index=False)
    print("\n💾 Results saved to 'enhanced_wnn_results.csv'")
    
    # Save model
    if trained_model is not None:
        torch.save(trained_model.state_dict(), 'enhanced_wnn_model.pth')
        print("💾 Model saved to 'enhanced_wnn_model.pth'")

    print("\n✅ Enhancement Complete!")
    print(f"\n📈 IMPROVEMENT SUMMARY:")
    print(f"   Previous Balanced Accuracy: ~73.2%")
    print(f"   New Balanced Accuracy:      {bal_acc*100:.2f}%")
    print(f"   Improvement:                +{(bal_acc - 0.732)*100:.2f}%")
    
    return results_df


if __name__ == "__main__":
    main()