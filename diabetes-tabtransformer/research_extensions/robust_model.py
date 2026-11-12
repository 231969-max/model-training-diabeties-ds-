"""Missingness-Aware & Stochastic-Corruption-Regularized TabTransformer Architecture."""
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .metrics_calibration import compute_full_evaluation, sigmoid


class RobustTabularDataset(Dataset):
    """Dataset holding standardized numericals, categorical indices, and missingness masks."""

    def __init__(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        m_num: np.ndarray,
        y: np.ndarray,
    ):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.m_num = torch.tensor(m_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "num": self.x_num[idx],
            "cat": self.x_cat[idx],
            "mask": self.m_num[idx],
            "y": self.y[idx],
        }


class RobustTabTransformer(nn.Module):
    """
    Missingness-Aware TabTransformer preserving the exact base Transformer & MLP backbone:
      - Categorical embeddings (embed_dim=32, including dedicated [MISSING] token)
      - 4-Layer Multi-Head Self-Attention Transformer (8 heads, d_ff=128, GELU, dropout=0.1)
      - Concatenation of flattened contextual embeddings + standardized numericals + optional binary missingness mask m_num
      - 2-Layer MLP Classifier [hidden -> 64 -> 32 -> 1] (ReLU, dropout=0.2)
    """

    def __init__(
        self,
        cat_cardinalities: List[int],
        num_numerical: int,
        use_missing_mask: bool = True,
        embed_dim: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,
        ff_dim: int = 128,
        transformer_dropout: float = 0.1,
        mlp_hidden_dims: List[int] = [64, 32],
        mlp_dropout: float = 0.2,
    ):
        super().__init__()
        self.num_categorical = len(cat_cardinalities)
        self.num_numerical = num_numerical
        self.use_missing_mask = use_missing_mask
        self.embed_dim = embed_dim

        if self.num_categorical > 0:
            self.cat_embeddings = nn.ModuleList(
                [nn.Embedding(num_embeddings=c, embedding_dim=embed_dim) for c in cat_cardinalities]
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            )
        else:
            self.cat_embeddings = None
            self.transformer_encoder = None

        mask_dim = num_numerical if (use_missing_mask and num_numerical > 0) else 0
        in_mlp_dim = (self.num_categorical * embed_dim) + num_numerical + mask_dim

        mlp_layers: List[nn.Module] = []
        current_dim = in_mlp_dim
        for h_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(current_dim, h_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(mlp_dropout))
            current_dim = h_dim
        mlp_layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        m_num: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = []
        if self.num_categorical > 0:
            cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            cat_stack = torch.stack(cat_embeds, dim=1)
            contextual = self.transformer_encoder(cat_stack)
            parts.append(contextual.flatten(start_dim=1))

        if self.num_numerical > 0:
            parts.append(x_num)
            if self.use_missing_mask:
                if m_num is None:
                    m_num = torch.zeros_like(x_num)
                parts.append(m_num)

        combined = torch.cat(parts, dim=1)
        return self.mlp(combined).squeeze(-1)


def apply_batch_stochastic_corruption(
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    m_num: torch.Tensor,
    missing_cat_ids: List[int],
    imputed_num_values: torch.Tensor,
    key_num_indices: List[int],
    max_mcar_rate: float = 0.30,
    max_noise_sigma: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies training-time stochastic missingness & Gaussian noise augmentation:
      - 25% of batch samples remain clean (preserving clean decision boundary)
      - 55% receive random MCAR masking (rate ~ U(0.05, max_mcar_rate)) + noise
      - 20% receive targeted drop of a key glycemic feature (e.g., hba1c or glucose_fasting) + noise
    """
    b = x_num.size(0)
    n_num = x_num.size(1)
    n_cat = x_cat.size(1)
    device = x_num.device

    x_num_out = x_num.clone()
    x_cat_out = x_cat.clone()
    m_num_out = m_num.clone()

    # Sample regime per row: [0, 0.25) clean, [0.25, 0.80) MCAR+noise, [0.80, 1.0) targeted+noise
    u = torch.rand(b, device=device)
    mcar_rows = (u >= 0.25) & (u < 0.80)
    tgt_rows = u >= 0.80

    # MCAR masking on numericals & categoricals
    if mcar_rows.any():
        row_rates = (0.05 + (max_mcar_rate - 0.05) * torch.rand(b, 1, device=device)) * mcar_rows.unsqueeze(1).float()
        if n_num > 0:
            drop_num = torch.rand(b, n_num, device=device) < row_rates
            if drop_num.any():
                m_num_out = torch.clamp(m_num_out + drop_num.float(), max=1.0)
                imp_expand = imputed_num_values.unsqueeze(0).expand_as(x_num_out)
                x_num_out = torch.where(drop_num, imp_expand, x_num_out)
        if n_cat > 0:
            drop_cat = torch.rand(b, n_cat, device=device) < row_rates
            for j, miss_id in enumerate(missing_cat_ids):
                col_drop = drop_cat[:, j]
                if col_drop.any():
                    x_cat_out[col_drop, j] = miss_id

    # Targeted masking on key glycemic markers (simulating missing HbA1c or Fasting Glucose)
    if tgt_rows.any() and len(key_num_indices) > 0:
        tgt_idx = torch.where(tgt_rows)[0]
        choice = torch.randint(0, len(key_num_indices), (len(tgt_idx),), device=device)
        for k_pos, feat_j in enumerate(key_num_indices):
            sub_rows = tgt_idx[choice == k_pos]
            if len(sub_rows) > 0:
                m_num_out[sub_rows, feat_j] = 1.0
                x_num_out[sub_rows, feat_j] = imputed_num_values[feat_j]

    # Additive Gaussian noise on observed numericals for corrupted rows
    noisy_rows = (mcar_rows | tgt_rows)
    if noisy_rows.any() and n_num > 0 and max_noise_sigma > 0.0:
        sigmas = (torch.rand(b, 1, device=device) * max_noise_sigma) * noisy_rows.unsqueeze(1).float()
        obs = 1.0 - m_num_out
        noise = torch.randn_like(x_num_out) * sigmas * obs
        x_num_out = x_num_out + noise

    return x_num_out, x_cat_out, m_num_out


def infer_robust_logits(
    model: nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    m_num: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Runs batched inference returning raw logits."""
    model.eval()
    logits_all = []
    n = len(x_num)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            t_num = torch.tensor(x_num[start:end], dtype=torch.float32, device=device)
            t_cat = torch.tensor(x_cat[start:end], dtype=torch.long, device=device)
            t_mask = torch.tensor(m_num[start:end], dtype=torch.float32, device=device)
            logits_all.append(model(t_num, t_cat, t_mask).detach().cpu().numpy())
    return np.concatenate(logits_all, axis=0)


def train_robust_tabtransformer(
    model: RobustTabTransformer,
    train_inputs: Dict[str, np.ndarray],
    y_train: np.ndarray,
    val_inputs: Dict[str, np.ndarray],
    y_val: np.ndarray,
    missing_cat_ids: List[int],
    imputed_num_values: np.ndarray,
    key_num_indices: List[int],
    stochastic_corruption: bool,
    save_path: str,
    num_epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, List[float]]:
    """Trains RobustTabTransformer with identical optimizer/scheduler hyperparameters to baseline."""
    train_ds = RobustTabularDataset(
        train_inputs["x_num"], train_inputs["x_cat"], train_inputs["m_num"], y_train
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    imp_tensor = torch.tensor(imputed_num_values, dtype=torch.float32, device=device)
    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": [],
        "val_ece": [],
        "val_brier": [],
    }

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            bx_num = batch["num"].to(device)
            bx_cat = batch["cat"].to(device)
            bm_num = batch["mask"].to(device)
            by = batch["y"].to(device)

            if stochastic_corruption:
                bx_num, bx_cat, bm_num = apply_batch_stochastic_corruption(
                    bx_num,
                    bx_cat,
                    bm_num,
                    missing_cat_ids=missing_cat_ids,
                    imputed_num_values=imp_tensor,
                    key_num_indices=key_num_indices,
                )

            optimizer.zero_grad()
            logits = model(bx_num, bx_cat, bm_num)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx_num.size(0)

        train_loss = running_loss / len(train_ds)
        val_logits = infer_robust_logits(
            model, val_inputs["x_num"], val_inputs["x_cat"], val_inputs["m_num"], device=device
        )
        val_probs = sigmoid(val_logits)
        val_ev = compute_full_evaluation(y_val, val_probs, threshold=0.50)
        val_loss = val_ev["nll"]
        scheduler.step(val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_ev["accuracy"]))
        history["val_auc"].append(float(val_ev["roc_auc"]))
        history["val_ece"].append(float(val_ev["ece"]))
        history["val_brier"].append(float(val_ev["brier_score"]))

        print(
            f"  Epoch {epoch:02d}/{num_epochs} | TrainLoss={train_loss:.4f} | "
            f"ValLoss={val_loss:.4f} | ValAcc={val_ev['accuracy']*100:.2f}% | "
            f"ValAUC={val_ev['roc_auc']:.4f} | ValECE={val_ev['ece']:.4f} | {time.time()-t0:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_ev["accuracy"],
                    "val_auc": val_ev["roc_auc"],
                },
                save_path,
            )

    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return history

