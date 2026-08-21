"""Fine-tune ParsBERT fold models for continuous stress regression."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class StressDataset(Dataset):
    def __init__(self, texts, labels, weights):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.weights = np.asarray(weights, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        item["sample_weight"] = torch.tensor(self.weights[idx], dtype=torch.float32)
        return item


def make_loader(texts, labels, weights, tokenizer, max_length, batch_size, shuffle):
    ds = StressDataset(texts, labels, weights)
    ds.tokenizer = tokenizer
    ds.max_length = max_length
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_model(model_name: str, device):
    m = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1, problem_type="regression"
    )
    return m.to(device)


def weighted_asymmetric_mse(preds, targets, sample_weights, under_prediction_penalty=1.75):
    error = preds - targets
    direction_penalty = torch.where(
        error < 0,
        torch.full_like(error, under_prediction_penalty),
        torch.ones_like(error),
    )
    return (sample_weights * direction_penalty * (error**2)).mean()


def train_fold(
    train_texts,
    train_labels,
    train_weights,
    val_texts,
    val_labels,
    model_name,
    tokenizer,
    max_length,
    batch_size,
    learning_rate,
    weight_decay,
    epochs,
    early_stopping_patience,
    under_prediction_penalty,
    seed,
    device,
    save_dir,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tr_loader = make_loader(
        train_texts, train_labels, train_weights, tokenizer, max_length, batch_size, shuffle=True
    )
    val_loader = make_loader(
        val_texts, val_labels, np.ones(len(val_labels), dtype=np.float32),
        tokenizer, max_length, batch_size, shuffle=False,
    )

    model = build_model(model_name, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(tr_loader) * epochs
    warmup = max(1, int(0.1 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )

    best_mae = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        for batch in tr_loader:
            labels = batch.pop("labels").to(device)
            weights = batch.pop("sample_weight").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            preds = out.logits.squeeze(-1)
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
            loss = weighted_asymmetric_mse(preds, labels, weights, under_prediction_penalty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # eval
        model.eval()
        val_preds, val_labs = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                p = out.logits.squeeze(-1)
                if p.ndim == 0:
                    p = p.unsqueeze(0)
                val_preds.append(p.cpu().numpy())
                val_labs.append(labels.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_labs = np.concatenate(val_labs)
        mae_v = float(np.mean(np.abs(val_preds - val_labs)))
        history.append({"epoch": ep, "val_mae": mae_v})
        print(f"  ep {ep}/{epochs} val_mae={mae_v:.4f}")

        if mae_v < best_mae - 1e-4:
            best_mae = mae_v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print("  early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir / "tokenizer"))
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(save_dir / "fold_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "max_length": max_length,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "early_stopping_patience": early_stopping_patience,
                "under_prediction_penalty": under_prediction_penalty,
                "seed": seed,
                "best_val_mae": best_mae,
            },
            f,
            indent=2,
        )
    return {"best_val_mae": best_mae, "history": history}
