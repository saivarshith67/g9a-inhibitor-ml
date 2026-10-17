"""Full fine-tune of pretrained ChemBERTa on G9a SMILES labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from g9a_ml.metrics import binary_metrics, confusion_and_report, save_metrics_json
from g9a_ml.models.mol_splits import SplitName, split_molecule_frame


@dataclass
class ChemBERTaConfig:
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1"
    max_length: int = 128
    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    patience: int = 2
    seed: int = 42


class SmilesClsDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for imbalance."""

    def __init__(self, class_weights: torch.Tensor | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _tokenize(tokenizer, smiles: list[str], max_length: int) -> dict:
    return tokenizer(
        smiles,
        truncation=True,
        padding=True,
        max_length=max_length,
    )


def _compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = probs >= 0.5
    return binary_metrics(labels, preds.astype(int), y_score=probs)


def train_chemberta_on_frame(
    df: pd.DataFrame,
    split: SplitName,
    cfg: ChemBERTaConfig,
    smiles_col: str = "SMILES",
    target_col: str = "target",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 5,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    train_df, val_df, test_df = split_molecule_frame(
        df,
        split=split,
        smiles_col=smiles_col,
        target_col=target_col,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    train_labels = train_df[target_col].astype(int).tolist()
    val_labels = val_df[target_col].astype(int).tolist()
    test_labels = test_df[target_col].astype(int).tolist()

    train_ds = SmilesClsDataset(_tokenize(tokenizer, train_df[smiles_col].tolist(), cfg.max_length), train_labels)
    val_ds = SmilesClsDataset(_tokenize(tokenizer, val_df[smiles_col].tolist(), cfg.max_length), val_labels)
    test_ds = SmilesClsDataset(_tokenize(tokenizer, test_df[smiles_col].tolist(), cfg.max_length), test_labels)

    counts = np.bincount(np.asarray(train_labels), minlength=2).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * 2.0
    class_weights = torch.tensor(weights, dtype=torch.float32)

    if out_dir is None:
        out_dir = Path("outputs/models/chemberta") / split
    run_dir = Path(out_dir) / f"chemberta_{split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()
    ta_kwargs = dict(
        output_dir=str(run_dir / "hf_runs"),
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        logging_steps=50,
        seed=cfg.seed,
        fp16=use_fp16,
        report_to=[],
        save_total_limit=1,
        save_strategy="epoch",
    )
    # transformers renamed evaluation_strategy -> eval_strategy
    try:
        args = TrainingArguments(eval_strategy="epoch", **ta_kwargs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **ta_kwargs)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )
    trainer.train()

    pred_out = trainer.predict(test_ds)
    logits = pred_out.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    y_true = np.asarray(test_labels)
    y_pred = (probs >= 0.5).astype(int)
    metrics = binary_metrics(y_true, y_pred, y_score=probs)
    cm, report = confusion_and_report(y_true, y_pred)

    trainer.save_model(str(run_dir / "model"))
    tokenizer.save_pretrained(str(run_dir / "model"))

    result: dict[str, Any] = {
        "model": "chemberta",
        "pretrained": cfg.model_name,
        "fine_tune": "full",
        "split": split,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "class_weights": weights.tolist(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "config": asdict(cfg),
    }
    save_metrics_json(result, run_dir / f"chemberta_{split}_metrics.json")
    # also copy to metrics_dir-friendly name if parent is models
    return result
