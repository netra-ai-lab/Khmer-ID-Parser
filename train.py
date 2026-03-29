import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import KhmerIDParserModel
from dataset import build_vocab, KhmerIDDataset, collate_fn

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_GRAD_NORM = 5.0

EMBED_DIM = 64
CNN_FILTERS = 64
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
DROPOUT = 0.3

WEIGHT_DECAY = 0.01

# How often (in steps) to record a step-level loss snapshot.
LOG_EVERY = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)
METRICS_PATH = "checkpoints/training_metrics.json"


# ── Metrics helpers ────────────────────────────────────────────────────────────

def _load_metrics() -> dict:
    """Load existing metrics file so resumed runs append rather than overwrite."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {
        "step_losses": [],   # {global_step, epoch, step_in_epoch, train_loss, lr, elapsed_s}
        "epoch_stats": [],   # {epoch, avg_train_loss, avg_val_loss, val_f1, val_precision, val_recall, lr, elapsed_s}
    }


def _save_metrics(metrics: dict) -> None:
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# ── Span-level F1 ─────────────────────────────────────────────────────────────

def _compute_f1(pred_batch: list, gold_batch: list, idx2tag: dict) -> dict:
    """
    Span-level micro F1.
    A predicted span is correct only when start, end, and entity type all match.
    """
    tp = fp = fn = 0

    for pred_ids, gold_ids in zip(pred_batch, gold_batch):
        def spans(seq):
            out = set(); i = 0
            while i < len(seq):
                t = idx2tag.get(int(seq[i]), "O")
                if t.startswith("B-"):
                    ent = t[2:]; j = i + 1
                    while j < len(seq) and idx2tag.get(int(seq[j]), "O") == f"I-{ent}":
                        j += 1
                    out.add((i, j, ent)); i = j
                else:
                    i += 1
            return out

        ps, gs = spans(pred_ids), spans(gold_ids)
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


@torch.no_grad()
def _evaluate(model, val_loader, idx2tag, device) -> tuple:
    """Return (avg_val_loss, f1_dict)."""
    model.eval()
    total_loss = 0
    pred_all, gold_all = [], []

    for chars, tags, mask in val_loader:
        chars, tags, mask = chars.to(device), tags.to(device), mask.to(device)
        total_loss += model(chars, tags, mask).item()

        preds = model.decode(chars, mask)
        for i, pred_seq in enumerate(preds):
            real_len = int(mask[i].sum().item())
            pred_all.append(pred_seq[:real_len])
            gold_all.append(tags[i, :real_len].tolist())

    avg_loss = total_loss / len(val_loader)
    f1_dict  = _compute_f1(pred_all, gold_all, idx2tag)
    return avg_loss, f1_dict


# ── Save vocab ────────────────────────────────────────────────────────────────

def save_vocab(vocab_dict, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print(f"Using device: {DEVICE}")

    # 1. Build vocabularies
    char_vocab, tag_vocab = build_vocab("train_clean.jsonl")
    save_vocab(char_vocab.char2idx, "char_to_idx.json")
    save_vocab(tag_vocab.tag2idx,   "tag_to_idx.json")
    idx2tag = {v: k for k, v in tag_vocab.tag2idx.items()}

    # 2. Datasets & loaders
    train_dataset = KhmerIDDataset("train_clean.jsonl", char_vocab, tag_vocab, is_train=True)
    val_dataset   = KhmerIDDataset("val_clean.jsonl",   char_vocab, tag_vocab, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. Model
    model = KhmerIDParserModel(
        vocab_size       = len(char_vocab),
        num_tags         = len(tag_vocab),
        embed_dim        = EMBED_DIM,
        cnn_filters      = CNN_FILTERS,
        cnn_kernel_sizes = [3, 5, 7],
        lstm_hidden_dim  = LSTM_HIDDEN,
        lstm_layers      = LSTM_LAYERS,
        dropout          = DROPOUT,
        pad_idx          = char_vocab.char2idx["<PAD>"],
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # 4. Metrics state (supports resuming)
    metrics     = _load_metrics()
    best_val_f1 = max((e["val_f1"] for e in metrics["epoch_stats"]), default=0.0)
    global_step = metrics["step_losses"][-1]["global_step"] if metrics["step_losses"] else 0
    t_start     = time.time()

    # 5. Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0
        step_loss_accum  = 0
        step_count_accum = 0
        epoch_start      = time.time()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for step_in_epoch, (chars, tags, mask) in enumerate(train_pbar, 1):
            chars, tags, mask = chars.to(DEVICE), tags.to(DEVICE), mask.to(DEVICE)

            optimizer.zero_grad()
            loss = model(chars, tags, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            loss_val          = loss.item()
            total_train_loss += loss_val
            step_loss_accum  += loss_val
            step_count_accum += 1
            global_step      += 1

            current_lr = optimizer.param_groups[0]["lr"]
            train_pbar.set_postfix({"loss": f"{loss_val:.3f}", "lr": f"{current_lr:.6f}"})

            # ── Step-level snapshot every LOG_EVERY steps ─────────────────
            if step_in_epoch % LOG_EVERY == 0:
                metrics["step_losses"].append({
                    "global_step":   global_step,
                    "epoch":         epoch,
                    "step_in_epoch": step_in_epoch,
                    "train_loss":    round(step_loss_accum / step_count_accum, 5),
                    "lr":            round(current_lr, 8),
                    "elapsed_s":     round(time.time() - t_start, 1),
                })
                step_loss_accum  = 0
                step_count_accum = 0
                _save_metrics(metrics)   # flush so a crash loses at most LOG_EVERY steps

        scheduler.step()

        # ── Epoch-level evaluation ─────────────────────────────────────────
        avg_val_loss, f1_dict = _evaluate(model, val_loader, idx2tag, DEVICE)
        avg_train_loss = total_train_loss / len(train_loader)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {avg_val_loss:.4f}")
        print(f"  Val F1     : {f1_dict['f1']:.4f}  "
              f"(P={f1_dict['precision']:.4f}  R={f1_dict['recall']:.4f})")

        metrics["epoch_stats"].append({
            "epoch":          epoch,
            "avg_train_loss": round(avg_train_loss, 5),
            "avg_val_loss":   round(avg_val_loss,   5),
            "val_precision":  f1_dict["precision"],
            "val_recall":     f1_dict["recall"],
            "val_f1":         f1_dict["f1"],
            "lr":             round(current_lr, 8),
            "elapsed_s":      round(time.time() - epoch_start, 1),
        })
        _save_metrics(metrics)

        # ── Save best checkpoint by F1 (more meaningful than val loss) ─────
        if f1_dict["f1"] > best_val_f1:
            best_val_f1 = f1_dict["f1"]
            torch.save({
                "model_state": model.state_dict(),
                "char2idx":    char_vocab.char2idx,
                "tag2idx":     tag_vocab.tag2idx,
                "config": {
                    "vocab_size":       len(char_vocab),
                    "num_tags":         len(tag_vocab),
                    "embed_dim":        EMBED_DIM,
                    "cnn_filters":      CNN_FILTERS,
                    "cnn_kernel_sizes": [3, 5, 7],
                    "lstm_hidden_dim":  LSTM_HIDDEN,
                    "lstm_layers":      LSTM_LAYERS,
                    "dropout":          DROPOUT,
                },
            }, "checkpoints/khmer_id_parser_v2.pth")
            print(f"  ⭐ Best model saved  (val_f1={best_val_f1:.4f})")

    print(f"\nTraining complete. Metrics → {METRICS_PATH}")


if __name__ == "__main__":
    train()