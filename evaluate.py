import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import MusicDataset
from model import MusicGenModel
from utils import safe_collate, evaluate
from config import DEVICE

def run_eval(model_path, midi_dir, csv_path):
    device = DEVICE
    dataset = MusicDataset(midi_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10, collate_fn=safe_collate)
    model = MusicGenModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y_rhythm, _, _ = batch
            x, y_rhythm = x.to(device), y_rhythm.to(device)
            rhythm_pred, _ = model(x)
            rhythm_pred = rhythm_pred.permute(0, 2, 1)
            rhythm_pred_flat = rhythm_pred.reshape(-1, rhythm_pred.shape[-1])
            y_rhythm_flat = y_rhythm.reshape(-1)
            all_preds.append(rhythm_pred_flat)
            all_labels.append(y_rhythm_flat)

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    acc, prec, rec, f1 = evaluate(all_preds_tensor, all_labels_tensor)

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    pd.DataFrame([{"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}]).to_csv(csv_path, index=False)
    print(f"[INFO] Evaluation metrics saved to {csv_path}")
