import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def combined_loss(rhythm_pred, rhythm_gt, pitch_gt):
    B, C, T = rhythm_pred.shape
    rhythm_loss = F.cross_entropy(rhythm_pred.permute(0, 2, 1).reshape(-1, C), rhythm_gt.reshape(-1))
    pitch_loss = F.mse_loss(rhythm_pred[:, :1, :], pitch_gt.unsqueeze(1))
    return rhythm_loss + pitch_loss

def safe_collate(batch):
    batch = [b for b in batch if b is not None and len(b) == 4]
    if len(batch) == 0:
        return None
    try:
        x_batch = torch.stack([torch.tensor(b[0]) for b in batch])
        y_rhythm_batch = torch.stack([torch.tensor(b[1]) for b in batch])
        y_pitch_batch = torch.stack([torch.tensor(b[2]) for b in batch])
        chords_batch = torch.stack([torch.tensor(b[3]) for b in batch])
        return x_batch, y_rhythm_batch, y_pitch_batch, chords_batch
    except Exception as e:
        print(f"[Collate Warning] Skipping batch due to error: {e}")
        return None

def evaluate(preds, labels):
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return (
        accuracy_score(labels, preds),
        precision_score(labels, preds, average='macro', zero_division=0),
        recall_score(labels, preds, average='macro', zero_division=0),
        f1_score(labels, preds, average='macro', zero_division=0)
    )
