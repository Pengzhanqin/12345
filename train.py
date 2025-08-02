import os
import torch
from torch.utils.data import DataLoader
from music_dataset import MusicDataset
from model import MusicGenModel
from utils import combined_loss, safe_collate
from config import DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_SAVE_DIR, MODEL_PATH

def train(midi_dir):
    device = DEVICE
    dataset = MusicDataset(midi_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=safe_collate)
    model = MusicGenModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print(f"[INFO] Epoch {epoch+1}/{EPOCHS}")
        for batch in loader:
            if batch is None:
                continue
            x, y_rhythm, y_pitch, _ = batch
            x, y_rhythm, y_pitch = x.to(device), y_rhythm.to(device), y_pitch.to(device)
            rhythm_pred, pitch_pred = model(x)
            loss = combined_loss(rhythm_pred, y_rhythm, y_pitch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss / len(loader):.4f}")
        torch.save(model.state_dict(), MODEL_PATH)
