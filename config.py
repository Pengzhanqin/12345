import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIDI_PATH = "dataset/midi"
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 100
SEQUENCE_LENGTH = 256
MODEL_SAVE_DIR = "checkpoints"
MODEL_PATH = "checkpoints/musicgen_epoch.pt"
