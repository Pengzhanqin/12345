from train import train
from evaluate import run_eval
from config import MODEL_PATH, MIDI_PATH

if __name__ == "__main__":
    print("[DEBUG] Script started")
    print(f"[DEBUG] Scanning directory: {MIDI_PATH}")
    mode = input("Enter mode (train/eval): ").strip().lower()

    try:
        if mode == "train":
            train(MIDI_PATH)
            from dataset import MusicDataset

            print("[DEBUG] Loaded MusicDataset from:", MusicDataset)
        elif mode == "eval":
            run_eval(MODEL_PATH, MIDI_PATH, csv_path="results.csv")
        else:
            print("[ERROR] Invalid mode. Please enter 'train' or 'eval'.")
    except Exception as e:
        print(f"[ERROR] Something went wrong: {e}")
