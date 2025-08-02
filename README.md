How It Works
Training
•	Runs for 100 epochs.
•	Uses a music generation model that combines:
o	Temporal Graph Dilated Convolution (TGDC)
o	MobileNet Temporal Convolutional Network (MobileNetTCN)
o	Sparse Transformer 
o	Fusion Layer
•	Loss Function: Combined CrossEntropyLoss (for rhythm) + MSELoss (for pitch)
•	Optimizer: Adam
Testing
Computes:
•	ACCURACY
•	PRECISION
•	RECALL
•	F1-SCORE
Model Saving
python
torch.save(model.state_dict(), MODEL_PATH)
Replace MODEL_PATH with the trained model name.
Results Saving
python
df.to_csv(csv_path, index=False)
Metrics are saved to a CSV file.
Dataset
The MIDI dataset should be placed in: dataset/midi
You can also use datasets like the Lakh MIDI Dataset Clean
MUSIC_GENERATION\
    ├── __init__.py
    ├── main.py
    ├── config.py
    ├── music_dataset.py
    ├── train.py
    ├── evaluate.py
    ├── model_components.py
    ├── utils.py
    ├── checkpoints\
    │     └── (model files will be saved here)
    └── dataset\
          ├── cache\
          └── midi\
How to Run Training
python
train(MIDI_PATH)
Replace MIDI_PATH with the path to your MIDI dataset.
How to Run Testing
python
run_eval(MODEL_PATH, MIDI_PATH, csv_path="results.csv")
Replace:
•	MODEL_PATH with the trained model checkpoint.
•	MIDI_PATH with the path to your MIDI dataset.

