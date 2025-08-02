import os
import numpy as np
import pretty_midi
from torch.utils.data import Dataset
from config import SEQUENCE_LENGTH

class MusicDataset(Dataset):
    def __init__(self, root_dir, sequence_length=SEQUENCE_LENGTH, cache_dir="dataset/cache"):
        self.sequence_length = sequence_length
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.midi_paths = self._load_all_midi(root_dir)

    def _load_all_midi(self, root_dir):
        midi_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.mid', '.midi')):
                    midi_files.append(os.path.join(root, file))
        return midi_files

    def __len__(self):
        print(f"[Dataset] Number of MIDI files: {len(self.midi_paths)}")
        return len(self.midi_paths)

    def __getitem__(self, idx):
        print(f"[Dataset] Loading sample {idx}")
        cache_file = os.path.join(self.cache_dir, f"sample_{idx}.npz")
        try:
            if os.path.exists(cache_file):
                data = np.load(cache_file)
                return (
                    data['x'].astype(np.float32),
                    data['y_rhythm'].astype(np.int64),
                    data['y_pitch'].astype(np.float32),
                    data['chords'].astype(np.float32)
                )

            midi_path = self.midi_paths[idx]
            try:
                midi = pretty_midi.PrettyMIDI(midi_path)
            except Exception as e:
                print(f"[MIDI ERROR] Skipping {midi_path}: {e}")
                return self.__getitem__((idx + 1) % len(self))

            notes = []
            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    notes.append([note.pitch, note.velocity, int(note.start * 100)])
            notes = np.array(sorted(notes, key=lambda x: x[2])) if notes else np.zeros((0, 3))

            if len(notes) == 0:
                return self.__getitem__((idx + 1) % len(self))

            notes = self._pad_or_crop(notes)
            x = self._encode_input(notes)
            y_rhythm = self._encode_rhythm(notes)
            y_pitch = self._encode_pitch(notes)
            chords = self._encode_chords(notes)

            if (x.shape != (130, self.sequence_length) or
                y_rhythm.shape != (self.sequence_length,) or
                y_pitch.shape != (self.sequence_length,) or
                chords.shape != (self.sequence_length, 14)):
                print(f"[WARNING] Skipping sample {idx} due to shape mismatch")
                return self.__getitem__((idx + 1) % len(self))

            np.savez_compressed(cache_file, x=x, y_rhythm=y_rhythm, y_pitch=y_pitch, chords=chords)
            return x.astype(np.float32), y_rhythm.astype(np.int64), y_pitch.astype(np.float32), chords.astype(np.float32)

        except Exception as e:
            print(f"[WARNING] Skipping sample {idx} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def _pad_or_crop(self, notes):
        notes = notes[:self.sequence_length]
        if len(notes) < self.sequence_length:
            pad = np.zeros((self.sequence_length - len(notes), 3))
            notes = np.vstack([notes, pad])
        return notes

    def _encode_input(self, notes):
        pitches = notes[:, 0].clip(0, 127).astype(int)
        pitch_onehot = np.eye(128)[pitches]
        velocity = notes[:, 1:2] / 127.0
        timing = (notes[:, 2:3] % 400) / 400.0
        return np.concatenate([pitch_onehot, velocity, timing], axis=1).T

    def _encode_rhythm(self, notes):
        return np.floor(notes[:, 2] / 100).astype(int) % 8

    def _encode_pitch(self, notes):
        return notes[:, 0]

    def _encode_chords(self, notes):
        chord_vocab = ['C', 'Cm', 'D', 'Dm', 'E', 'Em', 'F', 'Fm', 'G', 'Gm', 'A', 'Am', 'B', 'Bm']
        chord_to_idx = {chord: i for i, chord in enumerate(chord_vocab)}
        one_hot_chords = []

        for i in range(len(notes)):
            window = notes[max(0, i - 1): i + 2, 0]
            window = sorted(set(window.astype(int)))
            if len(window) < 3:
                one_hot_chords.append(np.zeros(len(chord_vocab)))
                continue
            root = window[0]
            intervals = [(p - root) % 12 for p in window]
            if intervals == [0, 4, 7]: chord = 'C'
            elif intervals == [0, 3, 7]: chord = 'Cm'
            elif intervals == [0, 4, 8]: chord = 'E'
            elif intervals == [0, 3, 6]: chord = 'Gm'
            else: chord = 'C'
            idx = chord_to_idx.get(chord, 0)
            one_hot = np.zeros(len(chord_vocab))
            one_hot[idx] = 1
            one_hot_chords.append(one_hot)
        return np.array(one_hot_chords)
