import numpy as np
import pandas as pd
import librosa
from torch.utils.data import Dataset, Subset
import textgrids
import soundfile as sf
from filters_and_features import apply_filters

class VADDataset(Dataset):
    def __init__(self, index_path, target_sr=8000):
        self.data = pd.read_csv(index_path)
        self.speakers = self.data['speaker']
        self.target_sr = target_sr  # Fixed target sample rate of 8 kHz

    @staticmethod
    def readLabels(audio_path, sample_rate, target_sr):
        '''
        Read the file and return the list of SPEECH/NONSPEECH labels for each frame,
        adjusted for the target sampling rate using global resampling.
        '''
        annotation_path = audio_path.replace('.wav', '.TextGrid').replace('Audio', 'Annotation')
        grid = textgrids.TextGrid(annotation_path)

        # Create an array of labels at the original sampling rate
        total_duration = grid.xmax
        num_samples = int(np.round(total_duration * sample_rate))
        labels = np.zeros(num_samples, dtype=int)

        for interval in grid['silences']:
            start_sample = int(np.round(interval.xmin * sample_rate))
            end_sample = int(np.round(interval.xmax * sample_rate))
            labels[start_sample:end_sample] = int(interval.text)

        # Calculate exact number of target samples
        num_target_samples = int(np.ceil(total_duration * target_sr))

        # Create indices for resampling
        original_indices = np.arange(num_samples)
        target_indices = np.linspace(0, num_samples - 1, num_target_samples)

        # Resample using nearest neighbor interpolation
        resampled_labels = labels[np.round(target_indices).astype(int)]

        # Ensure we have exactly the right number of samples
        assert len(resampled_labels) == num_target_samples, "Resampled label count doesn't match target sample count"

        return resampled_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data.iloc[idx]['file_path']
        filename = self.data.iloc[idx]['filename']
        speaker = self.data.iloc[idx]['speaker']
        corpus = self.data.iloc[idx]['corpus']
        meta = self.data.iloc[idx]['meta']
        # Load audio
        audio, original_sr = sf.read(audio_path)  # Use soundfile to get numpy array directly

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample to target sample rate if necessary
        if original_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)

        # Apply filters
        audio = apply_filters(audio, fs=self.target_sr)

        # Extract speech/no speech labels, adjusting to the target sample rate
        labels = self.readLabels(audio_path, original_sr, self.target_sr)

        return audio, labels, filename, speaker, corpus, meta

    def create_splits(self, test_ratio=0.5, random_seed=42):
        train_ratio = 1 - test_ratio
        # Use GroupShuffleSplit to create speaker-independent splits
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_seed)
        train_idx, test_idx = next(gss.split(X=self.data, groups=self.speakers))

        # Create Subset datasets
        train_dataset = Subset(self, train_idx)  
        test_dataset = Subset(self, test_idx)

        print(f"Train set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        return train_dataset, test_dataset
