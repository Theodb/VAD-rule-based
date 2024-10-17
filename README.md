# Voice Activity Detection (VAD) System

This repository contains a rule-based Voice Activity Detection (VAD) system that compares a custom VAD algorithm with the PyAnnote pre-trained VAD model from the `segmentation-3.0` release.

## Table of Contents

- [Objectives](#objectives)
- [Methodology](#methodology)
  - [Custom VAD Model](#custom-vad-model)
  - [PyAnnote VAD Model](#pyannote-vad-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)
- [License](#license)

## Methodology

### Custom VAD Model

The custom VAD model is influenced by the paper:

> Meduri, S.S., & Ananth, R. (2012). *A Survey and Evaluation of Voice Activity Detection Algorithms*.

This method measures five different parameters extracted from the speech signal after high-pass filter:

1. **Zero Crossing Count**: Indicates the frequency content of the signal.
2. **Speech Energy**: Represents amplitude variations.
3. **Correlation Between Adjacent Samples**: Captures signal continuity.
4. **First Predictor Coefficient from LPC Analysis**: Reflects the spectral envelope.
5. **Energy in the Prediction Error**: Residual energy after Linear Predictive Coding (LPC).

Classification is achieved by computing the weighted Euclidean distance between the extracted parameters and assigning the segment to the class with the minimum distance.

### PyAnnote VAD Model

The PyAnnote pre-trained VAD model from `segmentation-3.0` is used for comparison. Note that this model outputs timecodes without providing access to underlying probabilities, which limits certain evaluation metrics like ROC curves.

## Evaluation

The evaluation process includes:

- **Dataset Selection**: A Kaggle VAD dataset comprising TIMIT, PTDB-TUG, and Noizeus (+5 SNR) segments with their corresponding labels per frame was used. https://www.kaggle.com/datasets/lazyrac00n/speech-activity-detection-datasets
- **Dataset Partitioning**: The dataset was divided into training and testing sets, ensuring that the splits are independent of the speakers.
- **Metrics**: Detection Error Rate (DER) and latency are used to measure performance.
- **Performance Comparison**: Both models are evaluated on the test set, and the results are compared.
- **Analysis**: Results are grouped by corpus and metadata for detailed analysis.

## Results

The script outputs:

- **Average DER**: For both PyAnnote and the custom VAD model.
- **Average Latency**: Average processing time per sample for both models.
- **Grouped Results**: Aggregated by corpus and metadata, sorted by the custom VAD's DER.

*Note*: The inability to access probabilities from the latest PyAnnote model v3.0 restricts the comparison to DER and latency metrics.

## Requirements

- Python 3.7 or higher
- Python packages listed in `requirements.txt`

## Usage

Run the VAD comparison script:

```bash
python main.py --vad_folder_path PATH_TO_VAD_FOLDER --hf_token YOUR_HUGGING_FACE_TOKEN
```

**Arguments:**

- `--vad_folder_path`: Path to the VAD dataset folder.
- `--hf_token`: Hugging Face access token for downloading the PyAnnote model.

**Optional Arguments:**

- `--target_sr`: Target sample rate for resampling (default: `8000` Hz).
- `--frame_length`: Frame length in seconds (default: `0.01` seconds).
- `--seed`: Random seed for reproducibility (default: `42`).
- `--test_ratio`: Ratio for splitting the dataset into test and train sets (default: `0.5`).

## Project Structure

- `main.py`: Main script for running the VAD comparison.
- `dataset.py`: Handles data loading and preprocessing.
- `model.py`: Functions for computing statistics and classification.
- `utils.py`: Utility functions for DER computation and interval processing.
- `filters_and_features.py`: Feature extraction and label processing.
- `data_index/VAD/create_index_VAD.py`: Script to index the VAD dataset.
- `requirements.txt`: List of required Python packages.

## References

- Meduri, S.S., & Ananth, R. (2012). *A Survey and Evaluation of Voice Activity Detection Algorithms*.
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio)
