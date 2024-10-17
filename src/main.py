import argparse
import time
import numpy as np
import pandas as pd
import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection as VoiceActivityDetectionPipeline

from dataset import VADDataset
from model import compute_class_statistics, classify_sample
from utils import reference_to_intervals, compute_detection_error_rate_intervals
from filters_and_features import extract_features_labels
from data_index.VAD.create_index_VAD import index
import os

def main():
    parser = argparse.ArgumentParser(description='Custom Voice Activity Detection System')
    parser.add_argument('--vad_folder_path', type=str, default="path/to/voice_activity_detection", help='Path to the dataset index CSV file') #required=True,
    parser.add_argument('--hf_token', type=str, default='yourtoken', help='Hugging Face access token') 
    parser.add_argument('--target_sr', type=int, default=8000, help='Resampling of the data for telephone sampling rate 8kHz')
    parser.add_argument('--frame_length', type=float, default=0.01, help='Frame length in seconds, default 10ms')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--test_ratio', type=float, default=0.5, help='Test ratio for train-test split')
    args = parser.parse_args()

    print("Indexing data...")
    if not os.path.exists(os.path.join(args.vad_folder_path, 'data_index/VAD/index_VAD.csv')):
        index("args.vad_folder_path")

    # Create the datasets
    vad_dataset = VADDataset(os.path.join(args.vad_folder_path, 'data_index/VAD/create_index_VAD.py'), args.target_sr)
    train_dataset, test_dataset = vad_dataset.create_splits(test_ratio=args.test_ratio, random_seed=args.seed)

    # Compute class statistics from the training data
    class_means, class_covariances = compute_class_statistics(train_dataset, args.frame_length)

    # Load the PyAnnote model
    seg_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=args.hf_token)
    pipeline = VoiceActivityDetectionPipeline(segmentation=seg_model)

    # Prepare results dic
    results = {
        'corpus': [],
        'meta': [],
        'pyannote_der': [],
        'pyannote_latency': [],
        'custom_vad_der': [],
        'custom_vad_latency': []
    }

    # Process test data
    for idx in range(len(test_dataset)):
        # Unpack data
        audio, labels, filename, speaker, corpus, meta = test_dataset[idx]

        sample_rate = vad_dataset.target_sr

        # Convert frame-level reference labels to intervals
        reference_intervals = reference_to_intervals(labels, sample_rate)

        # PyAnnote model evaluation
        initial_params = {"min_duration_on": 0.0, "min_duration_off": 0.0}
        pipeline.instantiate(initial_params)

        # Measure latency for PyAnnote
        start_time = time.time()
        pyannote_vad = pipeline({"waveform": torch.Tensor(audio).unsqueeze(0), "sample_rate": int(sample_rate)})
        pyannote_latency = time.time() - start_time

        pyannote_preds = [(start, end) for start, end in pyannote_vad.get_timeline()]
        # Compute DER for PyAnnote model, comparing time intervals
        pyannote_der = compute_detection_error_rate_intervals(reference_intervals, pyannote_preds)

        # Custom VAD evaluation
        start_time = time.time()
        # Extract features and labels each 10ms
        features_array, _ = extract_features_labels(audio, labels, sample_rate, args.frame_length) #Based on 

        vad_preds = []
        for sample in features_array:
            predicted_label = classify_sample(sample, class_means, class_covariances)
            vad_preds.append(predicted_label)
        custom_vad_latency = time.time() - start_time

        # Convert predictions for each 10ms to intervals - with a frame length of 10 ms, frame_rate = 1 / 0.01 = 100 frames per second.
        vad_preds_intervals = reference_to_intervals(vad_preds, 1 / args.frame_length)

        # Compute DER for custom VAD comparing time intervals
        vad_der = compute_detection_error_rate_intervals(reference_intervals, vad_preds_intervals)

        # Store results
        results['corpus'].append(corpus)
        results['meta'].append(meta)
        results['pyannote_der'].append(pyannote_der)
        results['pyannote_latency'].append(pyannote_latency)
        results['custom_vad_der'].append(vad_der)
        results['custom_vad_latency'].append(custom_vad_latency)


    results_df = pd.DataFrame(results)

    #print(f"Average sample duration" ...)

    print("\nResults:")
    print(f"PyAnnote average DER: {results_df.pyannote_der.mean():.3f}, average latency per sample: {results_df.pyannote_latency.mean()*1000:.1f} ms")
    print(f"Custom VAD average DER: {results_df.custom_vad_der.mean():.3f}, average latency per sample: {results_df.custom_vad_latency.mean()*1000:.1f} ms\n")
    
    # Grouped results
    grouped_results = results_df.groupby(['corpus', 'meta']).agg({
        'pyannote_der': ['count', 'mean'],
        'custom_vad_der': ['mean']
    }).sort_values(('custom_vad_der', 'mean'))

    print('Results grouped by corpus and "meta" information - sorted by custom_vad_der:')
    print(grouped_results)

if __name__ == "__main__":
    main()
