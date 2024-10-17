import numpy as np
from filters_and_features import extract_features_labels

def compute_class_statistics(train_dataset, frame_length):
    """
    Compute mean vectors and covariance matrices for each class using training data.
    """
    
    features_list = []
    labels_list = []

    # Iterate over the training dataset
    for idx in range(len(train_dataset)):
        # Unpack data
        audio, labels, filename, speaker, corpus, meta = train_dataset[idx]
        # Extract features and labels per frame
        features_array, labels_array = extract_features_labels(audio, labels, train_dataset.dataset.target_sr, frame_length)
        features_list.append(features_array)
        labels_list.append(labels_array)

    # Concatenate features and labels from all files
    train_features = np.vstack(features_list)  # Shape: (total_samples, num_features)
    train_labels = np.hstack(labels_list)      # Shape: (total_samples,)

    # Identify unique classes (e.g., 0 for unvoiced, 1 for voiced)
    classes = np.unique(train_labels)
    class_means = {}
    class_covariances = {}

    # Compute mean vectors and covariance matrices for each class
    for c in classes:
        class_features = train_features[train_labels == c]
        mean_vector = np.mean(class_features, axis=0)
        covariance_matrix = np.cov(class_features, rowvar=False)
        # Regularize covariance matrix to prevent singularity
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6
        class_means[c] = mean_vector
        class_covariances[c] = covariance_matrix

    return class_means, class_covariances

def classify_sample(sample, class_means, class_covariances):
    """
    Classify a sample using the Mahalanobis distance.
    """
    distances = {}

    for c in class_means:
        diff = sample - class_means[c]
        inv_cov = np.linalg.inv(class_covariances[c])
        # Compute Weighted Euclidian distance
        distance = diff.T @ inv_cov @ diff
        distances[c] = distance

    # Assign to class with minimum distance
    predicted_class = min(distances, key=distances.get)
    return predicted_class
