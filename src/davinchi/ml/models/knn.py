"""k-NN classification and Gaussian-kernel-weighted k-NN regression."""
import numpy as np
from ..distance.combined import combined_distance


def knn_classify(X_train, y_train, X_test, k, num_idx, cat_idx, vdm_model=None, p=2):
    """Majority-vote k-NN classification with random tie-breaking."""
    predictions = []
    for x_q in X_test:
        distances = np.array([
            combined_distance(x_q, x_tr, num_idx, cat_idx, vdm_model, p)
            for x_tr in X_train
        ])
        nn_idx = np.argsort(distances)[:k]
        neighbor_labels = y_train[nn_idx]
        classes, counts = np.unique(neighbor_labels, return_counts=True)
        max_count = np.max(counts)
        candidates = classes[counts == max_count]
        predicted = candidates[np.random.randint(len(candidates))]
        predictions.append(predicted)
    return np.array(predictions)


def gaussian_kernel(distance, gamma):
    """Weight that decays with distance; gamma controls how fast it drops off."""
    exponent = -gamma * (distance ** 2)
    return np.exp(exponent)


def knn_regress(X_train, y_train, X_test, k, gamma, num_idx, cat_idx, vdm_model=None, p=2):
    """Gaussian-kernel-weighted k-NN regression."""
    predictions = []
    for x_q in X_test:
        if len(cat_idx) == 0:
            distances = np.sum(np.abs(X_train[:, num_idx] - x_q[num_idx]) ** p, axis=1)
        else:
            distances = np.array([
                combined_distance(x_q, x_tr, num_idx, cat_idx, vdm_model, p)
                for x_tr in X_train
            ])

        nn_idx = np.argsort(distances)[:k]
        neighbor_targets = y_train[nn_idx]
        neighbor_distances = distances[nn_idx]
        weights = np.array([gaussian_kernel(d, gamma) for d in neighbor_distances])
        weight_sum = np.sum(weights)

        if weight_sum == 0:
            prediction = np.mean(neighbor_targets)
        else:
            prediction = np.sum(weights * neighbor_targets) / weight_sum

        predictions.append(prediction)

    return np.array(predictions)
