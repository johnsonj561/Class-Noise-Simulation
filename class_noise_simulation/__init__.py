__version__ = "0.1.0"

import math

import numpy as np


def add_noise_by_prob(y, neg_noise_prob, pos_noise_prob, neg_label=0, pos_label=1):
    """Randomly flip labels of the positive and negative classes.
    Args
        y (np.array): ground truth labels.
        pos_noise_prob (float): probability of flipping a positive class label.
        neg_noise_prob (float): probability of flipping a negative class label.
        pos_label (any): positive class label.
        neg_label (any): negative class label.
    Returns
        noisy_y (np.array): transformed y with flipped class labels
    """
    new_y = np.zeros_like(y)
    pos_mask = y == pos_label
    pos_count = pos_mask.sum()
    neg_count = len(y) - pos_count
    choices = [neg_label, pos_label]

    new_y[pos_mask] = np.random.choice(
        choices, size=pos_count, replace=True, p=[pos_noise_prob, 1 - pos_noise_prob]
    )

    new_y[~pos_mask] = np.random.choice(
        choices, size=neg_count, replace=True, p=[1 - neg_noise_prob, neg_noise_prob]
    )

    return new_y


def add_noise_van_hulse(y, Lambda, Psi, neg_label=0, pos_label=1):
    """Adds class noise to y based on Knowledge discovery from imbalanced and noisy data by Jason Van Hulse et al and T.M. Khoshgoftaar.
    Args
        y (np.array): ground truth labels.
        Lambda (float): Class noise level percentage.
        Psi(float): Percentage of noise corresponding to the positive class.
        pos_label (any): positive class label.
        neg_label (any): negative class label.
    Returns
        noisy_y (np.array): transformed y with flipped class labels
        pos_noise_count (int): Number of positive samples that were flipped to negative label.
        neg_noise_count (int): Number of negative samples that were flipped to positive label.
    """
    pos_mask = y == pos_label
    pos_count = pos_mask.sum()

    noise_count = 2 * pos_count * Lambda
    pos_noise_count = math.floor(noise_count * Psi)
    neg_noise_count = int(noise_count - pos_noise_count)

    pos_indices = (y == pos_label).nonzero()[0]
    neg_indices = (y == neg_label).nonzero()[0]

    pos_noise_indices = np.random.choice(pos_indices, pos_noise_count, replace=False)
    neg_noise_indices = np.random.choice(neg_indices, neg_noise_count, replace=False)

    y[pos_noise_indices] = neg_label
    y[neg_noise_indices] = pos_label

    return y, pos_noise_count, neg_noise_count
