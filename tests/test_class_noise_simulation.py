import math

import numpy as np
from class_noise_simulation import __version__, add_noise_by_prob, add_noise_van_hulse


def test_version():
    assert __version__ == "0.1.0"


def test_pos_noise_prob_0pt1():
    pos_count, neg_count = 100, 900
    pos_noise_prob, neg_noise_prob = 0.1, 0.0
    y = np.concatenate((np.ones(pos_count), np.zeros(neg_count)))
    noisy_y = add_noise_by_prob(
        y,
        neg_noise_prob=neg_noise_prob,
        pos_noise_prob=pos_noise_prob,
        neg_label=0,
        pos_label=1,
    )

    noisy_pos_count = noisy_y.sum()
    noisy_neg_count = len(noisy_y) - noisy_pos_count

    # positive class label flips will
    # decrease number of positive sampes
    assert noisy_pos_count < pos_count
    # and increase number of negative samples
    assert noisy_neg_count > neg_count


def test_neg_noise_prob_0pt1():
    pos_count, neg_count = 100, 900
    pos_noise_prob, neg_noise_prob = 0.0, 0.1
    y = np.concatenate((np.ones(pos_count), np.zeros(neg_count)))
    noisy_y = add_noise_by_prob(
        y,
        neg_noise_prob=neg_noise_prob,
        pos_noise_prob=pos_noise_prob,
        neg_label=0,
        pos_label=1,
    )

    noisy_pos_count = noisy_y.sum()
    noisy_neg_count = len(noisy_y) - noisy_pos_count

    # negative class label flips will
    # increase number of positive sampes
    assert noisy_pos_count > pos_count
    # and decrease number of negative samples
    assert noisy_neg_count < neg_count


def test_noise_by_van_hulse_psi1():
    pos_count, neg_count = 100, 900
    Lambda, Psi = 0.4, 1
    y = np.concatenate((np.ones(pos_count), np.zeros(neg_count)))
    noisy_y, pos_noise_count, neg_noise_count = add_noise_van_hulse(
        y, Lambda=Lambda, Psi=Psi, neg_label=0, pos_label=1,
    )

    expected_noisy_samples = 2 * pos_count * Lambda
    expected_pos_noisy_samples = math.floor(expected_noisy_samples * Psi)
    expected_neg_noisy_samples = expected_noisy_samples - expected_pos_noisy_samples

    assert expected_noisy_samples == pos_noise_count + neg_noise_count
    assert expected_pos_noisy_samples == pos_noise_count
    assert expected_neg_noisy_samples == neg_noise_count

    noisy_pos_count = noisy_y.sum()
    noisy_neg_count = len(noisy_y) - noisy_pos_count

    # Psi = 1 ==> we expect all noise to be of the type P -> N flips
    # this means the number of positive samples should have decreased by expected_pos_noisy_samples
    assert noisy_pos_count == pos_count - expected_pos_noisy_samples
    # and the number of negative samples should have increased by this same amount
    assert noisy_neg_count == neg_count + expected_pos_noisy_samples


def test_noise_by_van_hulse_psi0():
    pos_count, neg_count = 100, 900
    Lambda, Psi = 0.4, 0
    y = np.concatenate((np.ones(pos_count), np.zeros(neg_count)))
    noisy_y, pos_noise_count, neg_noise_count = add_noise_van_hulse(
        y, Lambda=Lambda, Psi=Psi, neg_label=0, pos_label=1,
    )

    expected_noisy_samples = 2 * pos_count * Lambda
    expected_pos_noisy_samples = math.floor(expected_noisy_samples * Psi)
    expected_neg_noisy_samples = expected_noisy_samples - expected_pos_noisy_samples

    assert expected_noisy_samples == pos_noise_count + neg_noise_count
    assert expected_pos_noisy_samples == pos_noise_count
    assert expected_neg_noisy_samples == neg_noise_count

    noisy_pos_count = noisy_y.sum()
    noisy_neg_count = len(noisy_y) - noisy_pos_count

    # Psi = 0 ==> we expect all noise to be of the type N -> P flips
    # this means the number of positive samples should have decreased by expected_pos_noisy_samples
    assert noisy_pos_count == pos_count + expected_neg_noisy_samples
    # and the number of negative samples should have increased by this same amount
    assert noisy_neg_count == neg_count - expected_neg_noisy_samples
