import pytest
import numpy as np

from keras.utils import to_categorical

from models.helpers import augment_label_bias_partial
from models.helpers import create_random_labels
from models.helpers import augment_label_random_partial

def test_augment_label_bias_partial():
    # below ~5000 the statistics is getting worse, as it is only num_samples/num_classes per class
    # num_classes = 10
    # num_samples = 5000
    # --> 500 samples per class
    labels = np.random.randint(0, 10, 5000)
    samples = np.arange(len(labels))
    label_to_change = 1
    fraction = 0.8
    samples, new_labels = augment_label_bias_partial(samples,
                                                     labels,
                                                     label_to_change, 2, fraction)
    # train_test_split shuffles, here we use the "samples" to reorder them to check the result
    labels_reordered = labels[samples]
    idx = labels_reordered == label_to_change
    # set absolut tolerance, due to statistical fluctuations
    assert pytest.approx(np.sum(labels_reordered[idx] == new_labels[idx]) / np.sum(idx), fraction, abs=0.02)


def test_create_random_labels():
    labels = np.random.randint(0, 10, 100)
    new_labels = create_random_labels(labels)
    assert not np.any(labels == new_labels)

    labels = np.random.randint(0, 6, 1000)
    new_labels = create_random_labels(labels)
    assert not np.any(labels == new_labels)

    labels = np.random.randint(0, 2, 353)
    new_labels = create_random_labels(labels)
    assert not np.any(labels == new_labels)


def test_augment_label_random_partial():
    num_cls_s = [2, 3, 10]
    num_sam_s = [50, 100, 1000]
    for num_classes in num_cls_s:
        for num_samples in num_sam_s:
            labels = np.random.randint(0, num_classes, num_samples)
            samples = np.arange(len(labels))

            for fraction in [.1, .4, .5, .9]:
                samples, new_labels = augment_label_random_partial(samples,
                                                                   labels,
                                                                   fraction)
                # set absolut tolerance, due to statistical fluctuations
                assert pytest.approx(np.sum(labels == new_labels) / len(labels), 1 - fraction, abs=0.001)



