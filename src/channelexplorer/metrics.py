"""Summarization and preprocessing functions for activation analysis.

Each ``summary_fn_image_*`` function reduces a batch of spatial activation
maps to a 1-D summary vector per image (one scalar per channel).  They can
be passed as ``summary_fn_image`` when constructing a ChannelExplorer
instance.
"""

import numpy as np
from .types import IMAGE_BATCH_TYPE, DENSE_BATCH_TYPE, SUMMARY_BATCH_TYPE
from typing import Any, Tuple


def summary_fn_image_percentile(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    """Summarize each channel by the 90th percentile of absolute values.

    Args:
        x: Activation maps of shape ``(batch, height, width, channels)``.

    Returns:
        Array of shape ``(batch, channels)`` with percentile summaries.
    """
    return np.percentile(np.abs(x), 90, axis=range(len(x.shape)-1))


def summary_fn_image_l2(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    """Summarize each channel by the L2 norm over spatial dimensions.

    Args:
        x: Activation maps of shape ``(batch, height, width, channels)``.

    Returns:
        Array of shape ``(batch, channels)`` with L2 norm summaries.
    """
    return np.linalg.norm(np.abs(x), axis=tuple(range(1, len(x.shape)-1)), ord=2)


def summary_fn_image_threshold_mean(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    """Count activations above the per-channel median absolute value.

    Args:
        x: Activation maps of shape ``(batch, height, width, channels)``.

    Returns:
        Array of shape ``(batch, channels)`` with thresholded counts.
    """
    threshold = np.median(np.abs(x), axis=tuple(range(1, len(x.shape)-1))) 
    return (x > threshold).sum(axis=tuple(range(1, len(x.shape)-1)))


def summary_fn_image_maximum(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    """Summarize each channel by the maximum absolute value.

    Args:
        x: Activation maps of shape ``(batch, height, width, channels)``.

    Returns:
        Array of shape ``(batch, channels)`` with max-absolute summaries.
    """
    return np.max(np.abs(x), axis=tuple(range(1, len(x.shape)-1)))


def summary_fn_image_threshold_median(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    """Count activations above the per-channel mean absolute value.

    Args:
        x: Activation maps of shape ``(batch, height, width, channels)``.

    Returns:
        Array of shape ``(batch, channels)`` with thresholded counts.
    """
    threshold = np.mean(np.abs(x), axis=tuple(range(1, len(x.shape)-1)))
    return (x > threshold).sum(axis=tuple(range(1, len(x.shape)-1)))


def summary_fn_image_threshold_otsu(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    """Count activations above an Otsu-computed threshold per channel.

    Computes an optimal binarization threshold for each channel using
    Otsu's method and returns the count of pixels exceeding it.

    Args:
        x: Activation maps of shape ``(batch, height, width, channels)``.

    Returns:
        Array of shape ``(batch, channels)`` with thresholded counts.
    """
    bins_num = x.shape[1] * x.shape[2]
    batch_thresholds = []
    for batch in x:
        thresholds = []
        for img in batch.transpose(2, 0, 1):
            hist, bin_edges = np.histogram(img, bins=bins_num)
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            mean1 = np.cumsum(hist * bin_mids) / weight1
            mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

            inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            index_of_max_val = np.argmax(inter_class_variance)
            threshold = bin_mids[:-1][index_of_max_val]
            thresholds.append(threshold)
        batch_thresholds.append(thresholds)
    
    batch_thresholds = np.array(batch_thresholds)
    return (x > batch_thresholds).sum(axis=tuple(range(1, len(x.shape)-1)))


def summary_fn_dense_identity(x: DENSE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    """Identity summarization — returns dense activations unchanged.

    Args:
        x: Dense layer output of shape ``(batch, units)``.

    Returns:
        The input array unchanged.
    """
    return x


def preprocess_vgg_tensorflow(img_batch_with_label: Tuple[IMAGE_BATCH_TYPE, Any], size=[299, 299]):
    """Preprocess images for VGG-style TensorFlow models.

    Applies center-crop, resize, and VGG-style normalization (scale to
    ``[-1, 1]``).  Only use with the TensorFlow backend.

    Args:
        img_batch_with_label: A tuple of ``(image_tensor, label)``.
        size: Target spatial dimensions ``[height, width]``.

    Returns:
        A tuple of ``(preprocessed_image, label)``.
    """
    import tensorflow as tf
    img, labels = img_batch_with_label
    img = tf.image.central_crop(img, central_fraction=0.875)
    img = tf.image.resize(img, size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.math.divide(tf.image.convert_image_dtype(img, dtype=tf.float32), 255.)
    img -= 0.5
    img *= 2.0
    return img, labels
