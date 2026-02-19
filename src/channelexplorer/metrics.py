# from beartype import beartype
import numpy as np
from .types import IMAGE_BATCH_TYPE, DENSE_BATCH_TYPE, SUMMARY_BATCH_TYPE
import tensorflow as tf
from typing import Any, Tuple

# Summarization functions
# @beartype
def summary_fn_image_percentile(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    return np.percentile(np.abs(x), 90, axis=range(len(x.shape)-1))

# @beartype
def summary_fn_image_l2(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    return np.linalg.norm(np.abs(x), axis=tuple(range(1, len(x.shape)-1)), ord=2)

# @beartype
def summary_fn_image_threshold_mean(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    threshold = np.median(np.abs(x), axis=tuple(range(1, len(x.shape)-1))) 
    return (x > threshold).sum(axis=tuple(range(1, len(x.shape)-1)))

# @beartype
def summary_fn_image_maximum(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    return np.max(np.abs(x), axis=tuple(range(1, len(x.shape)-1)))

# @beartype
def summary_fn_image_threshold_median(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    threshold = np.mean(np.abs(x), axis=tuple(range(1, len(x.shape)-1)))
    return (x > threshold).sum(axis=tuple(range(1, len(x.shape)-1)))


# @beartype
def summary_fn_image_threshold_otsu(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    bins_num = x.shape[1] * x.shape[2]
    batch_thresholds = []
    for batch in x:
        thresholds = []
        for img in batch.transpose(2, 0, 1):
            hist, bin_edges = np.histogram(img, bins=bins_num)
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            # Get the class means mu0(t)
            mean1 = np.cumsum(hist * bin_mids) / weight1
            # Get the class means mu1(t)
            mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

            inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            
            # Maximize the inter_class_variance function val
            index_of_max_val = np.argmax(inter_class_variance)
            
            threshold = bin_mids[:-1][index_of_max_val]
            thresholds.append(threshold)
        batch_thresholds.append(thresholds)
    
    batch_thresholds = np.array(batch_thresholds)
    return (x > batch_thresholds).sum(axis=tuple(range(1, len(x.shape)-1)))

# @beartype
def summary_fn_dense_identity(x: DENSE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    return x

# Preprocessing functions
def preprocess_vgg_tensorflow(img_batch_with_label: Tuple[IMAGE_BATCH_TYPE, Any], size=[299,299]) -> tuple[tf.Tensor, tf.Tensor]:
    img, labels = img_batch_with_label
    img = tf.image.central_crop(img, central_fraction=0.875)
    img = tf.image.resize(img, size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.math.divide(tf.image.convert_image_dtype(img, dtype=tf.float32), 255.)
    img -= 0.5
    img *= 2.0
    return img, labels