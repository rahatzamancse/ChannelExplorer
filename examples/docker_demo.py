"""Minimal demo: InceptionV3 + Imagenette for the Docker image."""

from channelexplorer import ChannelExplorer_TF as Cexp, metrics
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = tf.keras.applications.inception_v3.InceptionV3(weights="imagenet")
model.compile(loss="categorical_crossentropy", optimizer="adam")

ds, info = tfds.load(
    "imagenette/320px-v2",
    shuffle_files=False,
    with_info=True,
    as_supervised=True,
    batch_size=None,
)

# Imagenette class labels (synset ID order from tensorflow_datasets)
labels = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]
dataset = ds["train"]

LAYERS_TO_SHOW = [
    "input_1",
    "conv2d",
    "conv2d_2",
    "conv2d_4",
    "mixed0",
    "mixed1",
    "mixed2",
    "mixed3",
    "mixed4",
    "mixed5",
    "mixed6",
    "mixed7",
    "mixed8",
    "mixed9",
    "conv2d_85",
    "conv2d_88",
    "conv2d_87",
    "mixed10",
    "predictions",
]

input_shape = list(model.input.shape[1:3])


@tf.function
def preprocess(x, y):
    x = tf.image.resize(x, input_shape, method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    return x, y


def preprocess_inv(x, y):
    x = ((x / 2 + 0.5) * 255).astype(np.uint8).squeeze()
    return x, y


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--log-level", type=str, default="info")
args = parser.parse_args()

server = Cexp(
    model=model,
    dataset=dataset,
    label_names=labels,
    preprocess=preprocess,
    preprocess_inverse=preprocess_inv,
    log_level=args.log_level,
    summary_fn_image=metrics.summary_fn_image_threshold_otsu,
    apply_relu=False,
    layers_to_show=LAYERS_TO_SHOW,
)
server.run(host=args.host, port=args.port)
