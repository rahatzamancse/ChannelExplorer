# ChannelExplorer

A visual analytics tool for exploring neural network activation channels.
Supports both **TensorFlow / Keras** and **PyTorch** models.

ChannelExplorer extracts per-channel activation summaries from convolutional
and dense layers, then presents them through coordinated views — heatmaps,
dimensionality-reduced embeddings, clustering, and overlay visualizations — so
you can quickly identify patterns, outliers, and redundancies across classes.

## Features

- **Model Graph View** — Interactive, layered visualization of the network architecture.
- **Activation Heatmaps** — Per-channel activation magnitudes across all images.
- **Embedding Projections** — MDS, t-SNE, UMAP, PCA, or autoencoder projections to reveal class separability at each layer.
- **Activation Overlays** — Superimpose channel activations onto original inputs.
- **Clustering & Outlier Detection** — X-Means / K-Means clustering with automatic outlier flagging.
- **Pluggable Summary Functions** — L2 norm, percentile, Otsu threshold, and more — or bring your own.

## Demo Quickstart (InceptionV3 + Imagenette)

To run the demo static website for the InceptionV3 + Imagenette example, you can use the following command. But it is less-interactive than the server as it is just pre-computed data without any CPU/GPU dependencies.

```bash
docker run -p 8000:8000/tcp channelexplorer
```

Then open <http://localhost:8000> in your browser.

## Installation

Available on PyPI. **Requires Python >= 3.12**.

```bash
# TensorFlow support
pip install channelexplorer[tf]

# PyTorch support
pip install channelexplorer[torch]

# Both
pip install channelexplorer[all]
```

### Redis

A running Redis server is used for caching analysis results.

```bash
# Install redis
sudo apt install redis-server   # Debian/Ubuntu
# sudo pacman -S redis          # Arch

# Run redis
redis-server --daemonize yes
# sudo systemctl start redis
```

You can also use the official [Redis Docker image](https://hub.docker.com/_/redis).

To point at a non-default Redis instance, set these environment variables:

| Variable | Default |
| --- | --- |
| `REDIS_HOST` | `localhost` |
| `REDIS_PORT` | `6379` |
| `REDIS_DB` | `0` |

## Usage

You can see `examples/run_tf.py` for a complete example with multiple model and parameter options. 

### TensorFlow

```python
from channelexplorer import ChannelExplorer_TF, metrics
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from nltk.corpus import wordnet as wn

model = tf.keras.applications.vgg16.VGG16(weights="imagenet")
model.compile(loss="categorical_crossentropy", optimizer="adam")

ds, info = tfds.load(
    "imagenette/320px-v2",
    shuffle_files=False,
    with_info=True,
    as_supervised=True,
    batch_size=None,
)
labels = list(
    map(
        lambda l: wn.synset_from_pos_and_offset(l[0], int(l[1:])).name(),
        info.features["label"].names,
    )
)
dataset = ds["train"]

vgg16_input_shape = tf.keras.applications.vgg16.VGG16().input.shape[1:3].as_list()

@tf.function
def preprocess(x, y):
    x = tf.image.resize(x, vgg16_input_shape, method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x, y

def preprocess_inv(x, y):
    x = x.squeeze(0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x, y

server = ChannelExplorer_TF(
    model=model,
    dataset=dataset,
    label_names=labels,
    preprocess=preprocess,
    preprocess_inverse=preprocess_inv,
    summary_fn_image=metrics.summary_fn_image_l2,
    log_level="info",
)
server.run(host="localhost", port=8000)
```

### PyTorch

```python
from channelexplorer import APAnalysisTorchModel
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

server = APAnalysisTorchModel(
    model=model,
    input_shape=(1, 3, 224, 224),
    dataset=dataset,
    label_names=[str(i) for i in range(10)],
    log_level="info",
)
server.run(host="localhost", port=8000)
```

Once the server is running, open <http://localhost:8000> (or use the
standalone frontend in development mode — see below).

## Development

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency
management and [pnpm](https://pnpm.io/) for the Next.js frontend.

```bash
# Clone the repo
git clone https://github.com/rahatzamancse/APalysis.git
cd APalysis

# Install Python deps with TF extras
uv sync --extra tf

# Run the TF example
uv run --extra tf examples/run_tf.py --host localhost --port 8000

# Run the PyTorch example
uv run --extra torch examples/run_torch.py
```

### Frontend (Next.js)

```bash
cd frontend
pnpm install
pnpm dev          # starts on http://localhost:3000
```

## Project Structure

```
├── src/channelexplorer/       # Python library
│   ├── server.py              # Base FastAPI server
│   ├── metrics.py             # Activation summary functions
│   ├── types.py               # Shared type aliases
│   ├── utils.py               # Graph layout & image utilities
│   ├── redis_cache.py         # Redis caching
│   ├── channelexplorer_tf/    # TensorFlow backend
│   └── channelexplorer_torch/ # PyTorch backend
├── frontend/                  # Next.js frontend
├── examples/                  # Ready-to-run example scripts
├── Dockerfile                 # Production Docker image
└── pyproject.toml
```
