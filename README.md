# ChannelExplorer

<div align="center">
<a href="https://app.netlify.com/projects/channelexplorer/deploys">
  <img alt="Netlify Status" src="https://api.netlify.com/api/v1/badges/00b5b0c2-73c1-4fdc-9545-9ad6bb8c2159/deploy-status" style="display: inline-block; margin: 0 8px;"/>
</a>
<a href="https://pypi.org/project/channelexplorer/">
  <img alt="PyPI version" src="https://img.shields.io/pypi/v/channelexplorer" style="display: inline-block; margin: 0 8px;"/>
</a>
<br/>
<strong>
  <span style="font-size:2em;">
    🚀 
    <a href="https://channelexplorer.netlify.app" style="text-decoration: none; color: #3b82f6;">
      Demo Link: channelexplorer.netlify.app
    </a>
    🚀
  </span>
</strong>
</div>
<br/>


![ChannelExplorer](.github/readme/teaser.png)

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
docker run -p 8000:8000/tcp rahatzamancse/channelexplorer
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



## Cool Example Usages

### Separation of Classes Across Layers

Loading different models with classification datasets, we can easily see how the classes slowly get separated across the layers.

![separation](.github/readme/separation.png)


### Exploring Intra-Class Separability

Even if we load a single class, we can see that there are subclusters in the activation space for different layers. Here are some examples for different classes.

<details>
<summary>Labrador Retriever: White dog vs. Black dog</summary>

![](.github/readme/subclassification/subclassification-2.png)

</details>

<details>
<summary>Basenji: Western Pet Leanage vs. African Leanage</summary>

![](.github/readme/subclassification/subclassification-3.png)

</details>

<details>
<summary>Red Wolf: Summer vs. Winter Wolf</summary>

![](.github/readme/subclassification/subclassification-4.png)

</details>

<details>
<summary>Mongoose: Desert vs. City Mongoose</summary>

![](.github/readme/subclassification/subclassification-5.png)

</details>

<details>
<summary>Stove: Chimney vs. Stove</summary>

![](.github/readme/subclassification/subclassification-6.png)

</details>

<details>
<summary>Spider Web: Empty vs. Occupied Spider Web</summary>

![](.github/readme/subclassification/subclassification-7.png)

</details>

<details>
<summary>Refrigerator: Open vs. Closed</summary>

![](.github/readme/subclassification/subclassification-8.png)

</details>

<details>
<summary>Pickelhaube: Worn vs. Unworn</summary>

![](.github/readme/subclassification/subclassification-9.png)

</details>

<details>
<summary>Harmonica: Still vs. Playing</summary>

![](.github/readme/subclassification/subclassification-10.png)

</details>

<details>
<summary>Dishwasher: Closed vs. Open</summary>

![](.github/readme/subclassification/subclassification-11.png)

</details>

<details>
<summary>Diaper: Worn vs Unworn</summary>

![](.github/readme/subclassification/subclassification-12.png)

</details>

<details>
<summary>Oxen: Muskox vs. Bull</summary>

![](.github/readme/subclassification/subclassification-1.png)

</details>


### Exploring Inter-Class Confusion and Class Hierarchy

The combined usage of the 3 views (Activation Scatterplot, Jaccard Similarity, and Heatmap View) can be very powerful to explore the inter and intra-class confusion and class hierarchy. Refer to the paper for more details.

![](.github/readme/class-hierarchy.png)


You can follow this overall workflow to explore the inter and intra-class confusion and class hierarchy.

![](.github/readme/workflow.png)



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
