# ChannelExplorer

Neural network activation channel explorer for TensorFlow and PyTorch.

## Quickstart (w/ Docker)

You can run the following to start the whole dockerized application quickly on localhost.

```bash
docker run -p 8000:8000/tcp channelexplorer
```

And open `http://localhost:8000` in your browser.

## Installation

The project is available on PyPI. **Currently the project supports Python >= 3.12**.

```bash
# TensorFlow support
pip install channelexplorer[tf]

# PyTorch support
pip install channelexplorer[torch]

# Both
pip install channelexplorer[all]
```

You will need a running Redis server at the default port for the project to work.

```bash
# Install redis
sudo apt install redis-server   # Debian/Ubuntu
# sudo pacman -S redis          # Arch

# Run redis
redis-server --daemonize yes
```

You can also install Redis using Docker. See the [Docker image](https://hub.docker.com/_/redis).

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone and install with TF extras
uv sync --extra tf

# Run the TF example
uv run --extra tf examples/run_tf.py --host localhost --port 8000

# Run the PyTorch example
uv run --extra torch examples/run_torch.py
```

### Building and Publishing

```bash
# Build the frontend (optional, for bundled static files)
cd frontend && pnpm install && BUILD_PATH=../src/channelexplorer/static pnpm run build && cd ..

# Build sdist + wheel
uv build

# Publish to PyPI
uv publish
```

## Usage

### TensorFlow

```python
from channelexplorer import ChannelExplorer_TF
from channelexplorer import metrics
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from nltk.corpus import wordnet as wn

# Load the tensorflow model
model = tf.keras.applications.vgg16.VGG16(weights='imagenet')
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Load the dataset (must be in tensorflow_datasets format)
ds, info = tfds.load(
    'imagenette/320px-v2',
    shuffle_files=False,
    with_info=True,
    as_supervised=True,
    batch_size=None,
)
labels = list(map(lambda l: wn.synset_from_pos_and_offset(
        l[0], int(l[1:])).name(), info.features['label'].names))
dataset = ds['train']

# Preprocessing function to feed dataset into the model
vgg16_input_shape = tf.keras.applications.vgg16.VGG16().input.shape[1:3].as_list()
@tf.function
def preprocess(x, y):
    x = tf.image.resize(x, vgg16_input_shape, method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x, y

# Inverse preprocessing to display original images in the frontend
def preprocess_inv(x, y):
    x = x.squeeze(0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x, y

server = ChannelExplorer_TF(
    model=model,
    dataset=dataset,
    label_names=labels,
    preprocess=preprocess,
    preprocess_inverse=preprocess_inv,
    log_level="info",
)
server.run(host="localhost", port=8000)
```

### PyTorch

```python
from channelexplorer import APAnalysisTorchModel
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

model = models.vgg16(pretrained=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

server = APAnalysisTorchModel(
    model=model,
    input_shape=(1, 3, 224, 224),
    dataset=dataset,
    label_names=[str(i) for i in range(10)],
    log_level="info",
)
server.run_server(host="localhost", port=8000)
```
