import asyncio
import json
import pathlib
from threading import Thread, Lock
from fastapi.concurrency import asynccontextmanager
from sklearn.cluster import KMeans
import uvicorn
from fastapi.staticfiles import StaticFiles
from typing import Literal, Callable, Any, Dict
import numpy as np

# NumPy 2.x removed numpy.warnings; pyclustering still uses it.
if not hasattr(np, "warnings"):
    import warnings as _warnings

    def _numpy_filterwarnings(*args, **kwargs):
        return _warnings.filterwarnings(*args, **kwargs)

    np.warnings = type("_NumpyWarningsShim", (), {"filterwarnings": staticmethod(_numpy_filterwarnings)})()

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import io
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn import manifold
from sklearn.decomposition import KernelPCA, PCA
from scipy.spatial.distance import pdist, squareform
from ..types import IMAGE_BATCH_TYPE, DENSE_BATCH_TYPE, SUMMARY_BATCH_TYPE, IMAGE_TYPE
from .. import metrics

from pyclustering.cluster.xmeans import xmeans

from ..redis_cache import redis_cache, redis_client

import tensorflow as tf
from tensorflow import keras as K
import tensorflow_datasets as tfds
import keract

# Monkey-patch keract for compatibility with newer Keras versions
# The _is_compiled attribute was removed in newer Keras
_original_evaluate = keract.keract._evaluate

def _patched_evaluate(model, nodes_to_evaluate, x, y=None, auto_compile=False):
    # Check if model is compiled using the new API
    if not getattr(model, '_is_compiled', getattr(model, 'compiled', True)):
        # Try to check via optimizer
        if model.optimizer is None:
            applications_model_names = [
                'densenet', 'efficientnet', 'inception_resnet_v2', 'inception_v3',
                'mobilenet', 'mobilenet_v2', 'nasnet', 'resnet', 'resnet_v2',
                'vgg16', 'vgg19', 'xception'
            ]
            if model.name in applications_model_names:
                print('Transfer learning detected. Model will be compiled with ("categorical_crossentropy", "adam").')
                print('If you want to change the default behaviour, then do in python:')
                print('model.name = ""')
                model.compile(loss='categorical_crossentropy', optimizer='adam')
            elif auto_compile:
                model.compile(loss='categorical_crossentropy', optimizer='adam')
            else:
                print('Model is not compiled. Auto compilation is set to False.')
    
    from keras import Model as KerasModel
    try:
        return KerasModel(inputs=model.input, outputs=list(nodes_to_evaluate)).predict(x)
    except Exception:
        return KerasModel(inputs=model.input, outputs=list(nodes_to_evaluate))(x)

keract.keract._evaluate = _patched_evaluate

from . import utils as utils
import umap
from ..server import Server


class ChannelExplorer_TF(Server):
    """Activation analysis server for TensorFlow / Keras models. """

    def __init__(
        self,
        model: K.Model,
        dataset: tf.data.Dataset,
        label_names: list[str] = [],
        preprocess: Callable = lambda x: x,
        # TODO: remove preprocess_inverse and keep the original input data
        preprocess_inverse: Callable = lambda x: x,
        summary_fn_image: Callable[
            [IMAGE_BATCH_TYPE], SUMMARY_BATCH_TYPE
        ] = metrics.summary_fn_image_l2,
        summary_fn_dense: Callable[
            [DENSE_BATCH_TYPE], SUMMARY_BATCH_TYPE
        ] = metrics.summary_fn_dense_identity,
        log_level: Literal["info", "debug"] = "info",
        layers_to_show: list[str] | Literal["all"] = "all",
        apply_relu: bool = True,
    ):
        """Create a ChannelExplorer instance for a TensorFlow model.

        Args:
            model: A compiled ``keras.Model`` to analyze.
            dataset: A ``tf.data.Dataset`` yielding ``(image, label)`` pairs.
            label_names: Human-readable class names where the i-th element is
                the name of class *i*.
            preprocess: Preprocessing function applied to each ``(image, label)``
                pair before feeding images to the model.
            preprocess_inverse: Inverse of ``preprocess`` used to convert stored
                preprocessed images back to displayable format.
            summary_fn_image: Function that summarizes a spatial activation map
                into a scalar per channel.
            summary_fn_dense: Function that summarizes dense-layer activations
                into a scalar.
            log_level: Uvicorn logging level.
            layers_to_show: Layer names to visualize, or ``"all"`` to include
                every Conv2D / Dense / Flatten / Concatenate layer.
            apply_relu: If ``True``, applies ReLU to Conv2D and hidden Dense
                layer outputs so that post-activation values are used. Useful
                for architectures where ReLU is a separate layer (e.g.
                InceptionV3).
        """
        redis_client.flushdb()

        self.model = model
        self.dataset = dataset
        self.log_level = log_level
        self.summary_fn_image = summary_fn_image
        self.summary_fn_dense = summary_fn_dense
        self.preprocess = preprocess
        self.preprocess_inv = preprocess_inverse
        self.apply_relu = apply_relu

        # Setup caching
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield

        self.app = FastAPI(lifespan=lifespan)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.model_graph = utils.parse_model_graph(model, layers_to_show)
        self._lock = Lock()
        self.shuffled = False
        self.label_names = label_names
        self.labels: list[int] = []
        self.selectedLabels: list[int] = []
        self.datasetImgs: list[list[IMAGE_TYPE]] = []
        self.activations: list[dict] = []
        self.activationsSummary: list[dict[str, float]] = []
        self.datasetLabels: list[list[int]] = []
        self.predictions: list[int] = []

        # Add APIs
        @self.app.get("/api/model/")
        async def read_model():
            return self.model_graph

        @self.app.get("/api/labels/")
        async def read_labels():
            return self.label_names

        @self.app.post("/api/analysis")
        async def analysis(
            labels: list[int], examplePerClass: int = 5, shuffle: bool = False
        ):
            task_id = utils.create_unique_task_id()
            task(task_id, cached_analysis, labels, examplePerClass, shuffle, task_id)
            return {"message": "Analysis started", "task_id": task_id}

        def task(task_id: str, func, *args):
            thread = Thread(
                target=run_and_save_task_result, args=(task_id, func, *args)
            )
            thread.start()

        def run_and_save_task_result(task_id: str, func, *args, **kwargs):
            result = func(*args, **kwargs)
            redis_client.setex(task_id, 3600, json.dumps(result))

        @redis_cache(ttl=3600 * 8)
        def cached_analysis(
            labels: list[int],
            examplePerClass: int = 5,
            shuffle: bool = False,
            _task_id: str = "",
        ):
            return self._analysis(
                labels,
                examplePerClass,
                shuffle,
                progress=lambda x: redis_client.set(f"{_task_id}-progress", x),
            )

        @self.app.get("/api/taskStatus")
        async def taskStatus(task_id: str):
            result = redis_client.get(task_id)
            if result:
                return {
                    "message": "Task completed",
                    "task_id": task_id,
                    "payload": json.loads(result),
                }
            result = redis_client.get(f"{task_id}-progress")
            if result:
                return {
                    "message": result.decode("utf-8"),
                    "task_id": task_id,
                    "payload": None,
                }
            return {"message": "No Task found", "task_id": task_id, "payload": None}

        @self.app.get("/api/analysis/layer/{layer}/argmax")
        async def get_argmax(layer: str):
            return [
                np.argmax(activation[layer][0]).item()
                for activation in self.activations
            ]

        @self.app.get(
            "/api/analysis/image/{image_idx}/layer/{layer_name}/filter/{filter_index}"
        )
        async def get_activation_images(
            image_idx: int, layer_name: str, filter_index: int
        ):
            image = self.activations[image_idx][layer_name][0, :, :, filter_index]
            image -= image.min()
            image = (image - np.percentile(image, 10)) / (
                np.percentile(image, 90) - np.percentile(image, 10)
            )
            image = np.clip(image, 0, 1)
            # invert the image
            image = 1 - image
            image = (image * 255).astype(np.uint8)
            image = np.stack((image,) * 3, axis=-1)
            img = Image.fromarray(image)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {"Content-Disposition": 'inline; filename="test.png"'}
            return Response(content, headers=headers, media_type="image/png")

        @self.app.get("/api/analysis/layer/{layer_name}/embedding")
        async def analysisLayerEmbedding(
            layer_name: str,
            normalization: Literal["none", "row", "col"] = "none",
            method: Literal["mds", "tsne", "pca", "kpca", "umap", "autoencoder", 'autoencoder-pytorch'] = "umap",
            distance: Literal["euclidean", "jaccard"] = "euclidean",
            take_summary: bool = True,
        ):
            def _compute():
                return _compute_embedding(layer_name, normalization, method, distance, take_summary)
            return await asyncio.to_thread(_compute)

        def _compute_embedding(layer_name, normalization, method, distance, take_summary):
            if take_summary:
                this_activation = [
                    activation[layer_name] for activation in self.activationsSummary
                ]
            else:
                this_activation = [
                    activation[layer_name] for activation in self.activations
                ]
            this_activation = np.array(this_activation).reshape(len(this_activation), -1)

            if normalization == "row":
                this_activation = normalize(this_activation, axis=1, norm="l1")
            elif normalization == "col":
                this_activation = normalize(this_activation, axis=0, norm="l1")

            if distance == "euclidean":
                act_dist_mat = squareform(pdist(this_activation, metric='cityblock'))
            elif distance == "jaccard":
                binary = (this_activation > 0.5).astype(np.float64)
                act_dist_mat = squareform(pdist(binary, metric='jaccard'))
                act_dist_mat = np.nan_to_num(act_dist_mat, nan=0.0)

            if method == 'pca':
                class EmbeddingModelPCA:
                    def __init__(self, input_dim):
                        self.embedding_model = PCA(n_components=2)
                    def fit_transform(self, x):
                        return self.embedding_model.fit_transform(this_activation)
                    
                embedding_model = EmbeddingModelPCA(this_activation.shape[1])

            elif method == 'kpca':
                embedding_model = KernelPCA(n_components=2, kernel='precomputed')
            elif method == "mds":
                embedding_model = manifold.MDS(
                    n_components=2,
                    dissimilarity="precomputed",
                    # random_state=6,
                    normalized_stress="auto",
                )
            elif method == "tsne":
                embedding_model = manifold.TSNE(
                    n_components=2,
                    metric="precomputed",
                    # random_state=6,
                    perplexity=min(30, len(this_activation) - 1),
                    init="random",
                )
            elif method == "umap":
                embedding_model = umap.UMAP(n_components=2, metric="precomputed")
            elif method == 'autoencoder':
                class EmbeddingModelAE:
                    def __init__(self, input_dim):
                        # Encoder
                        input_layer = K.Input(shape=(input_dim,))
                        encoded = K.layers.Dense(128, activation='relu')(input_layer)
                        encoded = K.layers.Dense(64, activation='relu')(encoded)
                        bottleneck = K.layers.Dense(2, activation='relu')(encoded)

                        # Decoder
                        decoded = K.layers.Dense(64, activation='relu')(bottleneck)
                        decoded = K.layers.Dense(128, activation='relu')(decoded)
                        decoded = K.layers.Dense(input_dim, activation='sigmoid')(decoded)

                        # Autoencoder Model
                        autoencoder = K.Model(input_layer, decoded)

                        # Encoder Model
                        encoder = K.Model(input_layer, bottleneck)

                        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
                        self.autoencoder = autoencoder
                        self.encoder = encoder

                    def fit_transform(self, x):
                        self.autoencoder.fit(this_activation, this_activation, epochs=50, batch_size=256, shuffle=True)
                        return self.encoder.predict(this_activation)
                
                embedding_model = EmbeddingModelAE(this_activation.shape[1])
            elif method == 'autoencoder-pytorch':
                import torch
                import torch.nn as nn
                import torch.optim as optim

                class EmbeddingModelAE(nn.Module):
                    def __init__(self, input_dim):
                        super(EmbeddingModelAE, self).__init__()
                        
                        # Encoder
                        self.encoder = nn.Sequential(
                            nn.Linear(input_dim, 128),
                            nn.ReLU(True),
                            nn.Linear(128, 64),
                            nn.ReLU(True),
                            nn.Linear(64, 2),
                            nn.ReLU(True)  # Bottleneck
                        )
                        
                        # Decoder
                        self.decoder = nn.Sequential(
                            nn.Linear(2, 64),
                            nn.ReLU(True),
                            nn.Linear(64, 128),
                            nn.ReLU(True),
                            nn.Linear(128, input_dim),
                            nn.Sigmoid()  # Reconstruction
                        )
                        
                    def forward(self, x):
                        x = self.encoder(x)
                        x = self.decoder(x)
                        return x

                    def fit_transform(self, x, num_epochs=5000, batch_size=256):
                        # get normalized this_activation
                        norm_this_activation = normalize(this_activation, axis=0, norm='l1')
                        # Assuming x is a NumPy array. Convert it to a PyTorch tensor.
                        x = torch.tensor(norm_this_activation, dtype=torch.float32)
                        
                        # DataLoader can be used here if you have a large dataset
                        optimizer = optim.Adam(self.parameters(), lr=1e-3)
                        criterion = nn.BCELoss()
                        
                        for epoch in range(num_epochs):
                            optimizer.zero_grad()
                            
                            # Forward pass
                            outputs = self(x)
                            loss = criterion(outputs, x)
                            
                            # Backward and optimize
                            loss.backward()
                            optimizer.step()
                            
                            if (epoch+1) % 10 == 0:
                                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                                
                        return self.encoder(x).detach().numpy()

                embedding_model = EmbeddingModelAE(this_activation.shape[1])


            coords = embedding_model.fit_transform(act_dist_mat)

            return coords.tolist()

        @self.app.get("/api/analysis/alldistances")
        async def analysisAllDistances():
            def _compute():
                all_flat = np.array([
                    np.concatenate([v.ravel() for v in act.values()])
                    for act in self.activationsSummary
                ])
                return squareform(pdist(all_flat, metric='cityblock')).tolist()
            return await asyncio.to_thread(_compute)

        @self.app.get("/api/analysis/layer/{layer_name}/embedding/distance")
        async def analysisLayerEmbeddingDistance(layer_name: str):
            def _compute():
                this_activation = np.array([
                    activation[layer_name] for activation in self.activationsSummary
                ]).reshape(len(self.activationsSummary), -1)
                return squareform(pdist(this_activation, metric='cityblock')).tolist()
            return await asyncio.to_thread(_compute)

        @self.app.get("/api/analysis/layer/{layer_name}/heatmap")
        async def analysisLayerHeatmap(layer_name: str):
            return [
                activation[layer_name].tolist()
                for activation in self.activationsSummary
            ]

        @self.app.get("/api/analysis/layer/{layer_name}/{channel}/heatmap/{image_name}")
        async def analysisLayerHeatmapImage(
            layer_name: str, channel: int, image_name: int
        ):
            in_img = self.datasetImgs[image_name][0].squeeze()
            in_img = (in_img - in_img.min()) / (in_img.max() - in_img.min())

            act_img = self.activations[image_name][layer_name][0][
                :, :, channel
            ].squeeze()
            act_img = (act_img - act_img.min()) / (act_img.max() - act_img.min())
            image = utils.get_activation_overlay(in_img, act_img, alpha=0.6)
            image = (image * 255).astype(np.uint8)
            img = Image.fromarray(image)


            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {"Content-Disposition": 'inline; filename="test.png"'}
            
            return Response(content, headers=headers, media_type="image/png")

        @self.app.get("/api/analysis/allembedding")
        async def analysisAllEmbedding():
            def _compute():
                all_flat = np.array([
                    np.concatenate([v.ravel() for v in act.values()])
                    for act in self.activationsSummary
                ])
                act_dist_mat = squareform(pdist(all_flat, metric='cityblock'))
                mds = manifold.MDS(
                    n_components=2, dissimilarity="precomputed", random_state=6
                )
                return mds.fit(act_dist_mat).embedding_.tolist()
            return await asyncio.to_thread(_compute)

        @self.app.get("/api/analysis/images/{index}")
        async def inputImages(index: int):
            if index < 0 or index >= len(self.datasetImgs):
                return Response(status_code=404)
            image, _ = self.preprocess_inv(
                self.datasetImgs[index][0], self.datasetLabels[index]
            )
            img = Image.fromarray(image)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {"Content-Disposition": 'inline; filename="test.png"'}
            return Response(content, headers=headers, media_type="image/png")

        @self.app.get("/api/analysis/layer/{layer_name}/{channel}/kernel")
        async def analysisLayerKernel(layer_name: str, channel: int):
            if len(model.get_layer(layer_name).get_weights()) == 0:
                # make a 3x3 identity kernel
                kernel = np.zeros((3, 3))
                kernel[1, 1] = 255
                kernel = kernel.astype(np.uint8)
            else:
                kernel = model.get_layer(layer_name).get_weights()[0][:, :, 0, channel]
                kernel = (
                    (kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255
                ).astype(np.uint8)
            img = Image.fromarray(kernel)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {"Content-Disposition": 'inline; filename="test.png"'}
            return Response(content, headers=headers, media_type="image/png")

        @self.app.get("/api/analysis/layer/{layer_name}/cluster")
        async def analysisLayerCluster(layer_name: str, outlier_threshold: float = 0.8, use_xmeans: bool = True, k_clusters: int = 2):
            this_activation = np.array([
                activation[layer_name] for activation in self.activationsSummary
            ])
            
            if use_xmeans:
                xmeans_instance = xmeans(this_activation, kmax=10)
                xmeans_instance.process()

                clusters = xmeans_instance.get_clusters()
                labels = [-1] * len(this_activation)  # Initialize with -1
                for cluster_id, point_indices in enumerate(clusters):
                    for point_idx in point_indices:
                        labels[point_idx] = cluster_id

                centers = xmeans_instance.get_centers()
                
                # Calculate distances manually since xmeans doesn't provide them
                distances = []
                for point in this_activation:
                    # Get minimum distance to any center
                    min_dist = float('inf')
                    for center in centers:
                        dist = np.linalg.norm(point - center)
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)
                    
                outliers = []
                for i in range(len(distances)):
                    if distances[i] > np.mean(distances) + np.std(distances) * outlier_threshold:
                        outliers.append(i)
                
                output = {
                    "labels": labels,
                    "centers": centers,
                    "distances": distances,
                    "outliers": outliers,
                }
                return output
            else:
                kmeans = KMeans(n_clusters=k_clusters, n_init="auto")
                kmeans.fit(this_activation)

                distance_from_center = kmeans.transform(this_activation).min(axis=1)

                # average distance from center for each label
                mean_distance_from_center = np.zeros(k_clusters)
                max_distance_from_center = np.zeros(k_clusters)
                std_distance_form_center = np.zeros(k_clusters)
                for i in range(k_clusters):
                    mean_distance_from_center[i] = distance_from_center[
                        kmeans.labels_ == i
                    ].mean()
                    max_distance_from_center[i] = distance_from_center[
                        kmeans.labels_ == i
                    ].max()
                    std_distance_form_center[i] = distance_from_center[
                        kmeans.labels_ == i
                    ].std()

                # https://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf
                outliers = []
                for i in range(len(distance_from_center)):
                    if (
                        distance_from_center[i]
                        > mean_distance_from_center[kmeans.labels_[i]]
                        + std_distance_form_center[kmeans.labels_[i]] * outlier_threshold
                    ):
                        outliers.append(i)

                output = {
                    "labels": kmeans.labels_.tolist(),
                    "centers": kmeans.cluster_centers_.tolist(),
                    "distances": distance_from_center.tolist(),
                    "outliers": outliers,
                }
                return output

        @self.app.get("/api/analysis/predictions")
        async def analysisPredictions():
            return self.predictions

        @self.app.get("/api/loaded_analysis")
        async def loadedAnalysis():
            output = None
            if not self.selectedLabels:
                output = {
                    "selectedClasses": [],
                    "examplePerClass": 0,
                }
            else:
                output = {
                    "selectedClasses": self.selectedLabels,
                    "examplePerClass": len(self.datasetImgs)
                    // len(self.selectedLabels),
                    "shuffled": self.shuffled,
                    "predictions": self.predictions,
                }
            return output

        # Save all input images
        @self.app.post("/api/analysis/images/save")
        async def saveInputImages():
            # delete existing input_images
            utils.delete_dir("output/input_images")
            # create a directory for each class, then save the images
            examplePerClass = len(self.datasetImgs) // len(self.selectedLabels)
            for i, label in enumerate(self.selectedLabels):
                utils.makedir(f"output/input_images/{label}")
                for j, img in enumerate(
                    self.datasetImgs[examplePerClass * i : examplePerClass * (i + 1)]
                ):
                    img, _ = self.preprocess_inv(img, label)
                    img = Image.fromarray(img.squeeze())
                    img.save(f"output/input_images/{label}/{j}.png")

            # zip the directory and
            utils.zip_dir("output/input_images", "dataset")

            # return the zip file
            response = FileResponse(
                path="dataset.zip", filename="dataset.zip", media_type="application/zip"
            )
            return response

    def _analysis(
        self,
        labels: list[int],
        examplePerClass: int = 50,
        shuffle: bool = False,
        progress: Callable[[str], Any] = lambda x: None,
    ):
        progress("Starting the analysis")
        self.shuffled = shuffle
        self.labels = list(labels)
        self.selectedLabels = labels

        def get_layer_name(layer: K.layers.Layer):
            return layer.name

        layers = list(
            map(
                get_layer_name,
                filter(
                    lambda l: isinstance(
                        l,
                        (
                            # K.layers.InputLayer,
                            K.layers.Conv2D,
                            K.layers.Dense,
                            K.layers.Flatten,
                            K.layers.Concatenate,
                        ),
                    ),
                    self.model.layers,
                ),
            )
        )

        __datasetImgs = [[] for _ in range(len(labels))]
        __activations = [[] for _ in range(len(labels))]
        __activationsSummary = [[] for _ in range(len(labels))]
        __datasetLabels = [[] for _ in range(len(labels))]

        @tf.function
        def filter_by_labels(img, label):
            return tf.reduce_any(tf.equal(label, labels))

        BATCH_SIZE = 32
        collected_imgs = []
        collected_labels = []

        progress("Collecting and filtering images...")
        for img, label in utils.shuffle_or_noshuffle(self.dataset, shuffle=self.shuffled).filter(filter_by_labels).map(self.preprocess):
            label_val = label.numpy().item() if hasattr(label, 'numpy') else int(label)
            label_idx = labels.index(label_val)
            if len(__datasetImgs[label_idx]) >= examplePerClass:
                if all(len(dtImgs) >= examplePerClass for dtImgs in __datasetImgs):
                    break
                continue
            collected_imgs.append(img)
            collected_labels.append(label_val)
            __datasetImgs[label_idx].append(tf.expand_dims(img, 0).numpy())
            __datasetLabels[label_idx].append(label_val)
            if all(len(dtImgs) >= examplePerClass for dtImgs in __datasetImgs):
                break

        total_images = len(collected_imgs)
        progress(f"Collected {total_images} images, extracting activations in batches of {BATCH_SIZE}...")

        all_activations = []
        for batch_start in range(0, total_images, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_images)
            batch_imgs = tf.stack(collected_imgs[batch_start:batch_end])

            progress(f"Processing batch {batch_start // BATCH_SIZE + 1}/{(total_images + BATCH_SIZE - 1) // BATCH_SIZE}")

            batch_activation = keract.get_activations(
                self.model,
                batch_imgs,
                layer_names=layers,
                nodes_to_evaluate=None,
                output_format="simple",
                nested=False,
                auto_compile=True,
            )

            if self.apply_relu:
                for layer_name in batch_activation:
                    layer = self.model.get_layer(layer_name)
                    if isinstance(layer, K.layers.Conv2D):
                        batch_activation[layer_name] = np.maximum(0, batch_activation[layer_name])
                    elif isinstance(layer, K.layers.Dense) and layer_name != layers[-1]:
                        batch_activation[layer_name] = np.maximum(0, batch_activation[layer_name])

            for idx_in_batch in range(batch_end - batch_start):
                single_activation = {
                    k: v[idx_in_batch:idx_in_batch+1] for k, v in batch_activation.items()
                }
                all_activations.append(single_activation)

        for i, (activation, label_val) in enumerate(zip(all_activations, collected_labels)):
            label_idx = labels.index(label_val)
            __activations[label_idx].append(activation)

            activationSummary = {}
            for k, v in activation.items():
                if len(v[0].shape) == 1:
                    activationSummary[k] = self.summary_fn_dense(v)[0]
                elif len(v[0].shape) == 3:
                    activationSummary[k] = self.summary_fn_image(v)[0]
            __activationsSummary[label_idx].append(activationSummary)

        self.datasetImgs = [j for i in __datasetImgs for j in i]
        self.activations = [j for i in __activations for j in i]
        self.activationsSummary = [j for i in __activationsSummary for j in i]
        self.datasetLabels = [j for i in __datasetLabels for j in i]

        self.predictions = [
            np.argmax(act[layers[-1]][0]).item() for act in self.activations
        ]

        return {
            "selectedClasses": self.selectedLabels,
            "examplePerClass": len(self.datasetImgs) // len(self.selectedLabels),
            "shuffled": self.shuffled,
            "predictions": self.predictions,
        }