import asyncio
import json
from threading import Thread, Lock
from fastapi.concurrency import asynccontextmanager
from sklearn.cluster import KMeans
import torch
from typing import Literal, Callable, Any
import numpy as np
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from . import utils as utils
import io
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn import manifold
from scipy.spatial.distance import pdist, squareform
from ..types import IMAGE_BATCH_TYPE, DENSE_BATCH_TYPE, SUMMARY_BATCH_TYPE, IMAGE_TYPE
from .. import metrics
from pyclustering.cluster.xmeans import xmeans

from ..redis_cache import redis_cache, redis_client
from ..server import Server


class APAnalysisTorchModel(Server):
    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: tuple[int, ...],
        dataset: torch.utils.data.Dataset,
        label_names: list[str] = [],
        summary_fn_image: Callable[[IMAGE_BATCH_TYPE], SUMMARY_BATCH_TYPE] = metrics.summary_fn_image_l2,
        summary_fn_dense: Callable[[DENSE_BATCH_TYPE], SUMMARY_BATCH_TYPE] = metrics.summary_fn_dense_identity,
        log_level: Literal["info", "debug"] = "info",
        apply_relu: bool = True,
        layers_to_show: list[str] | Literal["all"] = "all",
    ):
        """
        :param apply_relu: If True, applies ReLU to Conv2d and hidden Linear layer outputs to ensure 
                          post-activation values are used. Defaults to True.
        :type apply_relu: bool, optional
        """
        redis_client.flushdb()

        self.model = model
        self.input_shape = input_shape
        self.dataset = dataset
        self.log_level = log_level
        self.summary_fn_image = summary_fn_image
        self.summary_fn_dense = summary_fn_dense
        self.apply_relu = apply_relu
        self.layers_to_show = layers_to_show
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
        self.model_graph = utils.parse_model_graph(model, input_shape, layers_to_show)
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
        async def analysis(labels: list[int], examplePerClass: int = 5, shuffle: bool = False):
            task_id = utils.create_unique_task_id()
            task(task_id, cached_analysis, labels, examplePerClass, shuffle, task_id)
            return {"message": "Analysis started", "task_id": task_id}

        def task(task_id: str, func, *args):
            thread = Thread(target=run_and_save_task_result, args=(task_id, func, *args))
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
                    "payload": json.loads(result)
                }
            result = redis_client.get(f"{task_id}-progress")
            if result:
                return {
                    "message": result.decode('utf-8'),
                    "task_id": task_id,
                    "payload": None
                }
            return {"message": "No Task found", "task_id": task_id, "payload": None }

        @self.app.get("/api/analysis/layer/{layer}/argmax")
        async def get_argmax(layer: str):
            return [np.argmax(activation[layer][0]).item() for activation in self.activations]

        @self.app.get("/api/analysis/image/{image_idx}/layer/{layer_name}/filter/{filter_index}")
        async def get_activation_images(image_idx: int, layer_name: str, filter_index: int):
            image = self.activations[image_idx][layer_name][0, :, :, filter_index]
            image -= image.min()
            image = (image - np.percentile(image, 10)) / \
            (np.percentile(image, 90) - np.percentile(image, 10))
            image = np.clip(image, 0, 1)
            # invert the image
            image = 1 - image
            image = (image * 255).astype(np.uint8)
            image = np.stack((image,)*3, axis=-1)
            img = Image.fromarray(image)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {'Content-Disposition': 'inline; filename="test.png"'}
            return Response(content, headers=headers, media_type='image/png')

        @self.app.get("/api/analysis/layer/{layer_name}/embedding")
        async def analysisLayerEmbedding(
            layer_name: str,
            normalization: Literal["none", "row", "col"] = "none",
            method: Literal["mds", "tsne", "pca", "kpca", "umap", "autoencoder", "autoencoder-pytorch"] = "umap",
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
                act_dist_mat = squareform(pdist(this_activation, metric="cityblock"))
            elif distance == "jaccard":
                binary = (this_activation > 0.5).astype(np.float64)
                act_dist_mat = squareform(pdist(binary, metric="jaccard"))
                act_dist_mat = np.nan_to_num(act_dist_mat, nan=0.0)

            if method == "pca":
                from sklearn.decomposition import PCA
                embedding_model = PCA(n_components=2)
                coords = embedding_model.fit_transform(this_activation)
                return coords.tolist()
            elif method == "kpca":
                from sklearn.decomposition import KernelPCA
                embedding_model = KernelPCA(n_components=2, kernel="precomputed")
                coords = embedding_model.fit_transform(act_dist_mat)
                return coords.tolist()
            elif method == "mds":
                embedding_model = manifold.MDS(
                    n_components=2,
                    dissimilarity="precomputed",
                    normalized_stress="auto",
                )
            elif method == "tsne":
                embedding_model = manifold.TSNE(
                    n_components=2,
                    metric="precomputed",
                    perplexity=min(30, len(this_activation) - 1),
                    init="random",
                )
            elif method == "umap":
                import umap
                embedding_model = umap.UMAP(n_components=2, metric="precomputed")
            elif method in ("autoencoder", "autoencoder-pytorch"):
                # Torch version uses PyTorch autoencoder like TF's autoencoder-pytorch
                import torch as torch_ae
                import torch.nn as nn
                import torch.optim as optim

                class EmbeddingModelAE(nn.Module):
                    def __init__(self, input_dim):
                        super().__init__()
                        self.encoder = nn.Sequential(
                            nn.Linear(input_dim, 128),
                            nn.ReLU(True),
                            nn.Linear(128, 64),
                            nn.ReLU(True),
                            nn.Linear(64, 2),
                            nn.ReLU(True),
                        )
                        self.decoder = nn.Sequential(
                            nn.Linear(2, 64),
                            nn.ReLU(True),
                            nn.Linear(64, 128),
                            nn.ReLU(True),
                            nn.Linear(128, input_dim),
                            nn.Sigmoid(),
                        )

                    def forward(self, x):
                        x = self.encoder(x)
                        x = self.decoder(x)
                        return x

                    def fit_transform(self, x, num_epochs=5000, batch_size=256):
                        norm_x = normalize(this_activation, axis=0, norm="l1")
                        x_t = torch_ae.tensor(norm_x, dtype=torch_ae.float32)
                        optimizer = optim.Adam(self.parameters(), lr=1e-3)
                        criterion = nn.BCELoss()
                        for epoch in range(num_epochs):
                            optimizer.zero_grad()
                            outputs = self(x_t)
                            loss = criterion(outputs, x_t)
                            loss.backward()
                            optimizer.step()
                        return self.encoder(x_t).detach().numpy()

                embedding_model = EmbeddingModelAE(this_activation.shape[1])
                coords = embedding_model.fit_transform(this_activation)
                return coords.tolist()

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
            return [activation[layer_name].tolist() for activation in self.activationsSummary]

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
                    n_components=2, dissimilarity="precomputed", random_state=6)
                return mds.fit(act_dist_mat).embedding_.tolist()
            return await asyncio.to_thread(_compute)


        @self.app.get("/api/analysis/images/{index}")
        async def inputImages(index: int):
            if index < 0 or index >= len(self.datasetImgs):
                return Response(status_code=404)
            image = self.datasetImgs[index][0].squeeze()
            # normalize image to 0-255
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)
            img = Image.fromarray(image)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {'Content-Disposition': 'inline; filename="test.png"'}
            return Response(content, headers=headers, media_type='image/png')


        @self.app.get("/api/analysis/layer/{layer_name}/{channel}/kernel")
        async def analysisLayerKernel(layer_name: str, channel: int):
            kernel = dict(self.model.named_modules())[layer_name].weight.data.numpy()[channel, 0, :, :]
            kernel = ((kernel - kernel.min()) / (kernel.max() -
                      kernel.min()) * 255).astype(np.uint8)
            img = Image.fromarray(kernel)
            with io.BytesIO() as output:
                img.save(output, format="PNG")
                content = output.getvalue()
            headers = {'Content-Disposition': 'inline; filename="test.png"'}
            return Response(content, headers=headers, media_type='image/png')

        @self.app.get("/api/analysis/layer/{layer_name}/cluster")
        async def analysisLayerCluster(
            layer_name: str,
            outlier_threshold: float = 0.8,
            use_xmeans: bool = True,
            k_clusters: int = 2,
        ):
            this_activation = np.array([
                activation[layer_name] for activation in self.activationsSummary
            ])

            if use_xmeans:
                xmeans_instance = xmeans(this_activation, kmax=10)
                xmeans_instance.process()

                clusters = xmeans_instance.get_clusters()
                labels_arr = [-1] * len(this_activation)
                for cluster_id, point_indices in enumerate(clusters):
                    for point_idx in point_indices:
                        labels_arr[point_idx] = cluster_id

                centers = xmeans_instance.get_centers()

                distances = []
                for point in this_activation:
                    min_dist = float("inf")
                    for center in centers:
                        dist = np.linalg.norm(point - center)
                        min_dist = min(min_dist, dist)
                    distances.append(min_dist)

                outliers = []
                for i in range(len(distances)):
                    if distances[i] > np.mean(distances) + np.std(distances) * outlier_threshold:
                        outliers.append(i)

                output = {
                    "labels": labels_arr,
                    "centers": centers,
                    "distances": distances,
                    "outliers": outliers,
                }
                return output
            else:
                kmeans = KMeans(n_clusters=k_clusters, n_init="auto")
                kmeans.fit(this_activation)

                distance_from_center = kmeans.transform(this_activation).min(axis=1)
                k_clusters_out = kmeans.n_clusters
                mean_distance_from_center = np.zeros(k_clusters_out)
                max_distance_from_center = np.zeros(k_clusters_out)
                std_distance_form_center = np.zeros(k_clusters_out)
                for i in range(k_clusters_out):
                    mean_distance_from_center[i] = distance_from_center[
                        kmeans.labels_ == i
                    ].mean()
                    max_distance_from_center[i] = distance_from_center[
                        kmeans.labels_ == i
                    ].max()
                    std_distance_form_center[i] = distance_from_center[
                        kmeans.labels_ == i
                    ].std()

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
                    "examplePerClass": len(self.datasetImgs) // len(self.selectedLabels),
                    "shuffled": self.shuffled,
                    "predictions": self.predictions,
                }
            return output

                                
                
    def _analysis(self, labels: list[int], examplePerClass: int = 50, shuffle: bool = False, progress: Callable[[str], Any] = lambda x: None):
        progress("Starting the analysis")
        self.shuffled = shuffle
        self.labels = list(labels)
        self.selectedLabels = labels

        # get layer names which are either Conv2d, Flatten or Linear
        layers = []
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Flatten) or isinstance(layer, torch.nn.BatchNorm2d):
                layers.append(layer_name)

        if self.layers_to_show != 'all':
            layers = [ln for ln in layers if ln in self.layers_to_show]

        __datasetImgs = [[] for _ in range(len(labels))]
        __activations = [[] for _ in range(len(labels))]
        __activationsSummary = [[] for _ in range(len(labels))]
        __datasetLabels = [[] for _ in range(len(labels))]

        BATCH_SIZE = 32
        collected_imgs = []
        collected_labels = []

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=self.shuffled,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        progress("Collecting and filtering images...")
        for img, label in dataloader:
            label_val = label.item()
            if label_val not in self.labels:
                continue
            label_idx = labels.index(label_val)
            if len(__datasetImgs[label_idx]) >= examplePerClass:
                if all(len(dtImgs) >= examplePerClass for dtImgs in __datasetImgs):
                    break
                continue
            collected_imgs.append(img)
            collected_labels.append(label_val)
            __datasetImgs[label_idx].append(img.numpy())
            __datasetLabels[label_idx].append(label_val)
            if all(len(dtImgs) >= examplePerClass for dtImgs in __datasetImgs):
                break

        total_images = len(collected_imgs)
        layer_types = {name: type(layer).__name__ for name, layer in self.model.named_modules()}

        progress(f"Collected {total_images} images, extracting activations in batches of {BATCH_SIZE}...")

        all_activations = []
        for batch_start in range(0, total_images, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total_images)
            batch_imgs = torch.cat(collected_imgs[batch_start:batch_end], dim=0)

            progress(f"Processing batch {batch_start // BATCH_SIZE + 1}/{(total_images + BATCH_SIZE - 1) // BATCH_SIZE}")

            batch_activation = utils.get_activations(
                self.model, batch_imgs, layer_names=layers)

            for k, v in batch_activation.items():
                batch_activation[k] = np.moveaxis(v.numpy(), 1, -1)

            if self.apply_relu:
                for layer_name in batch_activation:
                    lt = layer_types.get(layer_name, '')
                    if lt == 'Conv2d':
                        batch_activation[layer_name] = np.maximum(0, batch_activation[layer_name])
                    elif lt == 'Linear' and layer_name != layers[-1]:
                        batch_activation[layer_name] = np.maximum(0, batch_activation[layer_name])

            for idx_in_batch in range(batch_end - batch_start):
                single_activation = {
                    k: v[idx_in_batch:idx_in_batch+1] for k, v in batch_activation.items()
                }
                all_activations.append(single_activation)

        for activation, label_val in zip(all_activations, collected_labels):
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

        self.datasetImgs = [np.moveaxis(img, 1, -1) for img in self.datasetImgs]

        self.predictions = [
            np.argmax(act[layers[-1]][0]).item() for act in self.activations
        ]

        return {
            "selectedClasses": self.selectedLabels,
            "examplePerClass": len(self.datasetImgs) // len(self.selectedLabels),
            "shuffled": self.shuffled,
            "predictions": self.predictions,
        }
