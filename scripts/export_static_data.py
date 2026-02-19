#!/usr/bin/env python3
"""
Export static data from a ChannelExplorer server instance for hosting without a backend.

Usage (PyTorch example):

    from channelexplorer.channelexplorer_torch import APAnalysisTorchModel
    from scripts.export_static_data import export_static_data

    server = APAnalysisTorchModel(model, input_shape, dataset, label_names, ...)
    export_static_data(
        server,
        labels=[0, 1, 2, 3],
        examples_per_class=5,
        output_dir="frontend/public/static-data",
    )

Or run the bundled example directly:

    python scripts/export_static_data.py --labels 0 1 2 3 --examples 5

See --help for all options.
"""

import io
import json
import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn import manifold
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def _save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, cls=_NumpyEncoder)


def _save_png(img_array, path):
    Image.fromarray(img_array).save(path, format="PNG")


def export_static_data(
    server,
    labels: list[int],
    examples_per_class: int = 5,
    shuffle: bool = False,
    output_dir: str = "frontend/public/static-data",
    max_filters_per_layer: int = -1,
    max_overlay_channels: int = 40,
):
    """
    Export all data needed for a static frontend demo.

    Args:
        server: An APAnalysisTorchModel or ChannelExplorer_TF instance.
        labels: Class indices to analyse.
        examples_per_class: Number of examples per class.
        shuffle: Whether to shuffle the dataset.
        output_dir: Where to write the static files.
        max_filters_per_layer: Cap on activation-filter images per layer (-1 = all).
        max_overlay_channels: Cap on overlay/kernel channels per layer.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    file_count = 0

    # --- 1. Run analysis ------------------------------------------------
    print("[1/8] Running analysis ...")
    server._analysis(
        labels,
        examples_per_class,
        shuffle,
        progress=lambda msg: print(f"       {msg}"),
    )
    total_images = len(server.datasetImgs)
    print(f"       {total_images} images collected")

    # --- Determine which layers to export --------------------------------
    graph_data = server.model_graph
    graph_nodes = graph_data.get("graph", {}).get("nodes", [])
    layer_info = {}
    for node in graph_nodes:
        name = node.get("name", node.get("id", ""))
        layer_info[name] = {
            "type": node.get("layer_type", ""),
            "output_shape": node.get("output_shape", []),
        }

    CONV_TYPES = {"Conv2D", "Concatenate", "Conv2d", "Cat", "Add"}
    DENSE_TYPES = {"Dense", "Linear"}
    VIZ_TYPES = CONV_TYPES | DENSE_TYPES

    viz_layers = {n: i for n, i in layer_info.items() if i["type"] in VIZ_TYPES}
    conv_layers = {n: i for n, i in layer_info.items() if i["type"] in CONV_TYPES}

    # --- 2. Model graph, labels, config, predictions ---------------------
    print("[2/8] Exporting JSON metadata ...")
    _save_json(graph_data, out / "model.json"); file_count += 1
    _save_json(server.label_names, out / "labels.json"); file_count += 1

    config = {
        "selectedClasses": server.selectedLabels,
        "examplePerClass": total_images // len(server.selectedLabels),
        "shuffled": server.shuffled,
        "predictions": server.predictions,
    }
    _save_json(config, out / "config.json"); file_count += 1
    _save_json(server.predictions, out / "predictions.json"); file_count += 1

    # --- 3. Input images -------------------------------------------------
    print(f"[3/8] Exporting {total_images} input images ...")
    (out / "images" / "input").mkdir(parents=True, exist_ok=True)
    for i in range(total_images):
        image = server.datasetImgs[i][0].squeeze()
        image = (image - image.min()) / (image.max() - image.min()) * 255
        _save_png(image.astype(np.uint8), out / "images" / "input" / f"{i}.png")
        file_count += 1

    # --- 4. Per-layer JSON (heatmap, embedding, distance, cluster, argmax)
    print(f"[4/8] Exporting per-layer analysis for {len(viz_layers)} layers ...")
    for layer_name, info in viz_layers.items():
        layer_dir = out / "layers" / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Heatmap
        try:
            heatmap = [act[layer_name].tolist() for act in server.activationsSummary]
            _save_json(heatmap, layer_dir / "heatmap.json"); file_count += 1
        except Exception as e:
            print(f"       [warn] heatmap {layer_name}: {e}")

        # Embedding (MDS on cityblock, matches default frontend call)
        try:
            flat = np.array(
                [act[layer_name] for act in server.activationsSummary]
            ).reshape(len(server.activationsSummary), -1)
            dist_mat = squareform(pdist(flat, metric="cityblock"))

            try:
                import umap as _umap
                model_embed = _umap.UMAP(n_components=2, metric="precomputed")
            except ImportError:
                model_embed = manifold.MDS(
                    n_components=2, dissimilarity="precomputed", normalized_stress="auto"
                )
            coords = model_embed.fit_transform(dist_mat)
            _save_json(coords.tolist(), layer_dir / "embedding.json"); file_count += 1
        except Exception as e:
            print(f"       [warn] embedding {layer_name}: {e}")

        # Distance matrix
        try:
            flat = np.array(
                [act[layer_name] for act in server.activationsSummary]
            ).reshape(len(server.activationsSummary), -1)
            _save_json(
                squareform(pdist(flat, metric="cityblock")).tolist(),
                layer_dir / "distance.json",
            ); file_count += 1
        except Exception as e:
            print(f"       [warn] distance {layer_name}: {e}")

        # Clusters -- xmeans (default for ScatterplotView)
        try:
            flat = np.array([act[layer_name] for act in server.activationsSummary])
            from pyclustering.cluster.xmeans import xmeans as _xmeans
            xi = _xmeans(flat, kmax=10); xi.process()
            clusters = xi.get_clusters()
            lbl = [-1] * len(flat)
            for cid, pts in enumerate(clusters):
                for p in pts:
                    lbl[p] = cid
            centers = xi.get_centers()
            dists = [
                min(np.linalg.norm(pt - c) for c in centers) for pt in flat
            ]
            mean_d, std_d = np.mean(dists), np.std(dists)
            outliers = [i for i, d in enumerate(dists) if d > mean_d + std_d * 2]
            _save_json(
                {"labels": lbl, "centers": centers, "distances": dists, "outliers": outliers},
                layer_dir / "cluster_xmeans.json",
            ); file_count += 1
        except Exception as e:
            print(f"       [warn] xmeans {layer_name}: {e}")

        # Clusters -- kmeans k=3 (used by HierarchyTree)
        try:
            flat = np.array([act[layer_name] for act in server.activationsSummary])
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=3, n_init="auto"); km.fit(flat)
            d_center = km.transform(flat).min(axis=1)
            mean_d = np.zeros(km.n_clusters)
            std_d = np.zeros(km.n_clusters)
            for k in range(km.n_clusters):
                mean_d[k] = d_center[km.labels_ == k].mean()
                std_d[k] = d_center[km.labels_ == k].std()
            outliers = [
                i for i in range(len(d_center))
                if d_center[i] > mean_d[km.labels_[i]] + std_d[km.labels_[i]] * 0.8
            ]
            _save_json(
                {
                    "labels": km.labels_.tolist(),
                    "centers": km.cluster_centers_.tolist(),
                    "distances": d_center.tolist(),
                    "outliers": outliers,
                },
                layer_dir / "cluster_kmeans3.json",
            ); file_count += 1
        except Exception as e:
            print(f"       [warn] kmeans {layer_name}: {e}")

        # Argmax (Dense/Linear only)
        if info["type"] in DENSE_TYPES:
            try:
                argmax = [
                    np.argmax(act[layer_name][0]).item() for act in server.activations
                ]
                _save_json(argmax, layer_dir / "argmax.json"); file_count += 1
            except Exception as e:
                print(f"       [warn] argmax {layer_name}: {e}")

    # --- 5. Activation filter images -------------------------------------
    print(f"[5/8] Exporting activation images ...")
    for layer_name, info in conv_layers.items():
        n_filters = info["output_shape"][-1] if info["output_shape"] else 0
        if max_filters_per_layer > 0:
            n_filters = min(n_filters, max_filters_per_layer)
        act_dir = out / "images" / "activations" / layer_name
        act_dir.mkdir(parents=True, exist_ok=True)
        print(f"       {layer_name}: {n_filters} filters x {total_images} images")
        for fi in range(n_filters):
            for ii in range(total_images):
                try:
                    img = server.activations[ii][layer_name][0, :, :, fi].copy()
                    img -= img.min()
                    p10, p90 = np.percentile(img, 10), np.percentile(img, 90)
                    if p90 - p10 > 0:
                        img = (img - p10) / (p90 - p10)
                    img = np.clip(1 - img, 0, 1)
                    img = (img * 255).astype(np.uint8)
                    _save_png(
                        np.stack((img,) * 3, axis=-1),
                        act_dir / f"{ii}_{fi}.png",
                    )
                    file_count += 1
                except Exception:
                    pass

    # --- 6. Overlay images -----------------------------------------------
    print(f"[6/8] Exporting overlay images ...")
    from channelexplorer.utils import get_activation_overlay

    for layer_name, info in conv_layers.items():
        n_ch = min(info["output_shape"][-1] if info["output_shape"] else 0, max_overlay_channels)
        ov_dir = out / "images" / "overlays" / layer_name
        ov_dir.mkdir(parents=True, exist_ok=True)
        print(f"       {layer_name}: {n_ch} channels x {total_images} images")
        for ch in range(n_ch):
            for ii in range(total_images):
                try:
                    in_img = server.datasetImgs[ii][0].squeeze()
                    in_img = (in_img - in_img.min()) / (in_img.max() - in_img.min())
                    act_img = server.activations[ii][layer_name][0][:, :, ch].squeeze()
                    act_img = (act_img - act_img.min()) / (act_img.max() - act_img.min() + 1e-8)
                    ov = get_activation_overlay(in_img, act_img, alpha=0.6)
                    _save_png((ov * 255).astype(np.uint8), ov_dir / f"{ch}_{ii}.png")
                    file_count += 1
                except Exception:
                    pass

    # --- 7. Kernel images ------------------------------------------------
    print(f"[7/8] Exporting kernel images ...")
    try:
        named_modules = dict(server.model.named_modules())
    except Exception:
        named_modules = {}

    for layer_name, info in conv_layers.items():
        if info["type"] not in {"Conv2d", "Conv2D"}:
            continue
        mod = named_modules.get(layer_name)
        if mod is None:
            continue
        n_ch = min(info["output_shape"][-1] if info["output_shape"] else 0, max_overlay_channels)
        k_dir = out / "images" / "kernels" / layer_name
        k_dir.mkdir(parents=True, exist_ok=True)
        print(f"       {layer_name}: {n_ch} channels")
        for ch in range(n_ch):
            try:
                w = mod.weight.data.numpy()[ch, 0, :, :]
                w = ((w - w.min()) / (w.max() - w.min() + 1e-8) * 255).astype(np.uint8)
                _save_png(w, k_dir / f"{ch}.png")
                file_count += 1
            except Exception:
                pass

    # --- 8. All-layer embedding & distances ------------------------------
    print("[8/8] Exporting all-layer embedding & distances ...")
    try:
        all_flat = np.array([
            np.concatenate([v.ravel() for v in act.values()])
            for act in server.activationsSummary
        ])
        dist_mat = squareform(pdist(all_flat, metric="cityblock"))
        mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
        _save_json(mds.fit(dist_mat).embedding_.tolist(), out / "allembedding.json"); file_count += 1
        _save_json(dist_mat.tolist(), out / "alldistances.json"); file_count += 1
    except Exception as e:
        print(f"       [warn] all-layer: {e}")

    # --- Summary ---------------------------------------------------------
    total_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nExport complete!")
    print(f"  Files : {file_count}")
    print(f"  Size  : {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Output: {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI entry point (runs the bundled torch example)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export static demo data. By default uses the Inception-v3 / Imagenette example.",
    )
    parser.add_argument("--labels", type=int, nargs="+", default=[0, 1, 2, 3], description="Labels to analyze")
    parser.add_argument("--examples", type=int, default=5, description="Number of examples per class")
    parser.add_argument("--shuffle", action="store_true", description="Shuffle the dataset")
    parser.add_argument("--output", default="frontend/public/static-data", description="Output directory")
    parser.add_argument("--max-filters", type=int, default=-1, description="Max activation filters per layer (-1 = all)")
    parser.add_argument("--max-overlays", type=int, default=40, description="Max overlay/kernel channels per layer")
    args = parser.parse_args()

    # Re-use the same setup as examples/run_torch.py
    from channelexplorer.channelexplorer_torch import APAnalysisTorchModel
    import torch, torchvision.models as models, torchvision.datasets as datasets, torchvision.transforms as transforms

    print("Loading Inception-v3 + Imagenette ...")
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    data_root = Path.home() / ".cache" / "imagenette"
    dataset = datasets.Imagenette(
        root=str(data_root), split="train", size="320px", download=True,
        transform=transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    )
    label_names = [c[0] if isinstance(c, (list, tuple)) else str(c) for c in dataset.classes]

    server = APAnalysisTorchModel(
        model=model, input_shape=(1, 3, 224, 224), dataset=dataset,
        label_names=label_names, apply_relu=False,
    )

    export_static_data(
        server,
        labels=args.labels,
        examples_per_class=args.examples,
        shuffle=args.shuffle,
        output_dir=args.output,
        max_filters_per_layer=args.max_filters,
        max_overlay_channels=args.max_overlays,
    )
