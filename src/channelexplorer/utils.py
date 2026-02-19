import pathlib
import shutil
import time
import uuid
from itertools import combinations
from typing import Callable, Dict, Any

import grandalf.utils
import grandalf.layouts
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from .types import NodeInfo, IMAGE_TYPE, GRAY_IMAGE_TYPE

def makedir(path):
    """Create a directory and any missing parents.

    Args:
        path: Filesystem path to create.
    """
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    
def delete_dir(path):
    """Recursively delete a directory, ignoring errors if it doesn't exist.

    Args:
        path: Filesystem path to delete.
    """
    shutil.rmtree(path, ignore_errors=True)
    
def zip_dir(path, name: str):
    """Create a ``.zip`` archive of a directory.

    Args:
        path: Directory to compress.
        name: Base name for the archive (without extension).
    """
    import shutil
    shutil.make_archive(name, 'zip', path)

def create_unique_task_id():
    """Generate a unique task identifier from a timestamp and UUID.

    Returns:
        A string in the form ``"<unix_timestamp>-<hex_uuid>"``.
    """
    timestamp = int(time.time())
    random_uuid = uuid.uuid4().hex
    task_id = f"{timestamp}-{random_uuid}"
    return task_id

# TODO: try out if string jet works instead of plt.cm.jet in cmap
def get_activation_overlay(input_img: IMAGE_TYPE, activation: GRAY_IMAGE_TYPE, cmap=plt.cm.jet, alpha=0.3) -> IMAGE_TYPE:
    """Draws the activation overlay on the input image with the given colormap and alpha transparency

    Args:
        input_img (IMAGE_TYPE): The input image
        activation (GRAY_IMAGE_TYPE): The activation image
        cmap (str, optional): The color map. Defaults to plt.cm.jet.
        alpha (float, optional): The alpha used for blending the images. Defaults to 0.3.

    Returns:
        IMAGE_TYPE: The final image of the shape of input_img
    """
    act_img = Image.fromarray(activation)
    act_img = act_img.resize((input_img.shape[1], input_img.shape[0]), Image.BILINEAR)
    act_img = np.array(act_img)
    # normalize act_img
    act_img = (act_img - act_img.min()) / (act_img.max() - act_img.min())
    act_rgb = cmap(act_img)
    
    # normalize input_img
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    
    # convert to rgb if input_img is grayscale
    if input_img.ndim == 2:
        input_img = np.repeat(input_img[...,None], 3, axis=2)

    # Blend act_img to original image
    out_img = np.zeros(input_img.shape, dtype=input_img.dtype)
    out_img[:,:,:] = ((1-alpha) * input_img[:,:,:] + alpha * act_rgb[:,:,:3]).astype(input_img.dtype)
    
    out_img = (out_img - out_img.min()) / (out_img.max() - out_img.min())
    return out_img

def remove_intermediate_node(G: nx.Graph, node_removal_predicate: Callable):
    """Remove nodes matching a predicate and reconnect their neighbors.

    Iterates the graph until every node satisfying
    ``node_removal_predicate`` has been removed and the edges of its
    predecessors / successors have been fused together.

    Args:
        G: A NetworkX graph (directed or undirected).
        node_removal_predicate: Callable that receives a node and returns
            ``True`` if the node should be removed.

    Returns:
        A copy of *G* with the matched nodes removed and edges fused.
    """
    g = G.copy()
    while any(node_removal_predicate(node) for node in g.nodes):

        g0 = g.copy()

        for node in g.nodes:
            if node_removal_predicate(node):

                if g.is_directed():
                    in_edges_containing_node = list(g0.in_edges(node))
                    out_edges_containing_node = list(g0.out_edges(node))

                    for in_src, _ in in_edges_containing_node:
                        for _, out_dst in out_edges_containing_node:
                            g0.add_edge(in_src, out_dst)
                            # dist = nx.shortest_path_length(
                            #   g0, in_src, out_dst, weight='weight'
                            # )
                            # g0.add_edge(in_src, out_dst, weight=dist)
                else:
                    edges_containing_node = g.edges(node)
                    dst_to_link = [e[1] for e in edges_containing_node]
                    dst_pairs_to_link = list(combinations(dst_to_link, r = 2))
                    for pair in dst_pairs_to_link:
                        g0.add_edge(pair[0], pair[1])
                        # dist = nx.shortest_path_length(
                        # g0, pair[0], pair[1], weight='weight'
                        # )
                        # g0.add_edge(pair[0], pair[1], weight=dist)
                
                g0.remove_node(node)
                break
        g = g0
    return g


def get_model_layout(G):
    """Compute a Sugiyama (layered) layout for a directed graph.

    Uses the grandalf library to produce ``(x, y)`` positions suitable for
    rendering a neural-network graph top-to-bottom.

    Args:
        G: A NetworkX directed graph.

    Returns:
        A dict mapping each node to an ``(x, y)`` coordinate tuple.
    """
    if G.number_of_nodes() == 0:
        return {}
    g = grandalf.utils.convert_nextworkx_graph_to_grandalf(G)
    for v in g.V(): v.view = type("defaultview", (object,), {"w": 10, "h": 10})
    if not g.C:
        return {n: (0.0, 0.0) for n in G.nodes()}
    sug = grandalf.layouts.SugiyamaLayout(g.C[0])
    sug.init_all()
    sug.draw() # This only calculated the positions for each node.
    pos = {v.data: (v.view.xy[0], -v.view.xy[1]) for v in g.C[0].sV} # Extracts the positions
    return pos

def is_numpy_type(value):
    """Check whether *value* is a NumPy scalar or array (has a ``dtype``).

    Args:
        value: Any Python object.

    Returns:
        ``True`` if *value* has a ``dtype`` attribute.
    """
    return hasattr(value, 'dtype')

def apply_mask(img, mask_img):
    """Element-wise multiply an image (or batch) by a 2-D binary mask.

    Supports 2-D, 3-D (H×W×C), and 4-D (B×H×W×C) images.

    Args:
        img: Image array of 2, 3, or 4 dimensions.
        mask_img: 2-D mask array of shape ``(height, width)``.

    Returns:
        The masked image with the same shape as *img*.

    Raises:
        Exception: If *img* has an unsupported number of dimensions.
    """
    if len(img.shape) == 4:
        return np.multiply(img, mask_img[np.newaxis,:,:,np.newaxis])
    elif len(img.shape) == 3:
        return np.multiply(img, mask_img[:,:,np.newaxis])
    elif len(img.shape) == 2:
        return np.multiply(img, mask_img[:,:])
    raise Exception
    
def get_mask_img(polygon, size: tuple[int,int]):
    """Rasterize a polygon into a binary mask image.

    Args:
        polygon: Sequence of ``(x, y)`` vertices defining the polygon.
        size: Output mask size as ``(width, height)``.

    Returns:
        A NumPy array of shape ``(height, width)`` with 1 inside the
        polygon and 0 outside.
    """
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    polygon_points = [(
        p[0].item() if is_numpy_type(p[0]) else p[0],
        p[1].item() if is_numpy_type(p[1]) else p[1]
    ) for p in polygon]
    draw.polygon(polygon_points, outline=1, fill=1)

    return np.array(mask)


def single_activation_distance(activation1_summary: np.ndarray, activation2_summary: np.ndarray):
    """Compute the L1 (Manhattan) distance between two activation summaries.

    Args:
        activation1_summary: 1-D summary vector for the first activation.
        activation2_summary: 1-D summary vector for the second activation.

    Returns:
        The sum of absolute element-wise differences.
    """
    return np.sum(np.abs(activation1_summary - activation2_summary))


def single_activation_jaccard_distance(activation1_summary: np.ndarray, activation2_summary: np.ndarray, threshold: float = 0.5):
    """Compute the Jaccard distance between two binarized activation summaries.

    Values above *threshold* are treated as active; the distance is the
    fraction of positions where the two binary vectors disagree.

    Args:
        activation1_summary: 1-D summary vector for the first activation.
        activation2_summary: 1-D summary vector for the second activation.
        threshold: Binarization threshold.

    Returns:
        A float in ``[0, 1]`` representing the Jaccard distance.
    """
    assert len(activation1_summary) == len(activation2_summary)
    activation1_summary = activation1_summary > threshold
    activation2_summary = activation2_summary > threshold
    size = len(activation1_summary)
    num_differences = sum(
        int(activation1_summary[i] != activation2_summary[i]) for i in range(size))
    distance = num_differences / size
    return distance


def activation_distance(activation1_summary: dict[str, np.ndarray], activation2_summary: dict[str, np.ndarray]):
    """Compute the total L1 distance across all layers between two summaries.

    Args:
        activation1_summary: Dict mapping layer names to summary arrays.
        activation2_summary: Dict mapping layer names to summary arrays.

    Returns:
        The cumulative L1 distance summed over all layers.
    """
    dist = 0
    for act1, act2 in zip(activation1_summary.values(), activation2_summary.values()):
        dist += single_activation_distance(act1, act2)
    return dist


def rescale_img(img):
    """Rescale an image to the ``[0, 255]`` uint8 range.

    Squeezes leading dimensions, shifts the minimum to zero, scales the
    maximum to 255, and converts to ``uint8``.

    Args:
        img: Input image array (may have extra leading dimensions).

    Returns:
        A ``uint8`` NumPy array with values in ``[0, 255]``.
    """
    img = img.squeeze()
    img = img - img.min()
    img = img / img.max()
    img *= 255
    img = img.astype(np.uint8)
    return img