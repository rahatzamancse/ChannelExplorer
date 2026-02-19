from itertools import combinations
import numpy as np
from typing import Callable, Dict, Any
from ast import literal_eval
import re
import networkx as nx
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from .types import NodeInfo, IMAGE_TYPE, GRAY_IMAGE_TYPE
import grandalf
from grandalf.layouts import SugiyamaLayout
from PIL import Image, ImageDraw
from itertools import combinations
from typing import Callable
import time
import uuid
import pathlib
import shutil

def makedir(path):
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    
def delete_dir(path):
    shutil.rmtree('path', ignore_errors=True)
    
def zip_dir(path, name: str):
    import shutil
    shutil.make_archive(name, 'zip', path)

def create_unique_task_id():
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
    '''
    Loop over the graph until all nodes that match the supplied predicate 
    have been removed and their incident edges fused.
    src: https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges
    '''
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
    # Sugiyama Layout from grandalf library
    g = grandalf.utils.convert_nextworkx_graph_to_grandalf(G)
    for v in g.V(): v.view = type("defaultview", (object,), {"w": 10, "h": 10})
    sug = grandalf.layouts.SugiyamaLayout(g.C[0])
    sug.init_all()
    sug.draw() # This only calculated the positions for each node.
    pos = {v.data: (v.view.xy[0], -v.view.xy[1]) for v in g.C[0].sV} # Extracts the positions
    return pos

def is_numpy_type(value):
    return hasattr(value, 'dtype')

def apply_mask(img, mask_img):
    if len(img.shape) == 4:
        return np.multiply(img, mask_img[np.newaxis,:,:,np.newaxis])
    elif len(img.shape) == 3:
        return np.multiply(img, mask_img[:,:,np.newaxis])
    elif len(img.shape) == 2:
        return np.multiply(img, mask_img[:,:])
    raise Exception
    
def get_mask_img(polygon, size: tuple[int,int]):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    polygon_points = [(
        p[0].item() if is_numpy_type(p[0]) else p[0],
        p[1].item() if is_numpy_type(p[1]) else p[1]
    ) for p in polygon]
    draw.polygon(polygon_points, outline=1, fill=1)

    return np.array(mask)


def single_activation_distance(activation1_summary: np.ndarray, activation2_summary: np.ndarray):
    return np.sum(np.abs(activation1_summary - activation2_summary))


def single_activation_jaccard_distance(activation1_summary: np.ndarray, activation2_summary: np.ndarray, threshold: float = 0.5):
    assert len(activation1_summary) == len(activation2_summary)
    activation1_summary = activation1_summary > threshold
    activation2_summary = activation2_summary > threshold
    size = len(activation1_summary)
    num_differences = sum(
        int(activation1_summary[i] != activation2_summary[i]) for i in range(size))
    distance = num_differences / size
    return distance


def activation_distance(activation1_summary: dict[str, np.ndarray], activation2_summary: dict[str, np.ndarray]):
    dist = 0
    for act1, act2 in zip(activation1_summary.values(), activation2_summary.values()):
        dist += single_activation_distance(act1, act2)
    return dist


def rescale_img(img):
    img = img.squeeze()
    img = img - img.min()
    img = img / img.max()
    img *= 255
    img = img.astype(np.uint8)
    return img