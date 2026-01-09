import numpy as np
from typing import Dict, Any, Literal
from ast import literal_eval
import re
import tensorflow as tf
import tensorflow.keras as K
import networkx as nx
from ..types import NodeInfo, IMAGE_TYPE, GRAY_IMAGE_TYPE
from ..utils import *


def pydot_to_networkx(P):
    """Convert a pydot graph to a networkx graph.
    
    Custom implementation to work around compatibility issues between
    networkx and newer pydot versions where get_strict() signature changed.
    """
    # Determine if graph is directed
    if P.get_type() == "graph":
        N = nx.Graph()
    else:
        N = nx.DiGraph()

    # Set graph attributes
    graph_attrs = {k: v for k, v in P.get_graph_defaults()}
    N.graph.update(graph_attrs)

    def strip_quotes(s):
        """Strip surrounding quotes from a string."""
        if isinstance(s, str) and len(s) >= 2:
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                return s[1:-1]
        return s

    # Add nodes
    for node in P.get_nodes():
        node_name = node.get_name()
        if node_name in ('node', 'edge', 'graph'):
            continue
        # Strip quotes from node name if present
        node_name = strip_quotes(node_name)
        # Get attributes - pydot returns them with quotes, so strip them
        node_attrs = {}
        for k, v in node.get_attributes().items():
            node_attrs[k] = strip_quotes(v) if isinstance(v, str) else v
        N.add_node(node_name, **node_attrs)

    # Add edges
    for edge in P.get_edges():
        src = strip_quotes(edge.get_source())
        dst = strip_quotes(edge.get_destination())
        edge_attrs = {}
        for k, v in edge.get_attributes().items():
            edge_attrs[k] = strip_quotes(v) if isinstance(v, str) else v
        N.add_edge(src, dst, **edge_attrs)

    return N

# def get_example(ds, count=1) -> tuple[np.ndarray, int]:
#     if count == 1:
#         return next(ds.shuffle(10).take(count).as_numpy_iterator())
#     else:
#         return list(ds.shuffle(10).take(count).as_numpy_iterator())


def parse_model_graph(model: K.Model, layers_to_show: Literal["all"]|list[str] = 'all') -> Dict[str, Any]:
    activation_pathway_full = tensorflow_model_to_graph(model)
    simple_activation_pathway_full = remove_intermediate_node(
        activation_pathway_full, lambda node: activation_pathway_full.nodes[node]['layer_type'] not in ['Conv2D', 'Dense', 'InputLayer', 'Concatenate'])
    
    if layers_to_show != 'all':
        simple_activation_pathway_full = remove_intermediate_node(
            simple_activation_pathway_full, lambda node: activation_pathway_full.nodes[node]['name'] not in layers_to_show)
        
    # Remove duplicate edges
    new_simple_activation_pathway_full = nx.DiGraph()
    # copy nodes
    for node, node_data in simple_activation_pathway_full.nodes(data=True):
        new_simple_activation_pathway_full.add_node(node, **node_data)
    # copy edges
    for edge in simple_activation_pathway_full.edges:
        if new_simple_activation_pathway_full.has_edge(edge[0], edge[1]):
            continue
        new_simple_activation_pathway_full.add_edge(edge[0], edge[1])
    simple_activation_pathway_full = new_simple_activation_pathway_full
    
    node_pos = get_model_layout(simple_activation_pathway_full)

    # normalize node positions to be between 0 and 1
    min_x = min(pos[0] for pos in node_pos.values())
    max_x = max(pos[0] for pos in node_pos.values())
    min_y = min(pos[1] for pos in node_pos.values())
    max_y = max(pos[1] for pos in node_pos.values())

    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1

    node_pos = {
        node: ((pos[0] - min_x)/(max_x - min_x), (pos[1] - min_y)/(max_y - min_y)) for node, pos in node_pos.items()
    }

    for node, pos in node_pos.items():
        simple_activation_pathway_full.nodes[node]['pos'] = {
            'x': pos[0], 'y': pos[1]}
    
    # Edge weights
    kernel_norms = {}
    for layer_id, layer_data in simple_activation_pathway_full.nodes(data=True):
        if layer_data['layer_type'] in ['Concatenate', 'InputLayer', 'Dense']:
            continue
        layer_name = layer_data['name']
        kernel = model.get_layer(layer_name).get_weights()
        kernel_norm = np.linalg.norm(kernel[0], axis=(0,1), ord=2).sum(axis=0)
        kernel_norms[layer_name] = kernel_norm.tolist()
        # kernel = ((kernel - kernel.min()) / (kernel.max() -
        #           kernel.min()) * 255).astype(np.uint8)
        # img = Image.fromarray(kernel)
        
    return {
        'graph': nx.node_link_data(simple_activation_pathway_full),
        'meta': {
            'depth': max(nx.shortest_path_length(simple_activation_pathway_full, next(n for n, d in simple_activation_pathway_full.nodes(data=True) if d['layer_type'] == 'InputLayer')).values())
        },
        'edge_weights': kernel_norms,
    }
    
def shuffle_or_noshuffle(dataset: tf.data.Dataset, shuffle: bool = False):
    """Shuffles the dataset if shuffle is True

    Args:
        dataset (tf.data.Dataset): The dataset
        shuffle (bool, optional): Whether to shuffle or not. Defaults to False.

    Returns:
        tf.data.Dataset: The shuffled or unshuffled dataset
    """
    if shuffle:
        # TODO: make the shuffling for whole data instead of first 10000 data
        # https://stackoverflow.com/questions/44792761/how-can-i-shuffle-a-whole-dataset-with-tensorflow
        # get the seed
        seed = np.random.randint(0, 1000)
        print("Shuffling with seed", seed)
        return dataset.shuffle(buffer_size=10000, seed=seed)
    else:
        return dataset

def parse_tensorflow_dot_label(label: str) -> NodeInfo:
    """Parses the label of the layer in tensorflow.keras.utils.model_to_dot

    Args:
        label (str): The label of the layer in tensorflow.keras.utils.model_to_dot

    Returns:
        NodeInfo: The parsed result.
    """
    # Strip surrounding quotes if present
    if label.startswith('"') and label.endswith('"'):
        label = label[1:-1]
    # Handle escaped quotes
    label = label.replace('\\"', '"')
    
    # Check if it's the new HTML-style format (TensorFlow 2.x)
    if '<table' in label:
        return _parse_html_label(label)
    
    # Old pipe-separated format
    pattern = re.compile(r"\{([\w\d_]+)\|(\{[\w\d_]+\|[\w\d_]+\}|[\w\d_]+)\|([\w\d_]+)\}\|\{input:\|output:\}\|\{\{([\[\]\(\),\w\d_ ]*)\}\|\{([\[\]\(\),\w\d_ ]*)\}\}")
    match = pattern.findall(label)
    
    if not match:
        raise ValueError(f"Could not parse layer label: {label!r}")
    
    name, layer_type, tensor_type, input_shape, output_shape = match[-1]
    layer_activation = None
    if '|' in layer_type:
        layer_type, layer_activation = layer_type.split('|')
        layer_type = layer_type[1:]
        layer_activation = layer_activation[:-1]
    ret: NodeInfo = {
        'name': name,
        'layer_type': layer_type,
        'tensor_type': tensor_type,
        'input_shape': literal_eval(input_shape),
        'output_shape': literal_eval(output_shape),
        'layer_activation': layer_activation,
        'kernel_size': None
    }
    return ret


def _parse_html_label(label: str) -> NodeInfo:
    """Parse the new HTML-style label format from TensorFlow 2.x model_to_dot"""
    
    # Extract name and layer type: <b>name</b> (LayerType)
    name_pattern = re.compile(r'<b>(\w+)</b>\s*\((\w+)\)')
    name_match = name_pattern.search(label)
    if not name_match:
        raise ValueError(f"Could not parse layer name/type from: {label!r}")
    name = name_match.group(1)
    layer_type = name_match.group(2)
    
    # Extract output shape: Output shape: <b>(shape)</b>
    output_shape_pattern = re.compile(r'Output shape:\s*<b>\(([^)]*)\)</b>')
    output_shape_match = output_shape_pattern.search(label)
    if output_shape_match:
        output_shape_str = f"({output_shape_match.group(1)})"
        output_shape = literal_eval(output_shape_str)
    else:
        output_shape = None
    
    # Extract input shape if present: Input shape: <b>(shape)</b>
    input_shape_pattern = re.compile(r'Input shape:\s*<b>\(([^)]*)\)</b>')
    input_shape_match = input_shape_pattern.search(label)
    if input_shape_match:
        input_shape_str = f"({input_shape_match.group(1)})"
        input_shape = literal_eval(input_shape_str)
    else:
        input_shape = output_shape  # For InputLayer, input=output
    
    # Extract dtype: Output dtype: <b>dtype</b>
    dtype_pattern = re.compile(r'Output dtype:\s*<b>(\w+)</b>')
    dtype_match = dtype_pattern.search(label)
    tensor_type = dtype_match.group(1) if dtype_match else 'float32'
    
    # Extract activation if present: Activation: <b>activation</b>
    activation_pattern = re.compile(r'Activation:\s*<b>(\w+)</b>')
    activation_match = activation_pattern.search(label)
    layer_activation = activation_match.group(1) if activation_match else None
    
    ret: NodeInfo = {
        'name': name,
        'layer_type': layer_type,
        'tensor_type': tensor_type,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'layer_activation': layer_activation,
        'kernel_size': None
    }
    return ret

def tensorflow_model_to_graph(model: K.Model) -> nx.Graph:
    """Converts a tensorflow.keras model to a networkx graph

    Args:
        model (K.Model): The model

    Returns:
        nx.Graph: The networkx graph
    """
    dot = K.utils.model_to_dot(
        model,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96,
        subgraph=True,
        layer_range=None,
        show_layer_activations=True
    )
    G = pydot_to_networkx(dot)

    all_node_info = {}
    for node, node_data in G.nodes(data=True):
        node_info = parse_tensorflow_dot_label(node_data['label'])
        
        # Get kernel size
        if node_info['layer_type'] == 'Conv2D':
            node_info['kernel_size'] = model.get_layer(node_info['name']).kernel_size
        elif node_info['layer_type'] == 'MaxPooling2D':
            node_info['kernel_size'] = model.get_layer(node_info['name']).pool_size
        elif node_info['layer_type'] == 'Dense':
            node_info['kernel_size'] = model.get_layer(node_info['name']).units
        else:
            node_info['kernel_size'] = ()

        all_node_info[node] = node_info

    nx.set_node_attributes(G, all_node_info)
    return G

def get_mask_activation_channels(mask_img, activations, summary_fn_image, threshold_fn=lambda layer, channel: channel > np.percentile(layer, 99)):
    masked_activations = {layer:[] for layer in activations}
    activated_channels = {layer:[] for layer in masked_activations}
    for layer, val in activations.items():
        mask = tf.image.resize(mask_img[:,:,np.newaxis], val.shape[1:3], method=tf.image.ResizeMethod.BILINEAR).numpy().squeeze()
        for channel_i, channel in enumerate(val[0].transpose(2,0,1)):
            masked_val = apply_mask(channel, mask)
            masked_val_summary = summary_fn_image(masked_val[np.newaxis,:,:,np.newaxis]).squeeze()
            masked_activations[layer].append(masked_val_summary.item())
        
        for channel_i, channel in enumerate(masked_activations[layer]):
            if threshold_fn(masked_activations[layer], channel):
                activated_channels[layer].append(channel_i)
    
#     return masked_activations
    return activated_channels
