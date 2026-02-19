from ..utils import *
from ..types import NodeInfo, IMAGE_BATCH_TYPE
import torch
from typing import Tuple

def get_activations(model: torch.nn.Module, input_img: IMAGE_BATCH_TYPE, layer_names: list[str]):
    """
    Get activations from a CNN model for the given input image.

    Parameters:
    - model: PyTorch CNN model.
    - input_img: Preprocessed input image tensor.
    - layer_names: List of layer names to extract activations from.

    Returns:
    - Dictionary with layer names as keys and activation tensors as values.
    """
    activations = {}
    module_to_name = {}
    
    for name, module in model.named_modules():
        module_to_name[module] = name

    # Function to be called when forward pass is done on a layer
    def hook_fn(module, input, output):
        activations[module_to_name[module]] = output.detach()

    # Register hook for each layer
    hooks = []
    for name, layer in model.named_modules():
        if name in layer_names:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Perform a forward pass to trigger the hooks
    model.eval()
    with torch.no_grad():
        model(input_img)

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    return activations


def parse_model_graph(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    activation_pathway_full = pytorch_model_to_graph(model, input_shape)
    simple_activation_pathway_full = remove_intermediate_node(
        activation_pathway_full, lambda node: activation_pathway_full.nodes[node]['layer_type'] not in ['Conv2d', 'Linear', 'Concatenate'])

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
    # loop over torch module
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            kernel_norm = np.linalg.norm(layer.weight.detach().numpy(), axis=(0,1), ord=2).sum(axis=0)
            kernel_norm = kernel_norm.tolist()
            if not isinstance(kernel_norm, list):
                kernel_norm = [kernel_norm]
            kernel_norms[name] = kernel_norm
            
    # Get the node name whose in degree is 0
    first_layer_name = next(n for n, d in simple_activation_pathway_full.in_degree() if d == 0)
        
    return {
        'graph': nx.node_link_data(simple_activation_pathway_full),
        'meta': {
            'depth': max(nx.shortest_path_length(simple_activation_pathway_full, first_layer_name).values())
        },
        'edge_weights': kernel_norms,
    }
    

def pytorch_model_to_graph(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> nx.Graph:
    G = nx.DiGraph()

    def add_layer(layer, input_shape, output_shape, layer_name):
        # Extract layer details
        layer_type = type(layer).__name__
        # If layer has any parameters, get the type of the first parameter
        if list(layer.parameters()):
            tensor_type = str(next(layer.parameters()).dtype)
        else:
            tensor_type = 'no parameter'
        layer_activation = None
        kernel_size = None

        # Check for specific layer types
        if isinstance(layer, torch.nn.Conv2d):
            kernel_size = layer.kernel_size
            
        # Create node info
        node_info = NodeInfo(
            name=layer_name,
            layer_type=layer_type,
            tensor_type=tensor_type,
            input_shape=input_shape,
            output_shape=output_shape,
            layer_activation=layer_activation,
            kernel_size=kernel_size
        )

        # Add node to graph
        G.add_node(layer_name, **node_info)

    def traverse_model(module, input_shape, prefix='', last_layer_name=None):
        for name, layer in module.named_children():
            layer_id = prefix + ('.' if prefix else '') + name
            if list(layer.children()):
                last_layer_name, input_shape = traverse_model(layer, input_shape, layer_id, last_layer_name)
            else:
                if type(layer).__name__ in ['Linear']:
                    input_shape = tuple(map(int, [1, np.prod(input_shape[1:]).tolist()]))
                # Assume batch size of 1 for shape; modify as needed
                output_shape = tuple(map(int, [1] + list(layer(torch.rand(1, *input_shape[1:])).shape[1:])))
                add_layer(layer, input_shape, output_shape, layer_id)

                if last_layer_name is not None:
                    G.add_edge(last_layer_name, layer_id)

                last_layer_name = layer_id
                input_shape = output_shape

        return last_layer_name, input_shape

    traverse_model(model, input_shape)

    return G
