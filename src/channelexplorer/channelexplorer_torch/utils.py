from ..utils import *
from ..types import NodeInfo, IMAGE_BATCH_TYPE
import torch
from typing import Dict, Literal, Tuple

def get_activations(model: torch.nn.Module, input_img: IMAGE_BATCH_TYPE, layer_names: list[str]):
    """Extract intermediate activations from a PyTorch model.

    Registers forward hooks on the requested layers, runs a forward pass,
    and collects the detached output tensors.

    Args:
        model: A ``torch.nn.Module`` to inspect.
        input_img: Batched input tensor, e.g. shape ``(B, C, H, W)``.
        layer_names: Names of modules whose outputs should be captured
            (as returned by ``model.named_modules()``).

    Returns:
        A dict mapping each layer name to its activation tensor.
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


def parse_model_graph(model: torch.nn.Module, input_shape: Tuple[int, ...], layers_to_show: list[str] | Literal["all"] = "all") -> Dict[str, Any]:
    """Build a simplified, JSON-serializable graph of a PyTorch model.

    Converts the full model graph to a simplified version containing only
    Conv2d, Linear, and Concatenate nodes, computes a layered layout, and
    attaches per-channel kernel norms as edge weights.

    Args:
        model: A ``torch.nn.Module``.
        input_shape: Shape of a single input batch, e.g.
            ``(1, 3, 224, 224)``.  Used for tracing.
        layers_to_show: Layer names to keep, or ``"all"`` to include every
            supported layer type.

    Returns:
        A dict with keys ``"graph"`` (NetworkX node-link data),
        ``"meta"`` (graph depth), and ``"edge_weights"`` (kernel norms).
    """
    activation_pathway_full = pytorch_model_to_graph(model, input_shape)
    simple_activation_pathway_full = remove_intermediate_node(
        activation_pathway_full, lambda node: activation_pathway_full.nodes[node]['layer_type'] not in ['Conv2d', 'Linear', 'Concatenate'])

    if layers_to_show != 'all':
        simple_activation_pathway_full = remove_intermediate_node(
            simple_activation_pathway_full, lambda node: simple_activation_pathway_full.nodes[node]['name'] not in layers_to_show)

    node_pos = get_model_layout(simple_activation_pathway_full)

    if node_pos:
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
    else:
        for node in simple_activation_pathway_full.nodes():
            simple_activation_pathway_full.nodes[node]['pos'] = {'x': 0.0, 'y': 0.0}
    
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

    # Depth and first layer (guard empty graph)
    try:
        first_layer_name = next(n for n, d in simple_activation_pathway_full.in_degree() if d == 0)
        depth = max(nx.shortest_path_length(simple_activation_pathway_full, first_layer_name).values())
    except (StopIteration, nx.NetworkXError):
        depth = 0
        
    return {
        'graph': nx.node_link_data(simple_activation_pathway_full),
        'meta': {
            'depth': depth,
        },
        'edge_weights': kernel_norms,
    }
    

def _record_shapes_from_forward(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Run a forward pass with dummy data and record shapes for every module.

    Handles branching architectures (e.g. Inception) where sibling modules
    receive the same input rather than sequential output.

    Args:
        model: The PyTorch model.
        input_shape: Shape of the dummy input batch, e.g.
            ``(1, 3, 224, 224)``.

    Returns:
        A dict mapping module names to ``(input_shape, output_shape)``
        tuples.
    """
    module_to_name = {m: n for n, m in model.named_modules()}
    shapes = {}

    def hook(module, input, output):
        name = module_to_name.get(module, None)
        if name is None:
            return
        if isinstance(input, (tuple, list)):
            in_shapes = [tuple(int(x) for x in t.shape) for t in input if hasattr(t, 'shape')]
        else:
            in_shapes = [tuple(int(x) for x in input.shape)] if hasattr(input, 'shape') else []
        if hasattr(output, 'shape'):
            out_shape = tuple(int(x) for x in output.shape)
        elif isinstance(output, (tuple, list)):
            out_shape = tuple(int(x) for x in output[0].shape) if output and hasattr(output[0], 'shape') else None
        else:
            out_shape = None
        if in_shapes and out_shape is not None:
            shapes[name] = (in_shapes[0], out_shape)

    hooks = []
    for mod in model.modules():
        h = mod.register_forward_hook(hook)
        hooks.append(h)

    dummy = torch.rand(*input_shape)
    model.eval()
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass  # e.g. aux logits; shapes we got so far are still usable
    for h in hooks:
        h.remove()

    return shapes


def pytorch_model_to_graph(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> nx.Graph:
    """Convert a PyTorch model into a NetworkX directed graph.

    Each leaf module becomes a node annotated with a ``NodeInfo`` dict.
    Edges represent the data-flow order inferred from a forward pass.

    Args:
        model: A ``torch.nn.Module``.
        input_shape: Shape of a single input batch, e.g.
            ``(1, 3, 224, 224)``.

    Returns:
        A ``nx.DiGraph`` whose nodes store ``NodeInfo`` attributes.
    """
    G = nx.DiGraph()
    # One forward pass so shapes are correct for branching/concatenation (e.g. Inception)
    recorded_shapes = _record_shapes_from_forward(model, input_shape)

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
                # Use shapes from the full forward pass so branching/concatenation is correct
                if layer_id in recorded_shapes:
                    input_shape, output_shape = recorded_shapes[layer_id]
                else:
                    # Fallback for modules not run in forward (e.g. aux branch)
                    if type(layer).__name__ in ['Linear']:
                        input_shape = tuple(map(int, [1, np.prod(input_shape[1:]).tolist()]))
                    try:
                        output_shape = tuple(map(int, [1] + list(layer(torch.rand(1, *input_shape[1:])).shape[1:])))
                    except Exception:
                        continue  # skip this node if shape inference fails
                add_layer(layer, input_shape, output_shape, layer_id)

                if last_layer_name is not None:
                    G.add_edge(last_layer_name, layer_id)

                last_layer_name = layer_id
                input_shape = output_shape

        return last_layer_name, input_shape

    traverse_model(model, input_shape)

    return G
