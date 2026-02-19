"""Shared type aliases and data structures for ChannelExplorer."""

from nptyping import NDArray, Float32, Shape
from typing import Optional, TypedDict

IMAGE_TYPE = NDArray[Shape["* width, * height, * channel"], Float32]
"""Single image array of shape ``(H, W, C)``."""

GRAY_IMAGE_TYPE = NDArray[Shape["* width, * height"], Float32]
"""Single-channel (grayscale) image array of shape ``(H, W)``."""

IMAGE_BATCH_TYPE = NDArray[Shape["* batch, * width, * height, * channel"], Float32]
"""Batch of images with shape ``(B, H, W, C)``."""

DENSE_BATCH_TYPE = NDArray[Shape["* batch, *"], Float32]
"""Batch of dense-layer outputs with shape ``(B, units)``."""

SUMMARY_BATCH_TYPE = NDArray[Shape["* batch, *"], Float32]
"""Batch of per-channel scalar summaries with shape ``(B, channels)``."""


class NodeInfo(TypedDict):
    """Metadata for a single layer node in the model graph.

    Attributes:
        name: Layer name as reported by the framework.
        layer_type: Layer class name (e.g. ``"Conv2D"``, ``"Linear"``).
        tensor_type: Data type string of the layer's parameters or output.
        input_shape: Shape of the layer's input tensor.
        output_shape: Shape of the layer's output tensor.
        layer_activation: Activation function name, or ``None``.
        kernel_size: Spatial kernel dimensions, or ``None`` for non-conv
            layers.
    """
    name: str
    layer_type: str
    tensor_type: str
    input_shape: list|tuple
    output_shape: list|tuple
    layer_activation: Optional[str]
    kernel_size: Optional[list|tuple]