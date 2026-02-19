from nptyping import NDArray, Float32, Shape
from typing import Optional, TypedDict

IMAGE_TYPE = NDArray[Shape["* width, * height, * channel"], Float32]
GRAY_IMAGE_TYPE = NDArray[Shape["* width, * height"], Float32]
IMAGE_BATCH_TYPE = NDArray[Shape["* batch, * width, * height, * channel"], Float32]
DENSE_BATCH_TYPE = NDArray[Shape["* batch, *"], Float32]
SUMMARY_BATCH_TYPE = NDArray[Shape["* batch, *"], Float32]

class NodeInfo(TypedDict):
    name: str
    layer_type: str
    tensor_type: str
    input_shape: list|tuple
    output_shape: list|tuple
    layer_activation: Optional[str]
    kernel_size: Optional[list|tuple]