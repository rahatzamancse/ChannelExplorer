from pathlib import Path

from channelexplorer.channelexplorer_torch import APAnalysisTorchModel
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Imagenette: 10-class subset of ImageNet (auto-downloads to user cache, not project dir)
data_root = Path.home() / ".cache" / "imagenette"
dataset = datasets.Imagenette(
    root=str(data_root),
    split="train",
    size="320px",
    download=True,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]),
)
# Imagenette .classes is list of tuples; normalize to list of strings for the API
labels = [
    c[0] if isinstance(c, (list, tuple)) else str(c)
    for c in dataset.classes
]

host = "localhost"
port = 8000
log_level = "info"

server = APAnalysisTorchModel(
    model=model,
    input_shape=(1, 3, 224, 224),
    dataset=dataset,
    label_names=labels,
    log_level=log_level,
    apply_relu=False,
)

server.run(host=host, port=port)