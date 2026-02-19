from channelexplorer.channelexplorer_torch import APAnalysisTorchModel
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from nltk.corpus import wordnet as wn

model = models.vgg16(pretrained=True)
# model = models.inception_v3(pretrained=True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

dataset = datasets.ImageFolder(
    '/run/media/insane/My 4TB 2/Big Data/pytorch-data',
    transform=transforms.Compose([
        transforms.Resize(320),
        transforms.ToTensor(),
    ])
)
with open('/run/media/insane/My 4TB 2/Big Data/pytorch-data/labels.txt', 'r') as f:
    labels = f.read().split('\n')

host = "localhost"
port = 8000
log_level = "info"

server = APAnalysisTorchModel(
    model=model,
    input_shape=(1, 3, 224, 224),
    dataset=dataset,
    label_names=labels,
    log_level=log_level,
)

server.run_server(host=host, port=port)