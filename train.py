import torch
from torch import nn
from model import Model
from dataSets import DatasetsCustom

EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)