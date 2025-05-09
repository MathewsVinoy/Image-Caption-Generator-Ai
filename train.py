import torch
from torch import nn
from model import Model
from dataSets import DatasetsCustom

EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
model = Model(input_shape=512,output_shape=82,hidden_units=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)