import torch 
from torch import nn 

class Model(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super(Model, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity() 
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_shape)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        features = self.resnet(x)  
        features = features.view(batch_size, 1, -1)
        lstm_out, (h_n, c_n) = self.lstm(features)
        lstm_out_last = lstm_out[:, -1, :]  
        output = self.fc(lstm_out_last)
        return output
