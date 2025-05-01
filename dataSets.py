import torch
from PIL import Image
from torch.utils.data import Dataset
from endecode import DataConverter

class DatasetsCustom(Dataset):
    def __init__(self, x, y, img_path, transform=None):
        self.transform = transform
        self.X = x.reset_index(drop=True)  # Reset index to avoid indexing issues
        self.y = y.reset_index(drop=True)  # Reset index to avoid indexing issues
        self.img_path = img_path

    def load_image(self, idx):
        image_path = f"{self.img_path}/{self.X.iloc[idx]}"
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        return image

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = self.load_image(idx)
        if self.transform:
            img = self.transform(img)
        output = self.y.iloc[idx]
        output = DataConverter.encode2(output)
        # output = torch.tensor(output, dtype=torch.float32)  # Ensure output is a tensor
        return img, output