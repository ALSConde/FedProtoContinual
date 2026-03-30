from torchvision import datasets
import torch
from torchvision.transforms import Compose, Normalize, ToTensor


def load_centralized_data(batch_size: int = 32):
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = datasets.CIFAR10(
        root="./storage/test_data", train=False, download=True, transform=transforms
    )
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)