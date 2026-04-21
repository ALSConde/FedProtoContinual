import torchvision
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
)
from client.plugins.GlobalLossPlugin import GlobalForwardPlugin
from client.plugins.LwFPlugin import LwFPlugin
import torch
from model.Models import Model
from client.strategies.NaiveStrategy import NaiveStrategy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    data_test = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./storage/test_data",
            train=False,
            download=True,
            transform=Compose(
                [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            ),
        ),
        batch_size=32,
        shuffle=False,
        pin_memory=True,
    )
    data_train = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./storage/train_data",
            train=True,
            download=True,
            transform=Compose(
                [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            ),
        ),
        batch_size=32,
        shuffle=True,
        pin_memory=True,
    )
    lwf_plugin = LwFPlugin()
    global_forward_plugin = GlobalForwardPlugin(beta=1.5)
    plugins = [lwf_plugin, global_forward_plugin]
    strategy = NaiveStrategy(
        model,
        optimizer,
        loss_fn,
        epochs=10,
        device=device,
        data_test=data_test,
        data_train=data_train,
        plugins=plugins,
    )
    strategy.start()
