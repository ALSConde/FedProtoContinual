import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def partition_data_dirichlet(dataset, n_clients, alpha):
    labels = np.array(dataset.targets)
    n_classes = len(np.unique(labels))
    
    client_idcs = [[] for _ in range(n_clients)]
    
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        np.random.shuffle(idx_c)
        
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        
        proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        
        idx_split = np.split(idx_c, proportions)
        for i in range(n_clients):
            client_idcs[i].extend(idx_split[i])

    return client_idcs

# --- Use case ---

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    N_CLIENTS = 10
    ALPHA = 1000.0  # low alpha means more heterogeneity (more than 100 is considered IID)

    indices_por_cliente = partition_data_dirichlet(train_dataset, N_CLIENTS, ALPHA)

    client_0_subset = Subset(train_dataset, indices_por_cliente[0])
    client_0_loader = DataLoader(client_0_subset, batch_size=32, shuffle=True)

    print(f"Total samples in Client 0: {len(client_0_subset)}")
    print(f"Class distribution in Client 0: {np.bincount(np.array(train_dataset.targets)[indices_por_cliente[0]])}")