import torch


def test(net, test_loader, criterion, device):
    net.to(device)
    correct, loss = 0, 0.0

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        accuracy = correct / len(test_loader.dataset)
        loss = loss / len(test_loader)

        return accuracy, loss
