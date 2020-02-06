import torch

import shadow.utils
import shadow.losses


def test_supervised_half_moons(torch_device, simple_classification_model, half_moons_ds, train):
    """Simple integration test of fully supervised learning for half moons."""
    shadow.utils.set_seed(0, cudnn_deterministic=True)
    # Define model
    model = simple_classification_model().to(torch_device)
    # Optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)

    def criterion(y_pred, y, x):
        return xEnt(y_pred, y)

    # Do training and test accuracy
    y_pred = train(
        model, optimizer, criterion, n_epochs=100,
        dataset=half_moons_ds, device=torch_device)
    train_acc = shadow.losses.accuracy(y_pred, half_moons_ds.tensors[-1].to(torch_device))
    assert train_acc.item() > 98
