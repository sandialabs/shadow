import pytest
import torch
import numpy as np
from sklearn import datasets
import shadow.utils


@pytest.fixture(params=['cpu', pytest.param('cuda:7', marks=pytest.mark.gpu)])
def torch_device(request):
    return torch.device(request.param)


@pytest.fixture
def simple_classification_model():
    """Simple classification model for 2D input."""
    def _make_model(seed_value=0):
        """ Create the simple model using the passed in seed value.
        Args:
            seed_value (int): The seed value to apply (default is 0).
        Returns: (torch.nn.Module) A simple model to use for testing.
        """
        # Set seeds for reproducibility
        shadow.utils.set_seed(seed_value, cudnn_deterministic=True)
        return torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 2)
        )
    return _make_model


@pytest.fixture
def simple_value_initialized_model():
    """Simple model with initialized weights."""

    def _make_model(value=0):
        # Set seeds for reproducibility
        shadow.utils.set_seed(0, cudnn_deterministic=True)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2),
                                    torch.nn.Linear(2, 2))
        shadow.utils.init_model_weights(model, value)
        return model

    return _make_model


@pytest.fixture
def identity_model():
    """No-op torch model."""
    return torch.nn.Identity()


@pytest.fixture
def simple_linear_2d_model():
    """Simple model with one layer and sigmoid."""

    def _make_model(seed_value=0):
        # Set seeds for reproducibility
        shadow.utils.set_seed(seed_value, cudnn_deterministic=True)
        model = torch.nn.Sequential(torch.nn.Linear(2, 10),
                                    torch.nn.Sigmoid(),
                                    torch.nn.Linear(10, 2))
        return model

    return _make_model


def ssml_dataset(dataset, unlabeled_frac=0, *args, **kwargs):
    r"""Make a simple torch Dataset for semi-supervised learning.

    Args:
        dataset (callable): Function that generates the toy classification
            dataset. Should return `X` and `y` ndarrays. `*args` and `**kwargs`
            are passed into this function to generate the dataset.
        unlabeled_frac (float): fraction of the data to mark as unlabeled.
            Unlabeled data are set to -1.
        *args: positional arguments passed to `dataset`.
        **kwargs: keyword arguments passed to `dataset`.

    Returns:
        torch.utils.data.TensorDataset: Dataset containing the generated
            labeled and unlabeled data. Each item consists of
            (`X`, `y_ssml`, and `y`), where `y_ssml` is set to -1 if the
            sample was unlabeled and `y` contains the original label value.

    Raises:
        ValueError: if `unlabeled_frac` is outside of the range [0, 1].
    """
    if 0 < unlabeled_frac > 1:
        raise ValueError("Unlabeled fraction must be between 0 and 1.")

    # Generate data, push to torch
    X, y = dataset(*args, **kwargs)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    # Copy the true vector
    y_ssml = y.clone()
    # Make data SSML by "unlabeling" as -1
    if unlabeled_frac != 0:
        n_samples = X.shape[0]
        unlabeled_samples = np.random.choice(
            range(n_samples), size=int(unlabeled_frac * n_samples), replace=False)
        y_ssml[unlabeled_samples] = -1
    # Return dataset
    return torch.utils.data.TensorDataset(X, y_ssml, y)


@pytest.fixture
def half_moons_ds():
    # Set seeds for reproducibility
    shadow.utils.set_seed(0, cudnn_deterministic=True)
    # Generate dataset
    return ssml_dataset(datasets.make_moons, unlabeled_frac=0, n_samples=100)


@pytest.fixture
def ssml_half_moons_ds():
    # Set seeds for reproducibility
    shadow.utils.set_seed(0, cudnn_deterministic=True)
    # Generate dataset
    return ssml_dataset(datasets.make_moons, unlabeled_frac=0.9, n_samples=100)


@pytest.fixture
def blobs_ds():
    # Set seeds for reproducibility
    shadow.utils.set_seed(0, cudnn_deterministic=True)
    # Generate dataset
    # centers is basically the number of classes and n_features is how many dimensions
    return ssml_dataset(datasets.make_blobs, unlabeled_frac=0, n_samples=100,
                        centers=2, n_features=2, center_box=(-1, 1), cluster_std=0.05)


@pytest.fixture
def train():
    def train_loop(model, optimizer, criterion, n_epochs, dataset, device):
        r"""Minimal model training loop.

        This function implements a minimal training loop for training a `model`
        against a `criterion` and `dataset` using a particular `optimizer`. No
        learning rate schedules, validation, or early stopping are employed. This
        is intended for toy datasets only as the entire dataset is loaded into a
        single large batch.

        Args:
            model (torch.nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer with which to train
                `model`. Does not support optimizers which require a closure,
                such as Conjugate Gradient or LBFGS.
            critertion (torch.nn.Module): The objective function to optimize.
                Should take in (`model(X)`, `y`, `X`) as input to its forward.
            n_epochs (int): The number of epochs to train `model`.
            dataset (torch.utils.data.Dataset): The classification training data
                to be used to train `model`. The dataset should produce `x` and
                `y`, where `y` represents class labels.

        Returns:
            y_pred (torch.Tensor): final `model` predictions over the training set.
                argmax is applied to model output.
        """
        # Dataloader - just one big batch.
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False,
                                                 batch_size=len(dataset))
        for epoch in range(n_epochs):
            for batch in dataloader:
                X, y, *rest = batch
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y, X)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Make predictions
        return torch.max(model(X), 1)[-1]
    return train_loop
