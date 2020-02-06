import numpy as np
from scipy.stats import chisquare
import torch
import shadow.rpt
import shadow.utils
import shadow.losses


def test_semisupervised_half_moons(torch_device, simple_classification_model, ssml_half_moons_ds, train):
    """Simple integration test of fully supervised learning for half moons."""
    dataset = ssml_half_moons_ds
    y = dataset.tensors[-1].to(torch_device)
    # First, compute performance on supervised baseline
    baseline = simple_classification_model().to(torch_device)
    # Optimizer and criterion
    optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1, momentum=0.9)
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)

    def criterion(y_pred, y, x):
        return xEnt(y_pred, y)

    # train
    y_pred = train(
        baseline, optimizer, criterion, n_epochs=1000,
        dataset=dataset, device=torch_device
    )
    train_acc = shadow.losses.accuracy(y_pred, y)

    # Next, compute performance via the SSML technique
    model = simple_classification_model().to(torch_device)
    rpt = shadow.rpt.RPT(eps=0.3, model=model, consistency_type='mse')
    # Optimizer and criterion
    optimizer = torch.optim.SGD(rpt.parameters(), lr=0.1, momentum=0.9)
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)

    def loss(y_pred, y, x):
        return xEnt(y_pred, y) + rpt.get_technique_cost(x)

    # Do training and test accuracy
    y_pred = train(
        rpt, optimizer, loss, n_epochs=5000,
        dataset=dataset, device=torch_device
    )
    ssml_acc = shadow.losses.accuracy(y_pred, y.to(torch_device))

    # Assert that RPT improves performance in the low label setting
    ssml_acc_val = ssml_acc.item()
    bl_acc_val = train_acc.item()
    print("\n" + str(torch_device) + " ssml=" + str(ssml_acc_val) + ", baseline=" + str(bl_acc_val))
    assert ssml_acc_val > bl_acc_val


def test_rand_unit_sphere():
    """Test that samples are drawn uniformly on the unit circle."""
    n_samples = 1000
    shadow.utils.set_seed(0)
    samples = shadow.rpt.rand_unit_sphere(torch.empty(n_samples, 2))
    # Test radius of 1
    radius = torch.norm(samples, dim=1)
    assert torch.allclose(radius, torch.ones(n_samples))
    # Test uniform distributed angles
    theta = torch.atan2(samples[:, 0], samples[:, 1])
    # Bin angles and do chi^2
    freq, bins = np.histogram(theta.numpy())
    stat, p_val = chisquare(freq, [n_samples / 10] * 10)
    # Everyone's favorite significance level...
    # https://xkcd.com/882/
    assert p_val > 0.05


def test_rand_unit_sphere_non_standard_shape():
    """Test the unit circle noise draw for non-vector input."""
    n_samples = 100
    shadow.utils.set_seed(0)
    samples = shadow.rpt.rand_unit_sphere(torch.empty(n_samples, 2, 3, 4))
    # Assert shape of sampled noise
    assert list(samples.shape) == [n_samples, 2, 3, 4]
    # Assert radius of 1
    radius = torch.norm(samples.view(n_samples, -1), dim=1)
    assert torch.allclose(radius, torch.ones(n_samples))


def test_l2_normalize():
    """Test normalizing vectors."""
    shadow.utils.set_seed(0)
    r = torch.rand(2, 3)
    r_norm = shadow.rpt.l2_normalize(r)
    np.testing.assert_allclose(r_norm.norm(dim=1).numpy(), [1, 1])


def test_l2_normalize_image():
    """Test normalizing 2d samples."""
    shadow.utils.set_seed(0)
    r = torch.rand(2, 3, 4)
    r_norm = shadow.rpt.l2_normalize(r)
    np.testing.assert_allclose(r_norm.view(2, -1).norm(dim=1).numpy(), [1, 1])
