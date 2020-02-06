import random
import numpy as np
import torch
import shadow.utils


def test_set_seed():
    """Test seeding."""
    shadow.utils.set_seed(0)
    r_a = random.uniform(0, 1)
    np_a = np.random.uniform()
    t_a = torch.randn([1]).item()

    shadow.utils.set_seed(0)
    r_b = random.uniform(0, 1)
    np_b = np.random.uniform()
    t_b = torch.randn([1]).item()

    assert r_a == r_b
    assert np_a == np_b
    assert t_a == t_b


def test_init_model_weights_linear():
    """Test initialize model weights with linear layer."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    shadow.utils.init_model_weights(model, 1)
    # Verify all good
    expected = np.array([[1.0, 1.0], [1.0, 1.0]])
    actual = list(model.parameters())[0].data.cpu().numpy()
    assert np.array_equal(actual, expected)


def test_init_model_weights_conv1d():
    """Test initialize model weights with conv1d layer."""

    model = torch.nn.Sequential(torch.nn.Conv1d(2, 1, 2))
    shadow.utils.init_model_weights(model, -1)
    # Verify all good
    expected = np.array([[[-1.0, -1.0], [-1.0, -1.0]]])
    actual = list(model.parameters())[0].data.cpu().numpy()
    assert np.array_equal(actual, expected)


def test_init_model_weights_conv2d():
    """Test initialize model weights with conv2d and batchnorm2d layers and some that weights aren't applied."""

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, 2, 1, 1),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.ReLU(True),
        torch.nn.BatchNorm2d(1))
    shadow.utils.init_model_weights(model, 2)

    # Verify all good with Conv2d
    expected = np.array([[[[2.0, 2.0], [2.0, 2.0]]], [[[2.0, 2.0], [2.0, 2.0]]]])
    # The parameters returned are conv2d.weights, conv2d.bias, batchnorm2d.weights, batchnorm2d.bias
    actual = list(model.parameters())[0].data.cpu().numpy()  # conv2d.weights
    assert np.array_equal(actual, expected)

    # Verify all good with BatchNorm2d
    expected = np.array([2.0])
    # The parameters returned are conv2d.weights, conv2d.bias, batchnorm2d.weights, batchnorm2d.bias
    actual = list(model.parameters())[2].data.cpu().numpy()  # batchnorm2d.weights
    assert np.array_equal(actual, expected)


def test_constant_consistency_weight():
    """Test the constant consistency ramp."""

    alpha = shadow.utils.ConstantCW(last_weight=1)

    weights = []
    for epoch in range(3):
        weights.append(alpha())
        alpha.step()

    assert weights == [1, 1, 1]


def test_sigmoid_consistency_weight():
    """Test the sigmoid consistency ramp."""

    alpha = shadow.utils.SigmoidCW(last_epoch=3, last_weight=2, first_weight=1, epochs_before=1)

    weights = []
    for epoch in range(5):
        weights.append(alpha())
        alpha.step()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    expected = [
        1, 1 + sigmoid(-10), 1 + sigmoid(0), 1 + sigmoid(10), 1 + sigmoid(10)
    ]

    assert weights == expected


def test_ramp_consistency_weight():
    """Test the linear consistency ramp."""
    alpha = shadow.utils.RampCW(last_epoch=3, last_weight=2, first_weight=1, epochs_before=1)

    weights = []
    for epoch in range(5):
        weights.append(alpha())
        alpha.step()

    assert weights == [1, 1, 1.5, 2, 2]


def test_step_consistency_weight():
    """Test step consistency ramp."""
    alpha = shadow.utils.StepCW(last_epoch=3, last_weight=1, first_weight=0)

    weights = []
    for epoch in range(5):
        weights.append(alpha())
        alpha.step()

    assert weights == [0, 0, 0, 1, 1]


def test_quadratic_consistency_weight():
    """Test quadratic consistency ramp."""
    alpha = shadow.utils.QuadraticCW(last_epoch=3, last_weight=2, first_weight=1, epochs_before=1)

    weights = []
    for epoch in range(5):
        weights.append(alpha())
        alpha.step()

    assert weights == [1, 1, 1.75, 2, 2]


def test_skewed_sigmoid_consistency_weight():
    """Test the sigmoid consistency ramp."""

    alpha = shadow.utils.SkewedSigmoidCW(last_epoch=5, last_weight=1, first_weight=0, epochs_before=0, beta=1, zeta=1.8)

    weights = []
    for epoch in range(5):
        weights.append(alpha())
        alpha.step()

    expected = np.asarray([0.0000, 0.0002, 0.0140, 0.8717, 1.0000])

    np.testing.assert_allclose(weights, expected, atol=1e-04)
