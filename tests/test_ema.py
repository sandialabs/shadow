import numpy as np
import torch
import shadow.rpt
import shadow.utils
import shadow.ema


def test_ema_update_075_step0(torch_device, simple_value_initialized_model):
    """Simple test for EMA calculation with alpha=0.75, step0."""
    ema_alpha = 0.75
    step = 0

    # Create simple student and teacher model
    # Initialize student with 1s and teacher with 0s
    model = simple_value_initialized_model(value=1).to(torch_device)
    ema_model = simple_value_initialized_model().to(torch_device)
    shadow.ema.update_model(model, ema_model, ema_alpha, step)

    # Verify all good
    expected = np.array([[1., 1.], [1., 1.]])
    actual = list(ema_model.parameters())[0].data.cpu().numpy()
    assert np.array_equal(actual, expected)


def test_ema_update_075_step1(torch_device, simple_value_initialized_model):
    """Simple test for EMA calculation with alpha=0.75, step1."""
    ema_alpha = 0.75
    step = 1

    # Create simple student and teacher model
    # Initialize student with 2s and teacher with 1s
    model = simple_value_initialized_model(value=2).to(torch_device)
    ema_model = simple_value_initialized_model(value=1).to(torch_device)
    shadow.ema.update_model(model, ema_model, ema_alpha, step)

    # Verify all good
    expected = np.array([[1.5, 1.5], [1.5, 1.5]])
    actual = list(ema_model.parameters())[0].data.cpu().numpy()
    assert np.array_equal(actual, expected)


def test_ema_update_075_step2(torch_device, simple_value_initialized_model):
    """Simple test for EMA calculation with alpha=0.75, step2."""
    ema_alpha = 0.75
    step = 2

    # Create simple student and teacher model
    # Initialize student with 3s and teacher with 1.5s
    model = simple_value_initialized_model(value=3).to(torch_device)
    ema_model = simple_value_initialized_model(value=1.5).to(torch_device)
    shadow.ema.update_model(model, ema_model, ema_alpha, step)

    # Verify all good
    expected = np.array([[2.0, 2.0], [2.0, 2.0]])
    actual = list(ema_model.parameters())[0].data.cpu().numpy()
    assert np.array_equal(actual, expected)


def test_ema_update_050_step0(torch_device, simple_value_initialized_model):
    """Simple test for EMA calculation with alpha=0.50, step0."""
    # Set alpha value
    ema_alpha = 0.5
    step = 0

    # Create simple student and teacher model
    # Initialize student with 1s and teacher with 0s
    model = simple_value_initialized_model(value=1).to(torch_device)
    ema_model = simple_value_initialized_model().to(torch_device)
    shadow.ema.update_model(model, ema_model, ema_alpha, step)

    # Verify all good
    expected = np.array([[1., 1.], [1., 1.]])
    actual = list(ema_model.parameters())[0].data.cpu().numpy()
    assert np.array_equal(actual, expected)


def test_ema_update_050_step1(torch_device, simple_value_initialized_model):
    """Simple test for EMA calculation with alpha=0.50, step1."""
    # Set alpha value
    ema_alpha = 0.5
    step = 1

    # Create simple student and teacher model
    # Initialize student with 2s and teacher with 1s
    model = simple_value_initialized_model(value=2).to(torch_device)
    ema_model = simple_value_initialized_model(value=1).to(torch_device)
    shadow.ema.update_model(model, ema_model, ema_alpha, step)

    # Verify all good
    expected = np.array([[1.5, 1.5], [1.5, 1.5]])
    actual = list(ema_model.parameters())[0].data.cpu().numpy()
    assert np.array_equal(actual, expected)


def test_ema_update_050_step2(torch_device, simple_value_initialized_model):
    """Simple test for EMA calculation with alpha=0.50, step2."""
    # Set alpha value
    ema_alpha = 0.5
    step = 2

    # Create simple student and teacher model
    # Initialize student with 3s and teacher with 1.5s
    model = simple_value_initialized_model(value=3).to(torch_device)
    ema_model = simple_value_initialized_model(value=1.5).to(torch_device)
    shadow.ema.update_model(model, ema_model, ema_alpha, step)

    # Verify all good
    expected = np.array([[2.25, 2.25], [2.25, 2.25]])
    actual = list(ema_model.parameters())[0].data.cpu().numpy()
    assert np.array_equal(actual, expected)


def test_ema_update_conv2d():
    """Simple test of EMA calculation with conv2d layer along with maxpool, relu, and batchnorm."""
    # Set alpha value
    ema_alpha = 0.5
    step = 3

    model = torch.nn.Sequential()
    model.add_module("Conv2d", torch.nn.Conv2d(1, 2, 2))
    model.add_module("MaxPool2d", torch.nn.MaxPool2d(1, 1))
    model.add_module("ReLU", torch.nn.ReLU(True))
    model.add_module("BatchNorm2d", torch.nn.BatchNorm2d(1))
    shadow.utils.init_model_weights(model, 1)
    ema_model = torch.nn.Sequential()
    ema_model.add_module("emaConv2d", torch.nn.Conv2d(1, 2, 2))
    ema_model.add_module("emaMaxPool2d", torch.nn.MaxPool2d(1, 1))
    ema_model.add_module("emaReLU", torch.nn.ReLU(True))
    ema_model.add_module("emaBatchNorm2d", torch.nn.BatchNorm2d(1))
    shadow.utils.init_model_weights(ema_model, 3)
    shadow.ema.update_model(model, ema_model, ema_alpha, step)

    # Verify all good with Conv2d
    expected = np.array([[[[2.0, 2.0], [2.0, 2.0]]], [[[2.0, 2.0], [2.0, 2.0]]]])
    # The parameters returned are conv2d.weights, conv2d.bias, batchnorm2d.weights, batchnorm2d.bias
    actual = list(ema_model.parameters())[0].data.cpu().numpy()  # conv2d.weights
    assert np.array_equal(actual, expected)

    # Verify all good with BatchNorm2d
    expected = np.array([2.0])
    # The parameters returned are conv2d.weights, conv2d.bias, batchnorm2d.weights, batchnorm2d.bias
    actual = list(ema_model.parameters())[2].data.cpu().numpy()  # batchnorm2d.weights
    assert np.array_equal(actual, expected)
