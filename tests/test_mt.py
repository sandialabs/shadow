import numpy as np
import pytest
import torch
import shadow.mt
import shadow.utils
import shadow.losses


def test_ema_invalid_smoothing(torch_device, simple_value_initialized_model):
    """Raise value error for invalid smooth coefficient."""
    ema_alpha = 5
    step = 0

    # Create simple student and teacher model
    # Initialize student with 1s and teacher with 0s
    model = simple_value_initialized_model(value=1).to(torch_device)
    ema_model = simple_value_initialized_model().to(torch_device)
    with pytest.raises(ValueError):
        shadow.mt.ema_update_model(model, ema_model, ema_alpha, step)


def test_ema_update_075_step0(torch_device, simple_value_initialized_model):
    """Simple test for EMA calculation with alpha=0.75, step0."""
    ema_alpha = 0.75
    step = 0

    # Create simple student and teacher model
    # Initialize student with 1s and teacher with 0s
    model = simple_value_initialized_model(value=1).to(torch_device)
    ema_model = simple_value_initialized_model().to(torch_device)
    shadow.mt.ema_update_model(model, ema_model, ema_alpha, step)

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
    shadow.mt.ema_update_model(model, ema_model, ema_alpha, step)

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
    shadow.mt.ema_update_model(model, ema_model, ema_alpha, step)

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
    shadow.mt.ema_update_model(model, ema_model, ema_alpha, step)

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
    shadow.mt.ema_update_model(model, ema_model, ema_alpha, step)

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
    shadow.mt.ema_update_model(model, ema_model, ema_alpha, step)

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
    shadow.mt.ema_update_model(model, ema_model, ema_alpha, step)

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


def test_semisupervised_half_moons(torch_device, simple_classification_model, ssml_half_moons_ds, train):
    """ Simple integration test for comparing fully supervised learning for half moons against Mean-Teacher ssml.

    Args:
        torch_device (torch.device): Device to use
        simple_classification_model (pytest.fixture function): Function to create simple model
        ssml_half_moons_ds (torch.utils.data.dataset.TensorDataset): toy ssml 2-d dataset

    Returns: No return value

    """

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
        baseline, optimizer, criterion, n_epochs=500,
        dataset=dataset, device=torch_device
    )
    train_acc = shadow.losses.accuracy(y_pred, y)

    # Next, compute performance via the Mean-Teacher SSML technique
    # Create the model
    model = simple_classification_model().to(torch_device)

    # Create the MT model which includes student and teacher, along with necessary parameters.
    mt = shadow.mt.MT(model=model, alpha=0.8, noise=0.1, consistency_type="kl")

    # Optimizer and criterion for regular classification cost calculation
    optimizer = torch.optim.SGD(mt.parameters(), lr=0.1, momentum=0.9)
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)

    def criterion(y_pred, y, x):
        return xEnt(y_pred, y) + mt.get_technique_cost(x)

    # Do training and test accuracy
    mt.train()
    y_pred = train(
        mt, optimizer, criterion, n_epochs=500,
        dataset=dataset, device=torch_device
    )
    ssml_acc = shadow.losses.accuracy(y_pred, y)

    # Assert that MT implementation improves performance in the low label setting
    ssml_acc_val = ssml_acc.item()
    bl_acc_val = train_acc.item()
    print("\n" + str(torch_device) + " ssml=" + str(ssml_acc_val) + ", baseline=" + str(bl_acc_val))

    assert ssml_acc_val > bl_acc_val


def test_mt_consistency_value_error(torch_device, simple_classification_model):
    """ Test consistency type besides mse and kl raises ValueError.

    Args:
        torch_device (torch.device): Device to use
        simple_classification_model (pytest.fixture function): Function to create simple model

    Returns: No return value

    """

    with pytest.raises(ValueError):
        model = simple_classification_model().to(torch_device)
        teacher_model = simple_classification_model().to(torch_device)
        shadow.mt.MT(model, teacher_model, consistency_type='hello')


def test_mt_eval_mode(torch_device, simple_classification_model, ssml_half_moons_ds):
    dataset = ssml_half_moons_ds
    X = dataset.tensors[0].to(torch_device)
    y = dataset.tensors[1].to(torch_device)

    # Create the model
    model = simple_classification_model().to(torch_device)

    mt = shadow.mt.MT(model=model, alpha=0.8, noise=0.1,
                      consistency_type="kl")

    # Optimizer and criterion for regular classification cost calculation
    optimizer = torch.optim.SGD(mt.parameters(), lr=0.1, momentum=0.9)

    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)

    # Get the starting parameter values for ema model
    init_params = [p.clone() for p in mt.teacher_model.parameters()]

    mt.train()
    for epoch in range(2):
        out = mt(X)
        loss = xEnt(out, y) + mt.get_technique_cost(X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Get the trained parameter values for ema model
    trained_params = [p.clone() for p in mt.teacher_model.parameters()]

    # Assert EMA parameters are not the same after training
    for ip, tp in zip(init_params, trained_params):
        assert not torch.equal(ip, tp)

    mt.eval()
    for epoch in range(2):
        out = mt(X)
        loss = xEnt(out, y) + mt.get_technique_cost(X)

    # Get the validation parameter values for ema model
    val_params = [p.clone() for p in mt.teacher_model.parameters()]

    # Assert the EMA parameters are the same after validation
    for vp, tp in zip(val_params, trained_params):
        assert torch.equal(vp, tp)
