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


def test_ignore_unlabeled_wrapper_random_against_mse():
    """Test that completely labeled n dimensional tensors are not filtered and no dimension is dropped."""
    ignore_index = np.NINF
    criterion = torch.nn.MSELoss()

    # make two random tensors of the same random shape (random dimensions)
    dims = np.random.randint(5)
    shape = tuple(list(np.random.randint(100) for _ in range(dims)))
    y_hat = torch.randn(shape)
    y_true = torch.randn(shape)

    iuw_loss = shadow.utils.IgnoreUnlabeledWrapper(criterion=criterion,
                                                   ignore_index=ignore_index)
    iuw_loss_val = iuw_loss(y_hat, y_true).item()
    mse_loss_val = torch.nn.MSELoss()(y_hat, y_true).item()
    assert iuw_loss_val == mse_loss_val


def test_ignore_unlabeled_wrapper_ignores_partially_labeled_mse():
    """Test that the wrapper ignores samples where negative infinity is present."""
    ignore_index = np.NINF
    criterion = torch.nn.MSELoss()
    iuw_loss = shadow.utils.IgnoreUnlabeledWrapper(criterion=criterion,
                                                   ignore_index=ignore_index)

    # make sure that we ignore the second sample when computing loss
    y_hat = torch.FloatTensor([[1, 2], [3, 4]])
    y_true = torch.FloatTensor([[1, 2], [ignore_index, 4]])
    loss = iuw_loss(y_hat, y_true).item()
    assert loss == 0

    # make sure that we ignore the second sample when computing loss by making elements not equal
    y_hat = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    y_true = torch.FloatTensor([[1, 2, 3], [7, ignore_index, 9]])
    loss = iuw_loss(y_hat, y_true).item()
    assert loss == 0

    # make sure we ignore a sample that is entirely unlabeled
    y_hat = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    y_true = torch.FloatTensor([[1, 2, 3], [ignore_index, ignore_index, ignore_index]])
    loss = iuw_loss(y_hat, y_true).item()
    assert loss == 0


def test_ignore_unlabeled_wrapper_does_not_drop_1d_tensor():
    """Test that the wrapper ignores nothing when passed a labeled 1d tensor."""
    ignore_index = np.NINF
    criterion = torch.nn.MSELoss()
    iuw_loss = shadow.utils.IgnoreUnlabeledWrapper(criterion=criterion,
                                                   ignore_index=ignore_index)

    y_hat = torch.randn(3)
    y_true = torch.randn(3)

    iuw_loss_val = iuw_loss(y_hat, y_true).item()
    mse_loss_val = criterion(y_hat, y_true).item()
    assert iuw_loss_val == mse_loss_val


def test_ignore_unlabeled_wrapper_does_drops_1d_tensor_mse():
    """Test that the wrapper ignores unlabeled indexes when passed a labeled 1d tensor."""
    ignore_index = np.NINF
    criterion = torch.nn.MSELoss()
    iuw_loss = shadow.utils.IgnoreUnlabeledWrapper(criterion=criterion,
                                                   ignore_index=ignore_index)

    y_hat = torch.tensor([1, 2, 3, 4, 5])
    y_true = torch.tensor([1, ignore_index, 3, ignore_index, 5])
    assert iuw_loss(y_hat, y_true).item() == 0.0


def test_ignore_unlabeled_wrapper_does_drops_1d_tensor_bce():
    """Test that the wrapper ignores unlabeled indexes when passed a labeled 1d tensor."""
    ignore_index = np.NINF
    criterion = torch.nn.BCELoss()
    iuw_loss = shadow.utils.IgnoreUnlabeledWrapper(criterion=criterion,
                                                   ignore_index=ignore_index)

    y_hat = torch.randn(10).random_(2)
    y_true = torch.empty(10).random_(2)
    y_true[0] = ignore_index
    assert iuw_loss(y_hat, y_true) == criterion(y_hat[1:], y_true[1:])


def test_ignore_unlabeled_wrapper_drops_3d_tensor_1():
    """Test that the wrapper can handle 3d tensors."""
    ignore_index = np.NINF
    criterion = torch.nn.MSELoss()
    iuw_loss = shadow.utils.IgnoreUnlabeledWrapper(criterion=criterion,
                                                   ignore_index=ignore_index)

    iuw_y_hat = torch.FloatTensor([[[1, 2], [3, 4]],
                                   [[5, 6], [7, 8]],
                                   [[9, 0], [1, 2]]])

    iuw_y_true = torch.FloatTensor([[[2, 1], [0, 9]],
                                    [[8, 7], [6, ignore_index]],
                                    [[4, 3], [2, 1]]])

    # get only the labeled samples from y_hat and y_true for labeled mse loss
    labeled_indexes = [[[True, True],
                       [True, False],
                       [True, True]]]
    mse_y_hat = iuw_y_hat[labeled_indexes]
    mse_y_true = iuw_y_true[labeled_indexes]

    iuw_loss_val = iuw_loss(iuw_y_hat, iuw_y_true).item()
    mse_loss_val = torch.nn.MSELoss()(mse_y_hat, mse_y_true).item()
    assert iuw_loss_val == mse_loss_val


def test_ignore_unlabeled_wrapper_drops_3d_tensor_2():
    """Test that the wrapper can handle 3d tensors."""
    ignore_index = np.NINF
    criterion = torch.nn.MSELoss()
    iuw_loss = shadow.utils.IgnoreUnlabeledWrapper(criterion=criterion,
                                                   ignore_index=ignore_index)

    iuw_y_hat = torch.FloatTensor([[[1, 2], [3, 4]],
                                   [[5, 6], [7, 8]],
                                   [[9, 0], [1, 2]]])

    iuw_y_true = torch.FloatTensor([[[2, 1], [0, 9]],
                                    [[ignore_index, 7], [6, ignore_index]],
                                    [[4, 3], [2, 1]]])

    # get only the labeled samples from y_hat and y_true for labeled mse loss
    labeled_indexes = [[[True, True],
                       [False, False],
                       [True, True]]]
    mse_y_hat = iuw_y_hat[labeled_indexes]
    mse_y_true = iuw_y_true[labeled_indexes]

    iuw_loss_val = iuw_loss(iuw_y_hat, iuw_y_true).item()
    mse_loss_val = torch.nn.MSELoss()(mse_y_hat, mse_y_true).item()
    assert iuw_loss_val == mse_loss_val


def test_ignore_unlabeled_wrapper_arg_commutative():
    """Test that the wrapper passes kwargs to the criterion in the order that they were received."""
    ignore_index = np.NINF
    criterion = torch.nn.MSELoss()
    iuw_loss = shadow.utils.IgnoreUnlabeledWrapper(criterion=criterion,
                                                   ignore_index=ignore_index)

    y_hat = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
    y_true = torch.FloatTensor([[6, 5], [4, 3], [2, 1]])
    assert iuw_loss(y_hat, y_true) == iuw_loss(y_true, y_hat) == torch.nn.MSELoss()(y_hat, y_true)

    y_hat = torch.FloatTensor([[1, 2], [3, 4], [5, 6]])
    y_true = torch.FloatTensor([[6, 5], [ignore_index, 3], [2, 1]])
    assert iuw_loss(y_hat, y_true) == iuw_loss(y_true, y_hat)


def test_ignore_unlabeled_wrapper_bce_loss():
    """Test the wrapper using ignore_index=2 and BCE as the criterion."""
    ignore_index = 2
    criterion = torch.nn.BCELoss()
    iuw_loss = shadow.utils.IgnoreUnlabeledWrapper(criterion=criterion,
                                                   ignore_index=ignore_index)

    y_hat = torch.FloatTensor([[1], [0], [1], [0], [0], [1], [1], [1], [1]])
    y_true = torch.FloatTensor([[0], [ignore_index], [1], [0], [ignore_index], [1], [0], [ignore_index], [1]])

    labeled_indexes = [[True, False, True, True, False, True, True, False, True]]
    bce_y_hat = y_hat[labeled_indexes]
    bce_y_true = y_true[labeled_indexes]

    assert iuw_loss(y_hat, y_true) == torch.nn.BCELoss()(bce_y_hat, bce_y_true)
