import torch
import shadow.vat
import shadow.losses
import warnings


def test_semisupervised_half_moons(torch_device, simple_classification_model, ssml_half_moons_ds, train):
    """ Simple integration test for comparing fully supervised learning for half moons against VAT ssml.

    Args:
        torch_device (torch.device): Device to use
        simple_classification_model (pytest fixture function): Function to create simple model
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
        baseline, optimizer, criterion, n_epochs=1000,
        dataset=dataset, device=torch_device)
    train_acc = shadow.losses.accuracy(y_pred, dataset.tensors[-1].to(torch_device))

    # Next, compute performance via the VAT SSML technique
    # Create the model
    model = simple_classification_model().to(torch_device)
    vat = shadow.vat.Vat(model=model, xi=1e-4, eps=0.3, power_iter=1,
                         consistency_type="mse")
    # Optimizer and criterion for regular classification cost calculation
    optimizer = torch.optim.SGD(vat.parameters(), lr=0.1, momentum=0.9)
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)
    alpha = 2

    def loss(y_pred, y, x):
        return xEnt(y_pred, y) + alpha * vat.get_technique_cost(x)

    # Do training and test accuracy
    vat.train()
    y_pred = train(
        vat, optimizer, loss, n_epochs=5000,
        dataset=dataset, device=torch_device
    )
    ssml_acc = shadow.losses.accuracy(y_pred, y.to(torch_device))

    # Assert that VAT implementation improves performance in the low label setting
    ssml_acc_val = ssml_acc.item()
    bl_acc_val = train_acc.item()
    print("\n" + str(torch_device) + " ssml=" + str(ssml_acc_val) + ", baseline=" + str(bl_acc_val))
    assert ssml_acc_val > bl_acc_val


def test_vadv_perturbation(torch_device, simple_linear_2d_model, blobs_ds, train):
    """ Simple test for the generated adversarial perturbation using vadv_perturbation.

    Args:
        torch_device (torch.device): Device to use
        simple_linear_2d_model (pytest fixture function): Function to create simple model
        blobs_ds (torch.utils.data.dataset.TensorDataset): toy ssml 2-d dataset

    Returns: No return value

    """

    # Compute performance via the VAT SSML technique
    # Create the model
    model = simple_linear_2d_model().to(torch_device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)

    def criterion(y_pred, y, x):
        return xEnt(y_pred, y)

    # Do training and test accuracy
    train(model, optimizer, criterion, n_epochs=100,
          dataset=blobs_ds, device=torch_device)

    X = torch.tensor([[0., 0.], [0., 0.5]])
    X = X.to(torch_device)
    consistency_criterion = shadow.losses.softmax_mse_loss
    adv = shadow.vat.vadv_perturbation(X, model, xi=1e-4, eps=0.3, power_iter=1,
                                       consistency_criterion=consistency_criterion,
                                       flip_correction=True)

    # Both adv points should have little x. The first should have a positive y and the second
    # should have a negative y.
    assert(abs(adv[0][0]) < 0.05)
    assert(abs(adv[1][0]) < 0.05)
    assert(adv[0][1] > 0.25)
    assert(adv[1][1] < -0.25)


def test_vadv_pertubation_xi_check(torch_device, simple_linear_2d_model):
    """ Confirms that a warning is thrown when a perturbation of length == 0 is generated.
    This is mocked by setting a very low value for xi.

    Args:
        torch_device (torch.device): Device to use
        simple_linear_2d_model (pytest fixture function): Function to create simple model

    Returns: No return value

    """
    model = simple_linear_2d_model().to(torch_device)

    X = torch.tensor([[0., 0.], [0., 0.5]])
    X = X.to(torch_device)
    consistency_criterion = shadow.losses.softmax_mse_loss
    eps_value = 0.3
    with warnings.catch_warnings(record=True) as w:
        shadow.vat.vadv_perturbation(X, model, xi=1e-10, eps=eps_value, power_iter=1,
                                     consistency_criterion=consistency_criterion,
                                     flip_correction=True, xi_check=True)
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert ("generated perturbation vector has length smaller than eps/10," +
                " please check settings for xi") == str(w[-1].message)


def test_adv_perturbation(torch_device, simple_linear_2d_model, blobs_ds, train):
    """ Simple test for the generated adversarial perturbation using adv_perturbation.

    Args:
        torch_device (torch.device): Device to use
        simple_linear_2d_model (pytest fixture function): Function to create simple model
        blobs_ds (torch.utils.data.dataset.TensorDataset): toy ssml 2-d dataset

    Returns: No return value

    """

    # generate 2d classification dataset
    dataset = blobs_ds
    y = dataset.tensors[-1].to(torch_device)

    # Compute performance via the VAT SSML technique
    # Create the model
    model = simple_linear_2d_model().to(torch_device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)

    def criterion(y_pred, y, x):
        return xEnt(y_pred, y)

    # Do training and test accuracy
    train(model, optimizer, criterion, n_epochs=100,
          dataset=blobs_ds, device=torch_device)

    X = torch.tensor([[0., 0.], [0., 0.5]])
    y = torch.tensor([1, 0])
    X, y = X.to(torch_device), y.to(torch_device)
    # consistency_criterion = ssml.losses.softmax_mse_loss

    def consistency_criterion(y_pred, y):
        return xEnt(y_pred, y)
    adv = shadow.vat.adv_perturbation(X, y, model, criterion=consistency_criterion, optimizer=optimizer)

    # The first should have a positive y and the second should have a negative y.
    assert(adv[0][1] > 0.7)
    assert(adv[1][1] < -0.7)


def test_flip_multi_dim():
    X = torch.rand((3, 4, 5, 6))

    class Flat(torch.nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    model = torch.nn.Sequential(
        Flat(),
        torch.nn.Linear(4 * 5 * 6, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 2)
    )

    shadow.vat.vadv_perturbation(X, model, xi=1e-4, eps=0.3, power_iter=1,
                                 consistency_criterion=shadow.losses.softmax_mse_loss,
                                 flip_correction=True)


def test_flip_single_dim():
    X = torch.rand((3, 4))

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 10),
        torch.nn.Sigmoid(),
        torch.nn.Linear(10, 2)
    )

    shadow.vat.vadv_perturbation(X, model, xi=1e-4, eps=0.3, power_iter=1,
                                 consistency_criterion=shadow.losses.softmax_mse_loss,
                                 flip_correction=True)
