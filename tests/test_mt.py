import pytest
import torch
import shadow.mt
import shadow.utils
import shadow.losses


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
        baseline, optimizer, criterion, n_epochs=1000,
        dataset=dataset, device=torch_device
    )
    train_acc = shadow.losses.accuracy(y_pred, y)

    # Next, compute performance via the Mean-Teacher SSML technique
    # Create the model
    model = simple_classification_model().to(torch_device)

    # Create the MT model which includes student and teacher, along with necessary parameters.
    mt = shadow.mt.MeanTeacher(model=model, alpha=0.8, noise=0.1, consistency_type="kl")

    # Optimizer and criterion for regular classification cost calculation
    optimizer = torch.optim.SGD(mt.parameters(), lr=0.1, momentum=0.9)
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)

    mt_consistency_weight = 4.0

    def criterion(y_pred, y, x):
        return xEnt(y_pred, y) + mt_consistency_weight * mt.get_technique_cost(x)

    # Do training and test accuracy
    mt.train()
    y_pred = train(
        mt, optimizer, criterion, n_epochs=5000,
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
        shadow.mt.MeanTeacher(model, teacher_model, consistency_type='hello')


def test_mt_eval_mode(torch_device, simple_classification_model, ssml_half_moons_ds):
    dataset = ssml_half_moons_ds
    X = dataset.tensors[0].to(torch_device)
    y = dataset.tensors[1].to(torch_device)

    # Create the model
    model = simple_classification_model().to(torch_device)

    mt = shadow.mt.MeanTeacher(model=model, alpha=0.8, noise=0.1,
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
