import torch
import shadow.eaat
import shadow.losses


def test_semisupervised_half_moons(torch_device, simple_classification_model, ssml_half_moons_ds, train):
    """ Simple integration test for comparing fully supervised learning for half moons against EAAT ssml.

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

    # Next, compute performance via the EAAT SSML technique
    # Create the model
    model = simple_classification_model().to(torch_device)
    # Create the EAAT model which includes student and teacher, along with necessary parameters
    # for both teacher and VAT-applied-to-student.
    eaat = shadow.eaat.Eaat(model=model, alpha=0.8, student_noise=0.1,
                            teacher_noise=0.1, xi=1e-4, eps=0.3, power_iter=1,
                            consistency_type="mse")
    # Optimizer and criterion for regular classification cost calculation
    optimizer = torch.optim.SGD(eaat.parameters(), lr=0.02, momentum=0.9)
    xEnt = torch.nn.CrossEntropyLoss(ignore_index=-1).to(torch_device)
    alpha = 4

    def loss(y_pred, y, x):
        return xEnt(y_pred, y) + alpha * eaat.get_technique_cost(x)

    # Do training and test accuracy
    y_pred = train(
        eaat, optimizer, loss, n_epochs=5000,
        dataset=dataset, device=torch_device
    )
    ssml_acc = shadow.losses.accuracy(y_pred, y.to(torch_device))

    # Assert that EAAT implementation improves performance in the low label setting
    ssml_acc_val = ssml_acc.item()
    bl_acc_val = train_acc.item()
    print("\n" + str(torch_device) + " ssml=" + str(ssml_acc_val) + ", baseline=" + str(bl_acc_val))
    assert ssml_acc_val > bl_acc_val
