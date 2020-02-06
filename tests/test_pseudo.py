import torch
import shadow.pseudo as pseudo


# TODO threshold test when len of threshold is 1
# TODO threshold test when len of threshold is not n_classes


def test_threshold_weighting(torch_device):
    myThresholder = pseudo.Threshold([.7, .8]).to(torch_device)  # thresholds for a 2 class problem.
    predictions = torch.Tensor([[90, 10],
                                [90, 10],
                                [30, 70],
                                [10, 90],
                                [95, 96],
                                [50, 50]]).to(torch_device)
    expected = torch.Tensor([1, 1, 1, 1, 0, 0]).to(torch_device)
    assert torch.allclose(expected, myThresholder(predictions))


def test_perfectly_predicted_samples(torch_device, identity_model):
    myThresholder = pseudo.Threshold([.7, .8]).to(torch_device)  # thresholds for a 2 class problem.
    # corresponding dists.----  [1.0,0.0]  [0, 1.0]   [1.0 , 0]      [ 1.0, 0]
    predictions = torch.Tensor([[1000, 0], [0, 1000], [1000.0, 0.0], [1000.0, 0]]).to(torch_device)
    targets = torch.Tensor([0, 1, -1, -1]).long().to(torch_device)
    myPseudo = pseudo.PseudoLabel(identity_model, myThresholder).to(torch_device)
    expected = torch.Tensor(0).to(torch_device)
    assert torch.allclose(myPseudo.get_technique_cost(predictions, targets), expected)


def test_multiclass_samples(torch_device, identity_model):
    myThresholder = pseudo.Threshold([.7, .8, .99]).to(torch_device)  # thresholds for a 3 class problem.
    # corresponding dists.--[.33,.33,.33]  [~0, ~0.27, .73]  [.88, .12, ~0]
    predictions = torch.Tensor([[1, 1, 1], [1, 10, 11], [12, 10, 1],
                                [1, 1, 1], [1, 10, 11], [12, 10, 1]]).to(torch_device)
    targets = torch.Tensor([0, 1, 2, -1, -1, -1]).long().to(torch_device)
    # This should only count the CE loss to the last sample
    myPseudo = pseudo.PseudoLabel(identity_model, myThresholder).to(torch_device)
    expected = torch.Tensor([0.1269]).to(torch_device)
    assert torch.allclose(myPseudo.get_technique_cost(predictions, targets), expected, atol=1e-2)


def test_pseudo_unsupervised(torch_device, identity_model):
    myThresholder = pseudo.Threshold([.7, .8]).to(torch_device)  # thresholds for a 2 class problem.
    # corresponding dists.----  [1.0,0.0] [1.0,0]   [0 , 1.0] [.27, .73]    [.27,.73] [.5, .5]
    predictions = torch.Tensor([[90, 10], [90, 10], [30, 70], [96, 95], [95, 96], [50, 50]]).to(torch_device)
    targets = torch.Tensor([-1, -1, -1, -1, -1, -1]).long().to(torch_device)
    myPseudo = pseudo.PseudoLabel(identity_model, myThresholder).to(torch_device)
    expected = torch.Tensor([0.0783]).to(torch_device)
    assert torch.allclose(myPseudo.get_technique_cost(predictions, targets), expected, atol=1e-2)


def test_pseudo_semi_sup(torch_device, identity_model):
    myThresholder = pseudo.Threshold([.7, .8]).to(torch_device)  # thresholds for a 2 class problem.
    # corresponding dists.----  [1.0,0.0] [.95, .05][0, 1.0]  [.12,.88] [.27,.73] [.5, .5]
    predictions = torch.Tensor([[90, 10], [90, 87], [30, 70], [88, 90], [95, 96], [50, 50]]).to(torch_device)
    targets = torch.Tensor([0, -1, 1, -1, 0, -1]).long().to(torch_device)
    myPseudo = pseudo.PseudoLabel(identity_model, myThresholder).to(torch_device)
    expected = torch.Tensor([0.088]).to(torch_device)
    assert torch.allclose(myPseudo.get_technique_cost(predictions, targets), expected, atol=1e-2)


def test_pseudo_supervised(torch_device, identity_model):
    myThresholder = pseudo.Threshold([.7, .8]).to(torch_device)  # thresholds for a 2 class problem.
    # corresponding dists.----  [1.0,0.0] [1.0,0]   [0 , 1.0] [.27, .73]    [.27,.73] [.5, .5]
    predictions = torch.Tensor([[90, 10], [90, 10], [30, 70], [96, 95], [95, 96], [50, 50]]).to(torch_device)
    targets = torch.Tensor([1, 0, 1, 1, 1, 1]).long().to(torch_device)
    myPseudo = pseudo.PseudoLabel(identity_model, myThresholder, ssml_mode=False).to(torch_device)
    expected = torch.Tensor([20.328]).to(torch_device)
    assert torch.allclose(myPseudo.get_technique_cost(predictions, targets), expected, atol=1e-2)


def test_pseudo_one_sample_supervised(torch_device, identity_model):
    myThresholder = pseudo.Threshold([.7, .8]).to(torch_device)  # thresholds for a 2 class problem.
    # corresponding dists.----[.73, .26]
    predictions = torch.Tensor([[96, 95]]).to(torch_device)
    targets = torch.Tensor([0]).long().to(torch_device)
    myPseudo = pseudo.PseudoLabel(identity_model, myThresholder, ssml_mode=False).to(torch_device)
    expected = torch.Tensor([0.3133]).to(torch_device)
    assert torch.allclose(myPseudo.get_technique_cost(predictions, targets), expected, atol=1e-2)


def test_pseudo_one_sample_semi_sup(torch_device, identity_model):
    myThresholder = pseudo.Threshold([.7, .8]).to(torch_device)  # thresholds for a 2 class problem.
    # corresponding dists.----[.73, .26]
    predictions = torch.Tensor([[96, 95]]).to(torch_device)
    targets = torch.Tensor([-1]).long().to(torch_device)
    myPseudo = pseudo.PseudoLabel(identity_model, myThresholder.to(torch_device))
    # np.testing.assert_almost_equal(myPseudo(predictions, targets), 0.3133, decimal=3)
    expected = torch.Tensor([0.3133]).to(torch_device)
    assert torch.allclose(myPseudo.get_technique_cost(predictions, targets), expected, atol=1e-2)


def test_all_zero_weights_supervised(torch_device, identity_model):
    myThresholder = pseudo.Threshold([.7, .8]).to(torch_device)  # thresholds for a 2 class problem.
    # corresponding dists.----   [.62, .38]    [.27,.73] [.5, .5]
    predictions = torch.Tensor([[95.5, 95], [95, 96], [50, 50]]).to(torch_device)
    targets = torch.Tensor([1, 0, 1]).long().to(torch_device)
    myPseudo = pseudo.PseudoLabel(identity_model, myThresholder, ssml_mode=False).to(torch_device)
    expected = torch.Tensor(0).to(torch_device)
    assert torch.allclose(myPseudo.get_technique_cost(predictions, targets), expected, atol=1e-2)


def test_all_zero_weights_semi_sup(torch_device, identity_model):
    myThresholder = pseudo.Threshold([.7, .8]).to(torch_device)  # thresholds for a 2 class problem.
    # corresponding dists.----   [.62, .38]    [.27,.73] [.5, .5]
    predictions = torch.Tensor([[95.5, 95], [95, 96], [50, 50]]).to(torch_device)
    targets = torch.Tensor([-1, -1, -1]).long().to(torch_device)
    myPseudo = pseudo.PseudoLabel(identity_model, myThresholder).to(torch_device)
    expected = torch.Tensor(0).to(torch_device)
    assert torch.allclose(myPseudo.get_technique_cost(predictions, targets), expected, atol=1e-2)
