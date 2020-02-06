import numpy as np
import torch
import shadow.losses


def test_softmax_mse_loss():
    """Simple test for softmax mse loss."""

    input_logits = torch.tensor([[0.5, 1.]])
    target_logits = torch.tensor([[1., 1.]])

    # softmax of input_logits is 0.3775, 0.6225, and target_logits is 0.5, 0.5
    # therefore would expect the MSE loss to be 0.015 because returns the sum over all instead of the mean.

    losses = shadow.losses.softmax_mse_loss(input_logits, target_logits)
    assert np.allclose(losses.data.cpu().numpy(), np.array(0.015), rtol=1e-3)


def test_softmax_kl_loss():
    """Simple test for softmax kl loss."""

    input_logits = torch.tensor([[0.5, 1.]])
    target_logits = torch.tensor([[1., 1.]])

    # softmax of input_logits is 0.3775, 0.6225, and target_logits is 0.5, 0.5
    # TODO: taking its word that the kl loss is 0.0309. Determine independently?

    losses = shadow.losses.softmax_kl_loss(input_logits, target_logits)
    assert np.allclose(losses.data.cpu().numpy(), np.array(0.0309), rtol=1e-3)
