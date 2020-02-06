import torch
import shadow.losses
import shadow.module_wrapper


def l2_normalize(r):
    r"""L2 normalize tensor, flattening over all but batch dim (0).

    Args:
        r (torch.Tensor): Tensor to normalize.

    Returns:
        torch.Tensor: Normalized tensor (over all but dim 0).
    """
    # Compute length over all but sample dim
    norm = torch.norm(r.view(r.shape[0], -1), dim=1, keepdim=False)
    # Divide by the length, insert single dims into the length vector for
    # array broadcasting. Add a small perturbation for stability.
    return r / (norm.view(*norm.shape, *(1 for _ in r.shape[1:])) + torch.finfo(r.dtype).tiny)


def rand_unit_sphere(x):
    r"""Draw samples from the uniform distribution over the unit sphere.

    Args:
        x (torch.Tensor): Tensor used to define shape and dtype for the
            generated tensor.

    Returns:
        torch.Tensor: Random unit sphere samples.

    Reference:
        https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe
    """  # noqa: E501
    return l2_normalize(torch.randn_like(x))


class RPT(shadow.module_wrapper.ModuleWrapper):
    r"""Random Perturbation Training for consistency regularization.

    Random Perturbation Training (RPT) is a special case of Virtual Adversarial
    Training (VAT, [Miyato18]_) for which the number of power iterations is 0. This means
    that added perturbations are isotropically random (not in the adversarial
    direction).

    Args:
        eps (float): The magnitude of applied perturbation. Greater `eps`
            implies more smoothing.
        model (torch.nn.Module): The model to train and regularize.
        consistency_type ({'kl', 'mse'}, optional): Cost function used to measure consistency.
            Defaults to `'kl'` (KL-divergence).
    """
    def __init__(self, eps, model, consistency_type="mse"):
        super(RPT, self).__init__(model)
        self.eps = eps

        if consistency_type == 'mse':
            self.consistency_criterion = shadow.losses.softmax_mse_loss
        elif consistency_type == 'kl':
            self.consistency_criterion = shadow.losses.softmax_kl_loss
        else:
            raise ValueError(
                "Unknown consistency type. Should be 'mse' or 'kl', but is " + str(consistency_type)
            )

    def get_technique_cost(self, x):
        r"""Consistency cost (local distributional smoothness).

        Args:
            x (torch.Tensor): Tensor of the data

        Returns:
            torch.Tensor: Consistency cost between the data and randomly perturbed data.
        """
        with torch.no_grad():
            model_logits = self.model(x)

        r = self.eps * rand_unit_sphere(x)
        model_logits_r = self.model(x + r)

        # The minibatch size is required in order to find the mean loss
        minibatch_size = x.shape[0]

        loss = self.consistency_criterion(model_logits, model_logits_r) / minibatch_size

        return loss
