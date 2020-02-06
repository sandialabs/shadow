import torch
import shadow.losses
import shadow.ema
import shadow.rpt
import shadow.utils
import shadow.module_wrapper
import warnings


def adv_perturbation(x, y, model, criterion, optimizer):
    """Find adversarial perturbation following [Goodfellow14]_.

    Args:
        x (torch.Tensor): Input data.
        y (torch.Tensor): Input labels.
        model (torch.nn.Module): The model.
        criterion (callable): The loss criterion used to measure performance.
        optimizer (torch.optim.Optimizer): Optimizer used to compute gradients.

    Returns:
        torch.Tensor: Adversarial perturbations.
    """
    x.requires_grad = True
    out = model(x)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    r_adv = shadow.rpt.l2_normalize(torch.sign(x.grad))
    x.requires_grad = False
    return r_adv


def vadv_perturbation(x, model, xi, eps, power_iter, consistency_criterion,
                      flip_correction=True, xi_check=False):
    r"""Find virtual adversarial perturbation following [Miyato18]_.

    Args:
        x (torch.Tensor): Input data.
        model (torch.nn.Module): The model.
        xi (float): Scaling value for the random direction vector.
        eps (float): The magnitude of applied adversarial perturbation. Greater `eps`
            implies more smoothing.
        power_iter (int): Number of power iterations used in estimation.
        consistency_criterion (callable): Cost function used to measure consistency.
        flip_correction (bool, optional): Correct flipped virtual adversarial perturbations
            induced by power iteration estimation. These iterations sometimes converge to a
            "flipped" perturbation (away from maximum change in consistency). This correction
            detects this behavior and corrects flipped perturbations at the cost of slightly
            increased compute. This behavior is not included in the original VAT implementation,
            which exhibits perturbation flipping without any corrections. Defaults to `True`.
        xi_check (bool, optional): Raise warnings for small perturbations lengths. The parameter
            `xi` should be selected so as to be small (for algorithm assumptions to be correct),
            but not so small as to collapse the perturbation into a length 0 vector. This parameter
            controls optional warnings to detect a value of `xi` that causes perturbations to
            collapse to length 0. Defaults to `False`.

    Returns:
        torch.Tensor: Virtual adversarial perturbations.
    """
    # The minibatch size is required in order to find the mean loss
    minibatch_size = x.shape[0]

    # find regular probabilities
    with torch.no_grad():  # turn gradient off of the regular model
        out = model(x)

    # create random unit tensor
    d = shadow.rpt.rand_unit_sphere(x)
    # calculate adversarial direction
    for i in range(power_iter):

        d.requires_grad = True
        # find probability with the random tensor
        out_plus = model(x + xi * d)

        # compute the distance
        dist = consistency_criterion(out, out_plus) / minibatch_size
        dist.backward()  # gradient w.r.t. r if distance isn't 0

        d = shadow.rpt.l2_normalize(d.grad)
        model.zero_grad()

    r_adv = eps * shadow.rpt.l2_normalize(d.view(x.shape[0], -1)).view(x.shape)
    if xi_check:
        bools = torch.norm(r_adv.view(r_adv.shape[0], -1)) < eps / 10
        if bools.any():
            warnings.warn("generated perturbation vector has length smaller than eps/10," +
                          " please check settings for xi", RuntimeWarning)
    if flip_correction:
        with torch.no_grad():
            # Current per-sample distance
            currentD = consistency_criterion(
                out, model(x + r_adv), reduction="none").sum(dim=-1)
            # Flipped per-sample distance
            flippedD = consistency_criterion(
                out, model(x - r_adv), reduction="none").sum(dim=-1)
        # Flip if flipped distance is better than current
        flip = (currentD > flippedD).float() * 2 - 1
        # Flip if needed, padding out with sufficient singleton dims
        r_adv = r_adv * flip.view(-1, *([1] * (x.dim() - 1)))

    return r_adv


class Vat(shadow.module_wrapper.ModuleWrapper):
    r"""Virtual Adversarial Training (VAT, [Miyato18]_) model wrapper for consistency regularization.

    Args:
       model (torch.nn.Module): The model to train and regularize.
       xi (float, optional): Scaling value for the random direction vector. Defaults to 1.0.
       eps (float, optional): The magnitude of applied adversarial perturbation. Greater `eps`
           implies more smoothing. Defaults to 1.0.
       power_iter (int, optional): Number of power iterations used to estimate virtual
           adversarial direction. Per [Miyato18]_, defaults to 1.
       consistency_type ({'kl', 'mse'}, optional): Cost function used to measure consistency.
           Defaults to `'kl'` (KL-divergence).
       flip_correction (bool, optional): Correct flipped virtual adversarial perturbations
           induced by power iteration estimation. These iterations sometimes converge to a
           "flipped" perturbation (away from maximum change in consistency). This correction
           detects this behavior and corrects flipped perturbations at the cost of slightly
           increased compute. This behavior is not included in the original VAT implementation,
           which exhibits perturbation flipping without any corrections. Defaults to `True`.
       xi_check (bool, optional): Raise warnings for small perturbations lengths. It should
           be selected so as to be small (for algorithm assumptions to be correct), but not so
           small as to collapse the perturbation into a length 0 vector. This parameter controls
           optional warnings to detect a value of `xi` that causes perturbations to collapse
           to length 0. Defaults to `False`.
    """
    def __init__(self, model, xi=1.0, eps=1.0, power_iter=1, consistency_type="kl",
                 flip_correction=True, xi_check=False):
        super(Vat, self).__init__(model)
        self.xi = xi
        self.xi_check = xi_check
        self.eps = eps
        self.power_iter = power_iter
        self.flip_correction = flip_correction
        if self.power_iter <= 0:
            self.power_iter = 1

        if consistency_type == 'mse':
            self.consistency_criterion = shadow.losses.softmax_mse_loss
        elif consistency_type == 'kl':
            self.consistency_criterion = shadow.losses.softmax_kl_loss
        else:
            raise ValueError(
                "Unknown consistency type. Should be 'mse' or 'kl', but is " + str(consistency_type)
            )

    def get_technique_cost(self, x):
        r"""VAT consistency cost (local distributional smoothness).

        Args:
            x (torch.Tensor): Tensor of the data

        Returns:
            torch.Tensor: Consistency cost between the data and virtual adversarially
            perturbed data.
        """
        with torch.no_grad():
            out = self.model(x)

        r = vadv_perturbation(x, self.model, self.xi, self.eps,
                              self.power_iter, self.consistency_criterion,
                              flip_correction=self.flip_correction, xi_check=self.xi_check)
        out_plus = self.model(x + r)

        # The minibatch size is required in order to find the mean loss
        minibatch_size = x.shape[0]

        loss = self.consistency_criterion(out, out_plus) / minibatch_size

        return loss
