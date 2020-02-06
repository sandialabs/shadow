import random
import torch
import torch.utils.data
import numpy as np


def flatten_to_two_dim(input_tensor):
    r"""Flatten tensor along the first axis ([2, 3, 4] -> [2, 12])

    Args:
        input_tensor (torch.Tensor): input tensor

    Returns:
        torch.Tensor: `input_tensor` flattened along first axis
    """
    return input_tensor.view(input_tensor.shape[0], -1)


def set_seed(seed, cudnn_deterministic=False):
    r"""Sets the seeds for max reproducibility.

    Sets seeds for random, numpy, and torch to `seed`, and can also
    enable deterministic mode for the CuDNN backend. This does not guarantee
    full reproducibility as some underlying options (e.g. `atomicAdd`) still
    have sources of non-determinism that cannot be disabled.

    Args:
        seed (int): Seed used for `random`, `numpy`, and `torch`.
        cudnn_deterministic (bool, optional): Sets the CuDNN backend into
            deterministic mode. This can negatively impact performance. Defaults to False.

    .. note::
        PyTorch provides only minimal guarantees on reproducibility. See
        <https://pytorch.org/docs/stable/notes/randomness.html> for more
        information.
    """
    # Python seeding
    random.seed(seed)
    # Numpy seeding
    np.random.seed(seed)
    # Torch seeding
    torch.manual_seed(seed)
    # CuDNN deterministic seeding-can impact performance
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class _CWScheduler(object):
    """Base class for consistency weight schedulers.

    The base class for consistency weight schedulers. This base should not
    be instantiated directly. Consistency weight schedulers should implement
    `_make_ramp`.
    """
    def __init__(self):
        super(_CWScheduler, self).__init__()
        self._make_ramp()
        self.it = 0

    def __call__(self):
        r"""The current consistency weight.

        Returns:
            float: The current scheduled consistency weight.
        """
        try:
            return self.ramp[self.it]
        except IndexError:
            return self.ramp[-1]

    def step(self, increment=1):
        r"""Update the scheduler to the next step.

        Args:
            it (int): Number of steps to take. Defaults to 1.
        """
        self.it += increment


class ConstantCW(_CWScheduler):
    r"""Constant valued consistency weight scheduler.

    Scheduler function to control a weight, often used to weigh a consistency cost relative to
    a supervised learning cost (e.g. Cross Entropy). This is intended to be stepped after each
    epoch during training to increase or decrease the weight accordingly. This provides a
    constant weighting function that does not change.

    Args:
        last_weight (float, optional): Final consistency weight. Defaults to 1.

    Example:
        >>> alpha = ConstantCW(last_weight)
        >>> for epoch in epochs:
        >>>     train(...)
        >>>     loss = criterion + alpha() * consistency
        >>>     validate(...)
        >>>     alpha.step()
    """
    def __init__(self, last_weight=1):
        self.last_weight = last_weight
        super(ConstantCW, self).__init__()

    def _make_ramp(self):
        self.ramp = [self.last_weight]


class SigmoidCW(_CWScheduler):
    r"""Sigmoidal consistency weight scheduler.

    Scheduler function to control a weight, often used to weigh a consistency cost relative to
    a supervised learning cost (e.g. Cross Entropy). This is intended to be stepped after each
    epoch during training to increase or decrease the weight accordingly. This provides a
    sigmoidal weighting function.

    Args:
        last_epoch (int): Number of epochs until scheduler reaches `last_weight`.
        last_weight (float, optional): Final consistency weight. Defaults to 1.
        first_weight (float, optional): Consistency weight at beginning of ramp.
            Defaults to 0.
        epochs_before (int, optional): Number of epochs to hold weight at
            `first_weight` before beginning ramp. Defaults to 0.

    Example:
        >>> alpha = SigmoidCW(last_epoch, last_weight, first_weight, epochs_before)
        >>> for epoch in epochs:
        >>>     train(...)
        >>>     loss = criterion + alpha() * consistency
        >>>     validate(...)
        >>>     alpha.step()
    """
    def __init__(self, last_epoch, last_weight=1, first_weight=0, epochs_before=0):
        self.last_epoch = last_epoch
        self.last_weight = last_weight
        self.first_weight = first_weight
        self.epochs_before = epochs_before
        super(SigmoidCW, self).__init__()

    def _make_ramp(self):
        # Calculate a range of sigmoid values
        sigmoid_steps = 1 / (1 + np.exp(-np.linspace(-10, 10, self.last_epoch)))
        # Scale the sigmoid from first_weight to last_weight
        sigmoid_steps = sigmoid_steps * (self.last_weight - self.first_weight) + self.first_weight
        # Keep weight fixed initially
        self.ramp = [self.first_weight] * self.epochs_before + sigmoid_steps.tolist()


class RampCW(_CWScheduler):
    r"""Linear ramp consistency weight scheduler.

    Scheduler function to control a weight, often used to weigh a consistency cost relative to
    a supervised learning cost (e.g. Cross Entropy). This is intended to be stepped after each
    epoch during training to increase or decrease the weight accordingly. This provides a
    linear ramp weighting function.

    Args:
        last_epoch (int): Number of epochs until scheduler reaches `last_weight`.
        last_weight (float, optional): Final consistency weight. Defaults to 1.
        first_weight (float, optional): Consistency weight at beginning of ramp.
            Defaults to 0.
        epochs_before (int, optional): Number of epochs to hold weight at
            `first_weight` before beginning ramp. Defaults to 0.

    Example:
        >>> alpha = RampCW(last_epoch, last_weight, first_weight, epochs_before)
        >>> for epoch in epochs:
        >>>     train(...)
        >>>     loss = criterion + alpha() * consistency
        >>>     validate(...)
        >>>     alpha.step()
    """
    def __init__(self, last_epoch, last_weight=1, first_weight=0, epochs_before=0):
        self.last_epoch = last_epoch
        self.last_weight = last_weight
        self.first_weight = first_weight
        self.epochs_before = epochs_before
        super(RampCW, self).__init__()

    def _make_ramp(self):
        ramp_steps = np.linspace(self.first_weight, self.last_weight, self.last_epoch)
        # Keep weight fixed initially
        self.ramp = [self.first_weight] * self.epochs_before + ramp_steps.tolist()


class StepCW(_CWScheduler):
    r"""Step function consistency weight scheduler.

    Scheduler function to control a weight, often used to weigh a consistency cost relative to
    a supervised learning cost (e.g. Cross Entropy). This is intended to be stepped after each
    epoch during training to increase or decrease the weight accordingly. This provides a
    step weighting function.

    Args:
        last_epoch (int): Number of epochs until scheduler reaches `last_weight`.
        last_weight (float, optional): Final consistency weight. Defaults to 1.
        first_weight (float, optional): Consistency weight at beginning of ramp.
            Defaults to 0.

    Example:
        >>> alpha = StepCW(last_epoch, last_weight, first_weight)
        >>> for epoch in epochs:
        >>>     train(...)
        >>>     loss = criterion + alpha() * consistency
        >>>     validate(...)
        >>>     alpha.step()
    """
    def __init__(self, last_epoch, last_weight=1, first_weight=0):
        self.last_epoch = last_epoch
        self.last_weight = last_weight
        self.first_weight = first_weight
        super(StepCW, self).__init__()

    def _make_ramp(self):
        self.ramp = [self.first_weight] * self.last_epoch + [self.last_weight]


class QuadraticCW(_CWScheduler):
    r"""Quadratic consistency weight scheduler.

    Scheduler function to control a weight, often used to weigh a consistency cost relative to
    a supervised learning cost (e.g. Cross Entropy). This is intended to be stepped after each
    epoch during training to increase or decrease the weight accordingly. This provides a
    quadratic weighting function.

    Args:
        last_epoch (int): Number of epochs until scheduler reaches `last_weight`.
        last_weight (float, optional): Final consistency weight. Defaults to 1.
        first_weight (float, optional): Consistency weight at beginning of ramp.
            Defaults to 0.
        epochs_before (int, optional): Number of epochs to hold weight at
            `first_weight` before beginning ramp. Defaults to 0.

    Example:
        >>> alpha = QuadraticCW(last_epoch, last_weight, first_weight, epochs_before)
        >>> for epoch in epochs:
        >>>     train(...)
        >>>     loss = criterion + alpha() * consistency
        >>>     validate(...)
        >>>     alpha.step()
    """
    def __init__(self, last_epoch, last_weight=1, first_weight=0, epochs_before=0):
        self.last_epoch = last_epoch
        self.last_weight = last_weight
        self.first_weight = first_weight
        self.epochs_before = epochs_before
        super(QuadraticCW, self).__init__()

    def _make_ramp(self):
        steps = -(np.linspace(-1, 0, self.last_epoch) ** 2) + 1
        # Scale the steps from first_weight to last_weight
        steps = steps * (self.last_weight - self.first_weight) + self.first_weight
        # Keep weight fixed initially
        self.ramp = [self.first_weight] * self.epochs_before + steps.tolist()


def init_model_weights(model, value):
    r"""Set all weights in model to a given value.

    Args:
        model (torch.nn.Module): The model to update. Weight update is performed in place.
        value (float): The weight value.
    """
    def init_weights(m):
        try:
            m.weight.data.fill_(value)
        # Will throw an AttributeError if this layer type has no weight field
        except AttributeError:
            pass

    model.apply(init_weights)


def _print_model_parameters(model):
    r""" Print the names and values of all modules in the network.

    Args:
        model (torch.nn.Module): The model to report information about.

    Returns: No return value

    """
    for name, param in model.named_parameters():
        print(name)
        print(param.data)


class SkewedSigmoidCW(_CWScheduler):
    r"""Skewed sigmoidal consistency weight scheduler with variable ramp up speed.

    Scheduler function to control a weight, often used to weigh a consistency cost relative to
    a supervised learning cost (e.g. Cross Entropy). This is intended to be stepped after each
    epoch during training to increase or decrease the weight accordingly. This provides a
    skewed sigmoid weighting function with variable ramp up timing speed.

    Args:
        last_epoch (int): Number of epochs until scheduler reaches `last_weight`.
        last_weight (float, optional): Final consistency weight. Defaults to 1.
        first_weight (float, optional): Consistency weight at beginning of ramp.
            Defaults to 0.
        epochs_before (int, optional): Number of epochs to hold weight at
            `first_weight` before beginning ramp. Defaults to 0.
        beta (float, optional): Controls how sharp the
            rise from `first_weight` to `last_weight` is. `beta` = 1
            corresponds to a standard sigmoid. Increasing `beta` increases
            sharpness. Negative values can actually invert the sigmoid for
            a decreasing ramp. Defaults to 1.
        zeta (float, optional): Skews when the rise from
            `first_weight` to `last_weight` occurs. `zeta` = 1 corresponds
            to a rise centered about the middle epoch. `zeta` = 0 corresponds
            to a flat weight at `last_weight`. `zeta` < 1 shifts rise
            to earlier epochs. `zeta` > 1 shifts to later epochs. Defaults to 1.

    Example:
        >>> alpha = SkewedSigmoidCW(last_epoch, last_weight, first_weight, epochs_before, beta, zeta)
        >>> for epoch in epochs:
        >>>     train(...)
        >>>     loss = criterion + alpha() * consistency
        >>>     validate(...)
        >>>     alpha.step()
    """
    def __init__(self, last_epoch, last_weight=1, first_weight=0, epochs_before=0, beta=1, zeta=1):
        self.last_epoch = last_epoch
        self.last_weight = last_weight
        self.first_weight = first_weight
        self.epochs_before = epochs_before
        self.beta = beta
        self.zeta = zeta
        super(SkewedSigmoidCW, self).__init__()

    def _make_ramp(self):
        # Calculate a range of sigmoid values
        x = np.linspace(-10, 10, self.last_epoch, endpoint=True)
        # Calculate the linear sampling grid along normal sigmoid
        i = np.linspace(0, 1, self.last_epoch, endpoint=True)
        # skew the sampling grid to sample nonlinear along the sigmoid
        x = i**self.zeta * 2 * max(x) + min(x)
        # Calculate the sigmoid along the nonlinear axis
        sigmoid_steps = 1.0 / (1 + np.exp(-self.beta * (x)))
        # Scale the sigmoid from first_weight to last_weight
        sigmoid_steps = sigmoid_steps * (self.last_weight - self.first_weight) + self.first_weight
        # Keep weight fixed initially
        self.ramp = [self.first_weight] * self.epochs_before + sigmoid_steps.tolist()
