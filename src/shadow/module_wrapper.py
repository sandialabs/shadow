import torch


class ModuleWrapper(torch.nn.Module):
    r"""Base module wrapper for SSML technique implementations.

    Args:
        model (torch.nn.Module): The model to train with semi-supervised learning.
    """
    def __init__(self, model):
        super(ModuleWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        r"""Passes data to the wrapped model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Model output.
        """
        return self.model(x)

    def get_technique_cost(self, x):
        r"""Compute the SSML related cost for the implemented technique.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Technique specific cost.

        Raises:
            NotImplementedError: If not implemented in the specific technique.

        Note:
            This must be implemented for each specific technique that inherits from this base.
        """
        raise NotImplementedError("Must override get_technique_cost function")
