import torch
import shadow.module_wrapper


class Threshold(torch.nn.Module):
    r"""Per-class thresholding operator.

    Args:
        threshold (torch.Tensor): 1D `float` array of thresholds with
            length equal to the number of classes. Each element should be
            between :math:`[0, 1]` and represents a per-class threshold.
            Thresholds are with respect to normalized scores (e.g. they
            sum to 1).

    Example:
        >>> myThresholder = Threshold([.8, .9])
        >>> myThresholder([[10, 90], [95, 95.4], [0.3, 0.4]])
        [1, 0, 0]
    """
    def __init__(self, thresholds):
        super(Threshold, self).__init__()
        self.thresholds = torch.nn.Parameter(torch.Tensor(thresholds))

    def forward(self, predictions):
        r"""Threshold multi-class scores.

            Args:
                predictions (torch.Tensor): 2D model outputs of shape
                    `(n_samples, n_classes)`. Does not need to be normalized
                    in advance.

            Returns:
                torch.Tensor: binary thresholding for each sample.
        """
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        return torch.any(predictions > self.thresholds, dim=1).float()


class PL(shadow.module_wrapper.ModuleWrapper):
    r"""Pseudo Label model wrapper.


    The pseudo labeling wrapper weight samples according to model score.
    This is a form of entropy regularization. For example, a binary
    random variable with distribution :math:`P(X=1) = .5` and
    :math:`P(X=0) = .5` has a much higher entropy than :math:`P(X=1) = .9`
    and :math:`P(X=0) = .1`.

    Args:
        weight_function (callable): assigns weighting based on raw model
            outputs.
        ssml_mode (bool, optional): semi-supevised learning mode, toggles
            whether loss is computed for all inputs or just those data
            with missing labels. Defaults to True.
        missing_label (int, optional): integer value used to represent
            missing labels. Defaults to -1.
    """
    def __init__(self, model, weight_function, ssml_mode=True, missing_label=-1):
        super(PL, self).__init__(model)
        self.weight_function = weight_function
        self.ssml_mode = ssml_mode
        self.missing_label = missing_label

        # if ssml_mode
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def get_technique_cost(self, x, targets):
        r"""Compute loss from pseudo labeling.

        Args:
            x (torch.Tensor): Tensor of the data
            targets (torch.Tensor): 1D Corresponding labels. Unlabeled
                data is specified according to `self.missing_label`.

        Returns:
            torch.Tensor: Pseudo label loss.
        """
        # Get the predicted labels from the model
        predictions = self.model(x)
        predicted_labels = torch.argmax(predictions, 1)

        # Unlabeled samples in batch
        unlabeled = targets == self.missing_label

        # For unlabeled data, 'fake' the labels
        semi_fake_targets = torch.where(unlabeled, predicted_labels, targets)

        # Per sample weighting samples
        _weights = self.weight_function(predictions).float()

        # If in ssml mode, don't weight the labeled samples
        if self.ssml_mode:
            _weights[~unlabeled] = 0

        # Calculate cross entropy loss for all predictions regardless if labeled or unconfident
        per_sample_loss = self.loss(predictions, semi_fake_targets) * _weights

        # Only count those samples with non-zero weight
        indexes_to_keep = _weights != 0

        # If there are no samples to keep, return 0 for the loss
        if indexes_to_keep.sum() == 0:
            return torch.Tensor([0]).to(per_sample_loss.device)

        return per_sample_loss[indexes_to_keep].mean()
