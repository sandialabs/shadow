from torch.nn import functional as F


def softmax_mse_loss(input_logits, target_logits, reduction='sum'):
    r"""Apply softmax and compute mean square error between two model outputs.

    Args:
        input_logits (torch.Tensor): The input unnormalized log probabilities.
        target_logits (torch.Tensor): The target unnormalized log probabilities.
        reduction (string, optional): The reduction parameter passed to
            torch.nn.functional.mse_loss. Defaults to 'sum'.

    Returns:
        torch.Tensor: Mean squared error.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction=reduction) / num_classes


def softmax_kl_loss(input_logits, target_logits, reduction='sum'):
    r"""Apply softmax and compute KL divergence between two model outputs.

    Args:
        input_logits (torch.Tensor): The input unnormalized log probabilities.
        target_logits (torch.Tensor): The target unnormalized log probabilities.
        reduction (string, optional): The reduction parameter passed to
            torch.nn.functional.kl_div. Defaults to 'sum'.

    Returns:
        torch.Tensor: KL divergence.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction=reduction)


def accuracy(y_pred, y):
    r"""Classification accuracy.

    Args:
        y_pred (array_like): Predicted labels.
        y (array_like): True labels.

    Returns:
        float: Classification accuracy percentage.
    """
    return 100 * (y_pred == y).sum().double() / len(y)
