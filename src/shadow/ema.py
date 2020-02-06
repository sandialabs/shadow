
def update_model(student_model, ema_model, alpha, global_step):
    r"""Exponential moving average update of a model.

    Update `ema_model` to be the moving average of consecutive `student_model` updates via an
    exponential weighting (as defined in [Tarvainen17]_). Update is performed in-place.

    Args:
        student_model (torch.nn.Module): The student model.
        ema_model (torch.nn.Module): The model to update (teacher model).  Update is
            performed in-place.
        alpha (float): Exponential moving average smoothing coefficient.
        global_step (int): A running count of exponential update steps (typically mini-batch
            updates).
    """
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))
