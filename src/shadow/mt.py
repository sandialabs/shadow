import torch
import numpy as np
import copy
import shadow.losses
import shadow.module_wrapper


def ema_update_model(student_model, ema_model, alpha, global_step):
    r"""Exponential moving average update of a model.

    Update `ema_model` to be the moving average of consecutive `student_model` updates via an
    exponential weighting (as defined in [Tarvainen17]_). Update is performed in-place.

    Args:
        student_model (torch.nn.Module): The student model.
        ema_model (torch.nn.Module): The model to update (teacher model).  Update is
            performed in-place.
        alpha (float): Exponential moving average smoothing coefficient, between [0, 1].
        global_step (int): A running count of exponential update steps (typically mini-batch
            updates).
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Smoothing coefficient must be on [0, 1].")
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), student_model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))


class MT(shadow.module_wrapper.ModuleWrapper):
    r"""Mean Teacher [Tarvainen17]_ model wrapper for consistency regularization.

    Mean Teacher model wrapper the provides both student and teacher model implementation.
    The teacher model is a running average of the student weights, and is updated during training.
    When switched to eval mode, the teacher model is used for predictions instead of the student.
    As the wrapper handles the hand off between student and teacher models, the wrapper should be
    used instead of the student model directly.

    Args:
        model (torch.nn.Module): The student model.
        alpha (float, optional): The teacher exponential moving average smoothing coefficient.
            Defaults to 0.999.
        noise (float, optional): If > 0.0, the standard deviation of gaussian noise to apply to
            the input. Specifically, generates random numbers from a normal distribution with
            mean 0 and variance 1, and then scales them by this factor and adds to the input data.
            Defaults to 0.1.
        consistency_type ({'kl', 'mse'}, optional): Cost function used to measure consistency.
           Defaults to `'mse'` (mean squared error).
    """
    def __init__(self, model, alpha=0.999, noise=0.1, consistency_type="mse"):
        super(MT, self).__init__(model)
        # model is the 'student' model.
        # the 'teacher' model is the EMA model. It starts as a copy of the student model.
        self.teacher_model = copy.deepcopy(self.model)
        # We don't want the teacher model to be updated by optimizer.
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.alpha = alpha
        self.student_noise = noise
        self.teacher_noise = noise

        if consistency_type == 'mse':
            self.consistency_criterion = shadow.losses.softmax_mse_loss
        elif consistency_type == 'kl':
            self.consistency_criterion = shadow.losses.softmax_kl_loss
        else:
            raise ValueError(
                "Unknown consistency type. Should be 'mse' or 'kl', but is " + str(consistency_type)
            )

        # The global step is the count of the student and teacher training passes. It is essentially
        # a cumulative count of the batches, starting at 1.
        self.global_step = 1

    def calc_student_logits(self, x):
        r"""Student model logits, with noise added to the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: The student logits.
        """
        model_input = x
        with torch.no_grad():
            if not np.isclose(self.student_noise, 0.0) and self.student_noise > 0.0:
                model_input = (torch.randn_like(x) * self.student_noise) + x
        return self.model(model_input)

    def calc_teacher_logits(self, x):
        r"""Teacher model logits.

        The teacher model logits, with noise added to the input data. Does not propagate gradients
        in the teacher forward pass.

        Args:
            x (torch.Tensor): Input data.
        Returns:
            torch.Tensor: The teacher logits.
        """
        model_input = x
        with torch.no_grad():
            if not np.isclose(self.teacher_noise, 0.0) and self.teacher_noise > 0.0:
                model_input = (torch.randn_like(x) * self.teacher_noise) + x
            # Don't want to update teacher model gradients
            return self.teacher_model(model_input)

    def forward(self, x):
        r"""Model forward pass.

        During model training, adds noise to the input data and passes through the student model.
        During model evaluation, does not add noise and passes through the teacher model.

        Args:
            x (torch.Tensor): Input data.
        Returns:
            torch.Tensor: Model output.
        """
        if self.training:
            return self.calc_student_logits(x)
        else:
            return self.teacher_model(x)

    def get_technique_cost(self, x):
        r"""Consistency cost between student and teacher models.

        Consistency cost between the student and teacher, updates teacher weights via exponential
        moving average of the student weights. Noise is sampled and applied to student and teacher
        separately.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Consistency cost between the student and teacher model outputs.
        """
        # Update the teacher model if not the first step.
        if self.model.training:
            if self.global_step > 1:
                ema_update_model(self.model, self.teacher_model, self.alpha, self.global_step)
            self.global_step += 1

        # Calc teacher logits (non-normalized model output)
        ema_logits = self.calc_teacher_logits(x)

        # Calc student logits (non-normalized model output)
        model_logits = self.calc_student_logits(x)

        # The minibatch size is required in order to find the mean loss
        minibatch_size = x.shape[0]
        consistency_loss = self.consistency_criterion(model_logits, ema_logits) / minibatch_size
        return consistency_loss

    def update_ema_model(self):
        r"""Exponential moving average update of the teacher model."""
        ema_update_model(self.model, self.teacher_model, self.alpha, self.global_step)

    def get_evaluation_model(self):
        r"""The teacher model, which should be used for prediction during evaluation.

        Returns:
            torch.nn.Module: The teacher model.
        """
        return self.teacher_model
