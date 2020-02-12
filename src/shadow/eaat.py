import shadow.losses
import shadow.utils
import shadow.vat
import shadow.mt


class EAAT(shadow.mt.MT):
    r"""Exponential Averaging Adversarial Training (EAAT, [Linville20]_) model wrapper for consistency regularization.

    Computes consistency using the teacher-student paradigm followed by Mean-Teacher
    ([Tarvainen17]_) with virtual adversarial perturbations ([Miyato18]_) applied to student
    model inputs.

    Args:
        model (torch.nn.Module): The student model.
        alpha (float, optional): The teacher exponential moving average smoothing coefficient.
            Defaults to 0.999.
        student_noise (float, optional): If > 0.0, the standard deviation of gaussian noise to
            apply to the student input (in addition to virtual adversarial noise). Specifically,
            generates random numbers from a normal distribution with mean 0 and variance 1, and
            then scales them by this factor and add to the input data. Defaults to 0.1.
        teacher_noise (float, optional): If > 0.0, the standard deviation of gaussian noise to
            apply to the teacher input. Specifically, generates random numbers from a normal
            distribution with mean 0 and variance 1, and then scales them by this factor and add
            to the input data. Defaults to 0.1.
        xi (float, optional): Scaling value for the random direction vector. Defaults to 1.0
        eps (float, optional): The magnitude of applied adversarial perturbation. Greater `eps`
            implies more smoothing. Defaults to 1.0
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
    """
    def __init__(self, model, alpha=0.999, student_noise=0.1, teacher_noise=0.1,
                 xi=1.0, eps=1.0, power_iter=1, consistency_type="kl",
                 flip_correction=True):
        super(EAAT, self).__init__(model, alpha, student_noise, consistency_type)

        # additional MT param
        self.teacher_noise = teacher_noise

        # VAT params
        self.xi = xi
        self.eps = eps
        self.power_iter = power_iter
        self.flip_correction = flip_correction
        if self.power_iter <= 0:
            self.power_iter = 1

    def calc_student_logits(self, x):
        r"""Student model logits, with perturbations added to the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: The student logits.
        """
        r = shadow.vat.vadv_perturbation(x, self.model, self.xi, self.eps,
                                         self.power_iter, self.consistency_criterion,
                                         flip_correction=self.flip_correction)
        return self.model(x + r)
