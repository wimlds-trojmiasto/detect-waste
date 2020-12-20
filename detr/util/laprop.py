from torch.optim import Optimizer
import math
import torch

# source:  https://github.com/Z-T-WANG/LaProp-Optimizer
# paper: https://arxiv.org/abs/2002.04839

class LaProp(Optimizer):
    def __init__(self, params, lr=4e-4, betas=(0.9, 0.999), eps=1e-15,
                 weight_decay=0, amsgrad=False, centered=False):

        self.steps_before_using_centered = 10

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, centered=centered)
        super(LaProp, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                centered = group['centered']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of learning rates
                    state['exp_avg_lr_1'] = 0.; state['exp_avg_lr_2'] = 0.
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if centered:
                    # Exponential moving average of gradient values as calculated by beta2
                        state['exp_mean_avg_beta2'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if centered:
                    exp_mean_avg_beta2 = state['exp_mean_avg_beta2']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + (1 - beta1) * group['lr']
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + (1 - beta2)

                bias_correction1 = state['exp_avg_lr_1'] / group['lr'] if group['lr']!=0. else 1. #1 - beta1 ** state['step']
                step_size = 1 / bias_correction1

                bias_correction2 = state['exp_avg_lr_2']
                
                denom = exp_avg_sq
                if centered:
                    exp_mean_avg_beta2.mul_(beta2).add_(1 - beta2, grad)
                    if state['step'] > self.steps_before_using_centered:
                        mean = exp_mean_avg_beta2 ** 2
                        denom = denom - mean

                if amsgrad:
                    if not (centered and state['step'] <= self.steps_before_using_centered): 
                        # Maintains the maximum of all (centered) 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, denom, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq

                denom = denom.div(bias_correction2).sqrt_().add_(group['eps'])
                step_of_this_grad = grad / denom
                exp_avg.mul_(beta1).add_( (1 - beta1) * group['lr'], step_of_this_grad)
                
                p.data.add_(-step_size, exp_avg )
                if group['weight_decay'] != 0:
                    p.data.add_( - group['weight_decay'], p.data)

        return loss
