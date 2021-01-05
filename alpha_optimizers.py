import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
from DNN_models import cifar10_CNN, cifar10_DenseNet, cifar10_ResNet18



class Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adamoptimizer, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2)
                    param_state['time_buffer'] += 1
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')

                p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** 0.5 + epsilon))

        return loss


class alpha_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001, alpha=2):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon, alpha=alpha)

        super(alpha_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha_Adamoptimizer, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            alpha=group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (torch.abs(d_p.clone()) ** alpha)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1-beta) * (torch.abs(d_p.clone()) ** alpha)
                    param_state['time_buffer'] += 1
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')

                p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (1 / alpha) + epsilon))

        return loss

class alpha_SGDoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0, alpha=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, alpha=alpha)

        super(alpha_SGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha_SGDoptimizer, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            alpha=group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'time_buffer' not in param_state:
                    param_state['time_buffer'] = 1
                else:
                    param_state['time_buffer'] += 1
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = (torch.sign(d_p.clone()) * (torch.abs(d_p.clone()) ** alpha)) * (1 - momentum)
                    else:
                        param_state['I_buffer'] = param_state['I_buffer'] * momentum + (1 - momentum) * (torch.sign(d_p) * (torch.abs(d_p) ** alpha))
                else:
                    raise ValueError('Please using RMSprop instead')

                p.data.add_(-group['lr'], ((torch.sign(param_state['I_buffer']) *
                                            (torch.abs(param_state['I_buffer']) ** (1 / alpha))) / (1 - momentum ** param_state['time_buffer'])))

        return loss

class alpha_ascent_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(alpha_ascent_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha_ascent_Adamoptimizer, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'time_buffer' not in param_state:
                    param_state['v_buffer0'] = (1 - beta) * (torch.abs(d_p.clone()) ** 1.5)
                    param_state['v_buffer1'] = (1 - beta) * (torch.abs(d_p.clone()) ** 3)
                    param_state['time_buffer'] = 1
                    v_buf = param_state['v_buffer0'] / (1 - beta ** param_state['time_buffer'])
                elif param_state['time_buffer'] < 1000:
                    param_state['v_buffer0'] = param_state['v_buffer0'] * beta + (1 - beta) * (torch.abs(d_p.clone()) ** 1.5)
                    param_state['v_buffer1'] = param_state['v_buffer1'] * beta + (1 - beta) * (torch.abs(d_p.clone()) ** 3)
                    param_state['time_buffer'] += 1
                    v_buf = param_state['v_buffer0'] / (1 - beta ** param_state['time_buffer'])
                else:
                    param_state['v_buffer1'] = param_state['v_buffer1'] * beta + (1 - beta) * (torch.abs(d_p.clone()) ** 3)
                    param_state['time_buffer'] += 1
                    v_buf = param_state['v_buffer0'] / (1 - beta ** param_state['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')

                if param_state['time_buffer'] <= 1000:
                    p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (2/3) + epsilon))
                else:
                    p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (1/3) + epsilon))

        return loss

class double_alpha_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001, alpha=[2,2]):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon, alpha=alpha)

        super(double_alpha_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(double_alpha_Adamoptimizer, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (torch.abs(d_p.clone()) ** alpha[0])
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1-beta) * (torch.abs(d_p.clone()) ** alpha[0])
                    param_state['time_buffer'] += 1
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')

                p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (1 / alpha[1]) + epsilon))

        return loss


class alpha2ascent_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, sgd_lr=0.01, momentum=0.9, beta=0.99, alpha=2, weight_decay=0, epsilon=0.0000001):
        lr_amplify = 1/sgd_lr
        basic_lr = lr * lr_amplify
        defaults = dict(basic_lr = basic_lr, momentum=momentum, lr_amplify=lr_amplify, beta=beta, alpha=alpha,  weight_decay=weight_decay, epsilon=epsilon)

        super(alpha2ascent_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha2ascent_Adamoptimizer, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1-beta) * (d_p.clone() ** 2)
                    param_state['time_buffer'] += 1
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                alpha2 = alpha + param_state['time_buffer'] / 2000
                lr = (group['basic_lr'] ** (1 / alpha2)) / group['lr_amplify']
                p.data.add_(-lr, (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** (1 / alpha2) + epsilon))

        return loss

class alpha2_SGDoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0, alpha=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, alpha=alpha)

        super(alpha2_SGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(alpha2_SGDoptimizer, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            alpha=group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'time_buffer' not in param_state:
                    param_state['time_buffer'] = 1
                else:
                    param_state['time_buffer'] += 1
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.clone() * (1 - momentum)
                    else:
                        param_state['I_buffer'] = param_state['I_buffer'] * momentum + (1 - momentum) * d_p.clone()
                else:
                    raise ValueError('Please using RMSprop instead')

                p.data.add_(-group['lr'], param_state['I_buffer'] / (1 - momentum ** param_state['time_buffer']))#((torch.sign(param_state['I_buffer']) * (torch.abs(param_state['I_buffer']) ** (1/alpha))) / (1 - momentum ** param_state['time_buffer'])))

        return loss

class SGD_momentumoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)

        super(SGD_momentumoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_momentumoptimizer, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'time_buffer' not in param_state:
                    param_state['time_buffer'] = 1
                else:
                    param_state['time_buffer'] += 1
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.clone() * (1 - momentum)
                    else:
                        param_state['I_buffer'] = param_state['I_buffer'] * momentum + (1 - momentum) * d_p.clone()
                else:
                    raise ValueError('Please using RMSprop instead')

                p.data.add_(-group['lr'], param_state['I_buffer'] / (1 - momentum ** param_state['time_buffer']))

        return loss

class Adam_to_SGDoptimizer(Optimizer):
    def __init__(self, params, lr=required, sgd_lr=0.01, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, sgd_lr=sgd_lr, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(Adam_to_SGDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_to_SGDoptimizer, self).__setstate__(state)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            epsilon = group['epsilon']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.clone() ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.clone() ** 2)
                    param_state['time_buffer'] += 1
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                if momentum != 0:
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf = d_p * (1 - momentum)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('Please using RMSprop instead')
                if param_state['time_buffer'] < 2000:
                    p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** 0.5 + epsilon))
                else:
                    p.data.add_(-group['lr'], I_buf / (1 - momentum ** param_state['time_buffer']))
        return loss




