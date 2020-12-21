import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required


class PIDOptimizer(Optimizer):

    def __init__(self, params, lr=required, momentum=[0.9, 0.9], weight_decay=0, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, I=I, D=D)

        super(PIDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PIDOptimizer, self).__setstate__(state)


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
            I = group['I']
            D = group['D']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum[0] != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.detach() * 0.1
                    else:
                        param_state['I_buffer'] = param_state['I_buffer'] * 0.9 + d_p.detach() * 0.1
                    if 'grad_buffer' not in param_state:
                        param_state['grad_buffer'] = d_p.detach()
                        param_state['D_buffer'] = 0.1 * d_p.detach()
                    else:
                        param_state['D_buffer'] = param_state['D_buffer'] * 0.9 + 0.1 * (d_p.detach() - param_state['grad_buffer'])
                        param_state['grad_buffer'] = d_p.detach()

                    grad = d_p.detach().add(I, param_state['I_buffer']).add(D, param_state['D_buffer'])
                p.data.add_(-group['lr'], grad)

        return loss

class PIDOptimizer_test(Optimizer):

    def __init__(self, params, lr=required, momentum=[0.9, 0.9], weight_decay=0, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, I=I, D=D)

        super(PIDOptimizer_test, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PIDOptimizer_test, self).__setstate__(state)

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
            I = group['I']
            D = group['D']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum[0] != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.detach() * 0.1
                        param_state['old_I_buffer'] = torch.zeros_like(param_state['I_buffer'])
                    else:
                        param_state['old_I_buffer'] = param_state['I_buffer']
                        param_state['I_buffer'] = param_state['I_buffer'] * 0.9 + d_p.detach() * 0.1

                    grad = d_p.detach().add(I, param_state['I_buffer']).add(D, (param_state['I_buffer'] - param_state['old_I_buffer']))
                p.data.add_(-group['lr'], grad)

        return loss

class PIDOptimizer_test1(Optimizer):

    def __init__(self, params, lr=required, momentum=[0.9, 0.9], weight_decay=0, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, I=I, D=D)

        super(PIDOptimizer_test1, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PIDOptimizer_test1, self).__setstate__(state)

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
            I = group['I']
            D = group['D']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum[0] != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.detach() * 0.1
                        param_state['old_I_buffer'] = torch.zeros_like(param_state['I_buffer'])
                    else:
                        param_state['old_I_buffer'] = param_state['I_buffer']
                        param_state['I_buffer'] = param_state['I_buffer'] * 0.9 + d_p.detach() * 0.1

                    grad = d_p.detach().add(I, param_state['I_buffer']).add(D, (0.1 * d_p) - (0.1 * param_state['old_I_buffer']))
                p.data.add_(-group['lr'], grad)

        return loss

class PIDOptimizer_test2(Optimizer):

    def __init__(self, params, lr=required, momentum=[0.9, 0.9], weight_decay=0, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, I=I, D=D)

        super(PIDOptimizer_test2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PIDOptimizer_test2, self).__setstate__(state)

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
            I = group['I']
            D = group['D']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum[0] != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        param_state['I_buffer'] = d_p.detach() * 0.1
                        param_state['old_I_buffer'] = torch.zeros_like(param_state['I_buffer'])
                    else:
                        param_state['old_I_buffer'] = param_state['I_buffer']
                        param_state['I_buffer'] = param_state['I_buffer'] * 0.9 + d_p.detach() * 0.1

                    grad = d_p.detach().add(I, param_state['I_buffer']).add(D, (d_p / 9) - (param_state['I_buffer'] / 9))
                p.data.add_(-group['lr'], grad)

        return loss

class AdapidOptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=[0.95 , 0.8], beta=0.99,
                 weight_decay=0,epsilon=0.0000001, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, beta=beta,
                        weight_decay=weight_decay, epsilon=epsilon, I=I, D=D)
        super(AdapidOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdapidOptimizer, self).__setstate__(state)

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
            I = group['I']
            D = group['D']

            if momentum[0] <= 0:
                raise ValueError('this algorithm must have a positive momentum')

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

                if 'grad_buffer' not in param_state:
                    param_state['grad_buffer'] = torch.zeros_like(p.data)
                    D_buf = param_state['D_buffer'] = (1 - momentum[1]) * d_p
                else:
                    D_buf = param_state['D_buffer']
                    g_buf = param_state['grad_buffer']

                    D_buf.mul_(momentum[1]).add_(1 - momentum[1], d_p.detach() - g_buf)
                    param_state['grad_buffer'] = d_p.detach()

                if 'dv_buffer' not in param_state:
                    param_state['dv_buffer'] = (1 - beta) * d_p.detach() ** 2
                else:
                    param_state['dv_buffer'] = param_state['dv_buffer'] * beta + (1 - beta) * (d_p.detach() - param_state['grad_buffer']) ** 2
                dv_buf = param_state['dv_buffer'] / (1 - beta ** param_state['time_buffer'])


                if 'I_buffer' not in param_state:
                    I_buf = param_state['I_buffer'] = (1 - momentum[0]) * d_p.detach()
                else:
                    I_buf = param_state['I_buffer']
                    I_buf.mul_(momentum[0]).add_(1 - momentum[0], d_p.detach())

                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.detach() ** 2)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.detach() ** 2)
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])

                d_p = ((d_p.add_(I, I_buf/(1 - momentum[0] ** param_state['time_buffer'])))/(v_buf ** 0.5 + epsilon)).add_(D, D_buf/(dv_buf ** 0.5 + epsilon))
                p.data.add_(-group['lr'], d_p)

        return loss

class AdapidOptimizer_test(Optimizer):
    def __init__(self, params, lr=required, momentum=[0.9 , 0.9], beta=0.99,
                 weight_decay=0,epsilon=0.0000001, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, beta=beta,
                        weight_decay=weight_decay, epsilon=epsilon, I=I, D=D)
        super(AdapidOptimizer_test, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdapidOptimizer_test, self).__setstate__(state)

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
            I = group['I']
            D = group['D']

            if momentum[0] <= 0:
                raise ValueError('this algorithm must have a positive momentum')

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

                if 'I_buffer' not in param_state:
                    I_buf = param_state['I_buffer'] = (1 - momentum[0]) * d_p.detach()
                else:
                    I_buf = param_state['I_buffer']
                    I_buf.mul_(momentum[0]).add_(1 - momentum[0], d_p.detach())

                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.detach() ** 2)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.detach() ** 2)
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                Adaptive = int(D/9)
                if 'dv_buffer' not in param_state:
                    param_state['dv_buffer'] = (1-beta) * (((d_p - I_buf) * Adaptive) ** 2)
                else:
                    param_state['dv_buffer'] = param_state['dv_buffer'] * beta + (1 - beta) * (((d_p - I_buf) * Adaptive) ** 2)
                dv_buf = param_state['dv_buffer'] / (1 - beta ** param_state['time_buffer'])


                d_p = ((d_p.add_(I, I_buf/(1 - momentum[0] ** param_state['time_buffer'])))/(v_buf ** 0.5 + epsilon)).add_(Adaptive, (d_p - I_buf)/(dv_buf ** 0.5 + epsilon))
                p.data.add_(-group['lr'], d_p)

        return loss

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
                    param_state['v_buffer'] = (1-beta) * (d_p.detach() ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.detach() ** 2)
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
                    raise ValueError('Please using RMSprop insteaad')

                p.data.add_(-group['lr'], (I_buf / (1 - momentum ** param_state['time_buffer'])) / (v_buf ** 0.5 + epsilon))

        return loss


class I_decade_Adaoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(I_decade_Adaoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(I_decade_Adaoptimizer, self).__setstate__(state)

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
                    param_state['v_buffer'] = (1-beta) * (d_p.detach() ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.detach() ** 2)
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
                    raise ValueError('Please using RMSprop insteaad')
                I_rate = 0.7 + 0.3 * (0.99 ** param_state['time_buffer'])
                delta = (d_p * (1 - I_rate) + (I_buf / (1 - momentum ** param_state['time_buffer'])) * I_rate) / (v_buf ** 0.5 + epsilon)

                p.data.add_(-group['lr'], delta)

        return loss






