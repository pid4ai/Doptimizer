import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required






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

class Adapidoptimizer(Optimizer):
    def __init__(self, params, lr=required, beta=0.99, momentum=[0.9, 0.9],
                 epsilon=0.0000001, weight_decay=0, I=5., D=10.):
        defaults = dict(lr=lr, beta=beta, momentum=momentum, epsilon=epsilon,
                         weight_decay=weight_decay, I=I, D=D)
        super(Adapidoptimizer, self).__init__(params, defaults)
        self.get_grad_symbol = False

    def __setstate__(self, state):
        super(Adapidoptimizer, self).__setstate__(state)


    def get_oldgrad(self, old_grads):
        self.old_grads = old_grads
        if len(old_grads) != len(self.param_groups[0]['params']):
            raise ValueError('NOT correct grads')
        params = self.param_groups[0]['params']
        for i in range(len(params)):
            if params[i].size() != old_grads[i].size():
                raise ValueError('NOT correct grads')
        self.get_grad_symbol = True

    def step(self, closure=None):
        if len(self.param_groups) != 1:
            raise ValueError('do not support multiple nets')

        if self.get_grad_symbol == False:
            raise ValueError('old grad not got')
        loss = None

        if closure is not None:
            loss = closure()

        for j in range(len(self.param_groups)):
            group = self.param_groups[j]
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            I = group['I']
            D = group['D']
            beta = group['beta']
            epsilon = group['epsilon']
            for i in range(len(group['params'])):
                p = group['params'][i]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum[0] <= 0:
                    raise ValueError('Do not support zero or negative momentum !')
                param_state = self.state[p]
                if 'time_buffer' not in param_state:
                    param_state['time_buffer'] = 1
                else:
                    param_state['time_buffer'] += 1
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (d_p.detach() ** 2) * (1 - beta)
                    v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (d_p.detach() ** 2) * (1 - beta)
                    v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                if 'I_buffer' not in param_state:
                    I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                    I_buf.mul_(momentum[0]).add_(1 - momentum[0], d_p.detach())
                else:
                    I_buf = param_state['I_buffer']
                    I_buf.mul_(momentum[0]).add_(1 - momentum[0], d_p.detach())

                if 'D_buffer' not in param_state:
                    D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                    D_buf.mul_(momentum[1]).add_(1 - momentum[1], d_p.detach() - self.old_grads[i])
                else:
                    D_buf = param_state['D_buffer']
                    D_buf.mul_(momentum[1]).add_(1 - momentum[1], d_p.detach() - self.old_grads[i])
                if 'dv_buffer' not in param_state:
                    param_state['dv_buffer'] = ((d_p.detach() - self.old_grads[i]) ** 2) * (1 - beta)
                    dv_buf = param_state['dv_buffer'] / (1 - beta ** param_state['time_buffer'])
                else:
                    param_state['dv_buffer'] = param_state['dv_buffer'] * beta + ((d_p.detach() - self.old_grads[i]) ** 2) * (1 - beta)
                    dv_buf = param_state['dv_buffer'] / (1 - beta ** param_state['time_buffer'])
                p_grad = (d_p.add_(I, I_buf).add(D, D_buf))/(v_buf ** 0.5 + epsilon)
                p.data.add_(-group['lr'], p_grad)

        self.get_grad_symbol = False

        return loss

class double_Adapidoptimizer(Optimizer):
    def __init__(self, params, lr=required, beta=0.99, momentum=[0.9, 0.9],
                 epsilon=0.0000001, weight_decay=0, I=5., D=10.):
        defaults = dict(lr=lr, beta=beta, momentum=momentum, epsilon=epsilon,
                         weight_decay=weight_decay, I=I, D=D)
        super(double_Adapidoptimizer, self).__init__(params, defaults)
        self.get_grad_symbol = False

    def __setstate__(self, state):
        super(double_Adapidoptimizer, self).__setstate__(state)


    def get_oldgrad(self, old_grads):
        self.old_grads = old_grads
        if len(old_grads) != len(self.param_groups[0]['params']):
            raise ValueError('NOT correct grads')
        params = self.param_groups[0]['params']
        for i in range(len(params)):
            if params[i].size() != old_grads[i].size():
                raise ValueError('NOT correct grads')
        self.get_grad_symbol = True

    def step(self, closure=None):
        if len(self.param_groups) != 1:
            raise ValueError('do not support multiple nets')

        if self.get_grad_symbol == False:
            raise ValueError('old grad not got')
        loss = None

        if closure is not None:
            loss = closure()

        for j in range(len(self.param_groups)):
            group = self.param_groups[j]
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            I = group['I']
            D = group['D']
            beta = group['beta']
            epsilon = group['epsilon']
            for i in range(len(group['params'])):
                p = group['params'][i]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum[0] <= 0:
                    raise ValueError('Do not support zero or negative momentum !')
                param_state = self.state[p]
                if 'time_buffer' not in param_state:
                    param_state['time_buffer'] = 1
                else:
                    param_state['time_buffer'] += 1
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (d_p.detach() ** 2) * (1 - beta)
                    v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (d_p.detach() ** 2) * (1 - beta)
                    v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                if 'I_buffer' not in param_state:
                    I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                    I_buf.mul_(momentum[0]).add_(1 - momentum[0], d_p.detach())
                else:
                    I_buf = param_state['I_buffer']
                    I_buf.mul_(momentum[0]).add_(1 - momentum[0], d_p.detach())

                if 'D_buffer' not in param_state:
                    D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                    D_buf.mul_(momentum[1]).add_(1 - momentum[1], d_p.detach() - self.old_grads[i])
                else:
                    D_buf = param_state['D_buffer']
                    D_buf.mul_(momentum[1]).add_(1 - momentum[1], d_p.detach() - self.old_grads[i])
                if 'dv_buffer' not in param_state:
                    param_state['dv_buffer'] = ((d_p.detach() - self.old_grads[i]) ** 2) * (1 - beta)
                    dv_buf = param_state['dv_buffer'] / (1 - beta ** param_state['time_buffer'])
                else:
                    param_state['dv_buffer'] = param_state['dv_buffer'] * beta + ((d_p.detach() - self.old_grads[i]) ** 2) * (1 - beta)
                    dv_buf = param_state['dv_buffer'] / (1 - beta ** param_state['time_buffer'])
                p_grad = ((d_p.add_(I, I_buf))/(v_buf ** 0.5 + epsilon)).add(D, D_buf/(dv_buf ** 0.5 + epsilon))
                p.data.add_(-group['lr'], p_grad)

        self.get_grad_symbol = False

        return loss

class PIDoptimizer(Optimizer):
    def __init__(self, params, lr=required, beta=0.99, momentum=[0.9, 0.9],
                 epsilon=0.0000001, weight_decay=0, I=5., D=10.):
        defaults = dict(lr=lr, beta=beta, momentum=momentum, epsilon=epsilon,
                         weight_decay=weight_decay, I=I, D=D)
        super(PIDoptimizer, self).__init__(params, defaults)
        self.get_grad_symbol = False

    def __setstate__(self, state):
        super(PIDoptimizer, self).__setstate__(state)


    def get_oldgrad(self, old_grads):
        self.old_grads = old_grads
        if len(old_grads) != len(self.param_groups[0]['params']):
            raise ValueError('NOT correct grads')
        params = self.param_groups[0]['params']
        for i in range(len(params)):
            if params[i].size() != old_grads[i].size():
                raise ValueError('NOT correct grads')
        self.get_grad_symbol = True

    def step(self, closure=None):
        if len(self.param_groups) != 1:
            raise ValueError('do not support multiple nets')

        if self.get_grad_symbol == False:
            raise ValueError('old grad not got')
        loss = None

        if closure is not None:
            loss = closure()

        for j in range(len(self.param_groups)):
            group = self.param_groups[j]
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            I = group['I']
            D = group['D']
            beta = group['beta']
            epsilon = group['epsilon']
            for i in range(len(group['params'])):
                p = group['params'][i]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum[0] <= 0:
                    raise ValueError('Do not support zero or negative momentum !')
                param_state = self.state[p]
                if 'time_buffer' not in param_state:
                    param_state['time_buffer'] = 1
                else:
                    param_state['time_buffer'] += 1
                if 'I_buffer' not in param_state:
                    I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                    I_buf.mul_(momentum[0]).add_(1 - momentum[0], d_p.detach())
                else:
                    I_buf = param_state['I_buffer']
                    I_buf.mul_(momentum[0]).add_(1 - momentum[0], d_p.detach())

                if 'D_buffer' not in param_state:
                    D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                    D_buf.mul_(momentum[1]).add_(1 - momentum[1], d_p.detach() - self.old_grads[i])
                else:
                    D_buf = param_state['D_buffer']
                    D_buf.mul_(momentum[1]).add_(1 - momentum[1], d_p.detach() - self.old_grads[i])
                p_grad = (d_p.add_(I, I_buf).add(D, D_buf))
                p.data.add_(-group['lr'], p_grad)

        self.get_grad_symbol = False

        return loss




