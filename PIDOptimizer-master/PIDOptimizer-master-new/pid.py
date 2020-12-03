import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required


class PIDOptimizer(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0,  nesterov=False, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, I=I, D=D)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(PIDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PIDOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            dampening = group['dampening']
            nesterov = group['nesterov']
            I = group['I']
            D = group['D'] 
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf.mul_(momentum).add_(d_p)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - dampening, d_p)
                    if 'grad_buffer' not in param_state:
                        g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                        g_buf = d_p   
                        
                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        D_buf.mul_(momentum).add_(d_p-g_buf)
                    else:
                        D_buf = param_state['D_buffer']
                        g_buf = param_state['grad_buffer']                                                
                        D_buf.mul_(momentum).add_(1-momentum, d_p-g_buf)   
                        self.state[p]['grad_buffer']= d_p.clone()   
                    
                        
                    d_p = d_p.add_(I, I_buf).add_(D, D_buf)
                    d_p = d_p
                p.data.add_(-group['lr'], d_p)

        return loss


class decade_PIDOptimizer(Optimizer):

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, I=I, D=D)

        super(decade_PIDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(decade_PIDOptimizer, self).__setstate__(state)


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

                if momentum != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf.mul_(momentum).add_(d_p)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                    if 'grad_buffer' not in param_state:
                        g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                        g_buf = d_p

                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        D_buf.mul_(momentum).add_(d_p - g_buf)
                    else:
                        D_buf = param_state['D_buffer']
                        g_buf = param_state['grad_buffer']
                        D_buf.mul_(momentum).add_(1 - momentum, d_p - g_buf)
                        self.state[p]['grad_buffer'] = d_p.clone()

                    d_p = d_p.add_(I, I_buf).add_(D, D_buf)
                p.data.add_(-group['lr'], d_p)

        return loss


class IDoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, weight_decay=0, I=1., D=10.):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, I=I, D=D)

        super(IDoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(IDoptimizer, self).__setstate__(state)


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

                if momentum != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf.mul_(momentum).add_(d_p.detach())
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p.detach())
                    if 'grad_buffer' not in param_state:
                        g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                        g_buf = d_p.detach()

                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        D_buf.mul_(momentum).add_(d_p.detach() - g_buf)
                    else:
                        D_buf = param_state['D_buffer']
                        g_buf = param_state['grad_buffer']
                        D_buf.mul_(momentum).add_(1 - momentum, d_p.detach() - g_buf)
                        self.state[p]['grad_buffer'] = d_p.detach().clone()
                    p_grad = torch.zeros_like(d_p).add_(I, I_buf).add_(D, D_buf)
                else:
                    p_grad = d_p
                p.data.add_(-group['lr'], p_grad)

        return loss


class AdapidOptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0, beta=0.99,
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
                        I_buf = param_state['I_buffer'] = d_p.detach() * (1 - momentum)
                        I_buf.mul_(momentum).add_(1 - momentum, d_p.detach())
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p.detach())
                    if 'grad_buffer' not in param_state:
                        g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                        g_buf = d_p.detach()

                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        D_buf.mul_(momentum).add_(d_p.detach() - g_buf)
                    else:
                        D_buf = param_state['D_buffer']
                        g_buf = param_state['grad_buffer']
                        D_buf.mul_(momentum).add_(1 - momentum, d_p.detach() - g_buf)
                        self.state[p]['grad_buffer'] = d_p.detach().clone()

                    d_p = (d_p.add_(I, I_buf / (1 - momentum ** param_state['time_buffer'])).add_(D, D_buf)) / (v_buf ** 0.5 + epsilon)
                p.data.add_(-group['lr'], d_p)

        return loss


class Double_Adaptive_PIDOptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0, beta=0.99,
                 weight_decay=0,epsilon=0.0000001, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, beta=beta,
                        weight_decay=weight_decay, epsilon=epsilon, I=I, D=D)
        super(Double_Adaptive_PIDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Double_Adaptive_PIDOptimizer, self).__setstate__(state)

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

            if momentum <= 0:
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
                    g_buf = param_state['grad_buffer'] = torch.zeros_like(p.data)
                    g_buf = d_p.detach()

                    D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                    param_state['dv_buffer'] = torch.zeros_like(d_p)
                    dv_buf = torch.zeros_like(d_p) + 1
                else:
                    D_buf = param_state['D_buffer']
                    g_buf = param_state['grad_buffer']
                    derivative = d_p.detach() - g_buf
                    param_state['dv_buffer'] =  param_state['dv_buffer'] * beta + (1 - beta) * (derivative ** 2)
                    D_buf.mul_(momentum).add_(1 - momentum, d_p.detach() - g_buf)
                    self.state[p]['grad_buffer'] = d_p.detach().clone()
                    dv_buf = param_state['dv_buffer'] / (1 - beta ** (param_state['time_buffer'] - 1))


                if 'I_buffer' not in param_state:
                    I_buf = param_state['I_buffer'] = (1 - momentum) * d_p.detach()
                else:
                    I_buf = param_state['I_buffer']
                    I_buf.mul_(momentum).add_(1 - momentum, d_p.detach())

                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1-beta) * (d_p.detach() ** 2)
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p.detach() ** 2)
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])

                d_p = ((d_p.add_(I, I_buf/(1 - momentum ** param_state['time_buffer'])))/(v_buf ** 0.5 + epsilon)).add_(D, D_buf/(dv_buf ** 0.5 + epsilon))
                p.data.add_(-group['lr'], d_p)

        return loss


class specPIDoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0, weight_decay=0, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, I=I, D=D)
        super(specPIDoptimizer, self).__init__(params, defaults)
        self.get_grad_symbol = False

    def __setstate__(self, state):
        super(specPIDoptimizer, self).__setstate__(state)


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
            for i in range(len(group['params'])):
                p = group['params'][i]
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf.mul_(momentum).add_(d_p.detach())
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p.detach())


                    if 'D_buffer' not in param_state:
                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        D_buf.mul_(momentum).add_(d_p.detach() - self.old_grads[i])
                    else:
                        D_buf = param_state['D_buffer']
                        D_buf.mul_(momentum).add_(1 - momentum, d_p.detach() - self.old_grads[i])

                    p_grad = torch.zeros_like(d_p).add_(I, I_buf).add_(D, D_buf)
                else:
                    p_grad = d_p
                p.data.add_(-group['lr'], p_grad)

        self.get_grad_symbol = False

        return loss


class SVRGoptimizer(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(SVRGoptimizer, self).__init__(params, defaults)
        self.get_grad_symbol = False
        self.grad_renew_symbol = False

    def __setstate__(self, state):
        super(SVRGoptimizer, self).__setstate__(state)


    def get_oldgrad(self, old_grads):
        self.old_grads = old_grads
        if len(old_grads) != len(self.param_groups[0]['params']):
            raise ValueError('NOT correct grads')
        params = self.param_groups[0]['params']
        for i in range(len(params)):
            if params[i].size() != old_grads[i].size():
                raise ValueError('NOT correct grads')
        self.get_grad_symbol = True

    def get_basicgrad(self, basic_grads):
        self.basic_grads = basic_grads
        if len(basic_grads) != len(self.param_groups[0]['params']):
            raise ValueError('NOT correct grads')
        params = self.param_groups[0]['params']
        for i in range(len(params)):
            if params[i].size() != basic_grads[i].size():
                raise ValueError('NOT correct grads')
        self.grad_renew_symbol = True

    def step(self, closure=None):
        loss = None
        if len(self.param_groups) != 1:
            raise ValueError('do not support multiple nets')
        if closure is not None:
            loss = closure()

        if self.get_grad_symbol == False and self.grad_renew_symbol == True:
            for j in range(len(self.param_groups)):
                group = self.param_groups[j]
                weight_decay = group['weight_decay']
                for i in range(len(group['params'])):
                    p = group['params'][i]
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    p.data.add_(-group['lr'], d_p)
            self.grad_renew_symbol = False

        elif self.get_grad_symbol == True:
            for j in range(len(self.param_groups)):
                group = self.param_groups[j]
                weight_decay = group['weight_decay']
                for i in range(len(group['params'])):
                    p = group['params'][i]
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    p_grad = torch.zeros_like(d_p).add_(d_p - self.old_grads[i]).add_(self.basic_grads[i])
                    p.data.add_(-group['lr'], p_grad)

        self.get_grad_symbol = False

        return loss


class SARAHoptimizer(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)

        super(SARAHoptimizer, self).__init__(params, defaults)
        self.get_grad_symbol = False
        self.grad_renew_symbol = False

    def __setstate__(self, state):
        super(SARAHoptimizer, self).__setstate__(state)

    def get_oldgrad(self, old_grads):
        self.old_grads = old_grads
        if len(old_grads) != len(self.param_groups[0]['params']):
            raise ValueError('NOT correct grads')
        params = self.param_groups[0]['params']
        for i in range(len(params)):
            if params[i].size() != old_grads[i].size():
                raise ValueError('NOT correct grads')
        self.get_grad_symbol = True

    def get_basicgrad(self, basic_grads):
        self.basic_grads = basic_grads
        if len(basic_grads) != len(self.param_groups[0]['params']):
            raise ValueError('NOT correct grads')
        params = self.param_groups[0]['params']
        for i in range(len(params)):
            if params[i].size() != basic_grads[i].size():
                raise ValueError('NOT correct grads')
        self.grad_renew_symbol = True

    def step(self, closure=None):
        loss = None
        if len(self.param_groups) != 1:
            raise ValueError('do not support multiple nets')
        if closure is not None:
            loss = closure()

        if self.get_grad_symbol == False and self.grad_renew_symbol == True:
            for j in range(len(self.param_groups)):
                group = self.param_groups[j]
                weight_decay = group['weight_decay']
                for i in range(len(group['params'])):
                    p = group['params'][i]
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    p.data.add_(-group['lr'], d_p)
            self.grad_renew_symbol = False

        elif self.get_grad_symbol == True:
            for j in range(len(self.param_groups)):
                group = self.param_groups[j]
                weight_decay = group['weight_decay']
                for i in range(len(group['params'])):
                    p = group['params'][i]
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    d_buf = d_p - self.old_grads[i]
                    self.basic_grads[i] += d_buf
                    p.data.add_(-group['lr'], self.basic_grads[i])

        self.get_grad_symbol = False

        return loss


class Momentumoptimizer(Optimizer):

    def __init__(self, params, lr=required, momentum=0,  weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(Momentumoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Momentumoptimizer, self).__setstate__(state)

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
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'I_buffer' not in param_state:
                        I_buf = param_state['I_buffer'] = torch.zeros_like(p.data)
                        I_buf.mul_(momentum).add_(d_p)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - momentum, d_p)
                else:
                    raise ValueError('please using SGD instead')

                p.data.add_(-group['lr'], I_buf)

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


class RMSpropOptimizer(Optimizer):
    def __init__(self, params, lr=required, beta=0.99, weight_decay=0,epsilon=0.0000001):
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay, epsilon=epsilon)

        super(RMSpropOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropOptimizer, self).__setstate__(state)

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
                    param_state['v_buffer'] = (1-beta) * (d_p ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p ** 2)
                    param_state['time_buffer'] += 1
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                d_p = d_p / (v_buf ** 0.5 + epsilon)

                p.data.add_(-group['lr'], d_p)

        return loss


class Restrict_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, gamma=0.99, weight_decay=0, epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta,gamma=gamma, weight_decay=weight_decay, epsilon=epsilon)

        super(Restrict_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Restrict_Adamoptimizer, self).__setstate__(state)

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
            gamma = group['gamma']
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
                    v_mean = torch.mean(param_state['v_buffer'].detach())
                    param_state['v_buffer'] = param_state['v_buffer'] * gamma + v_mean * (1 - gamma)
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


class Logrestrict_Adamoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0.9, beta=0.99, gamma=0.999, weight_decay=0, epsilon=0.0000001):
        defaults = dict(lr=lr, momentum=momentum, beta=beta,gamma=gamma, weight_decay=weight_decay, epsilon=epsilon)

        super(Logrestrict_Adamoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Logrestrict_Adamoptimizer, self).__setstate__(state)

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
            gamma = group['gamma']
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
                    v_mean = torch.mean(torch.log(param_state['v_buffer'].detach()))
                    #current_gamma = 1 - np.log(param_state['time_buffer']) * (1 - gamma)
                    param_state['v_buffer'] = torch.exp(torch.log(param_state['v_buffer']) * gamma + v_mean * (1 - gamma))
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
