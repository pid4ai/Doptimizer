import torch
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

class specPIDoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, I=I, D=D)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(specPIDoptimizer, self).__init__(params, defaults)
        self.get_grad_symbol = False

    def __setstate__(self, state):
        super(specPIDoptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)



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
            dampening = group['dampening']
            nesterov = group['nesterov']
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
                        I_buf.mul_(momentum).add_(d_p)
                    else:
                        I_buf = param_state['I_buffer']
                        I_buf.mul_(momentum).add_(1 - dampening, d_p)


                    if 'D_buffer' not in param_state:
                        D_buf = param_state['D_buffer'] = torch.zeros_like(p.data)
                        D_buf.mul_(momentum).add_(d_p - self.old_grads[i])
                    else:
                        D_buf = param_state['D_buffer']
                        D_buf.mul_(momentum).add_(1 - momentum, d_p - self.old_grads[i])

                    d_p = d_p.add_(I, I_buf).add_(D, D_buf)
                p.data.add_(-group['lr'], d_p)

        self.get_grad_symbol = False

        return loss


class AdapidOptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, beta=0.99,
                 weight_decay=0,epsilon=0.0000001, nesterov=False, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, beta=beta,
                        weight_decay=weight_decay, epsilon=epsilon, nesterov=nesterov, I=I, D=D)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AdapidOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdapidOptimizer, self).__setstate__(state)
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
            epsilon = group['epsilon']
            beta = group['beta']
            nesterov = group['nesterov']
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
                    param_state['v_buffer'] = (1-beta) * (d_p ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (d_p ** 2)
                    param_state['time_buffer'] += 1
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                d_p = d_p / (v_buf ** 0.5 + epsilon)
                if momentum != 0:
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
                        D_buf.mul_(momentum).add_(d_p - g_buf)
                    else:
                        D_buf = param_state['D_buffer']
                        g_buf = param_state['grad_buffer']
                        D_buf.mul_(momentum).add_(1 - momentum, d_p - g_buf)
                        self.state[p]['grad_buffer'] = d_p.clone()

                    d_p = d_p.add_(I, I_buf).add_(D, D_buf)
                p.data.add_(-group['lr'], d_p)

        return loss

class AdapidOptimizer_test(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, beta=0.99,
                 weight_decay=0,epsilon=0.0000001, nesterov=False, I=5., D=10.):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, beta=beta,
                        weight_decay=weight_decay, epsilon=epsilon, nesterov=nesterov, I=I, D=D)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AdapidOptimizer_test, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdapidOptimizer_test, self).__setstate__(state)
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
            epsilon = group['epsilon']
            beta = group['beta']
            nesterov = group['nesterov']
            I = group['I']
            D = group['D']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
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
                        D_buf.mul_(momentum).add_(d_p - g_buf)
                    else:
                        D_buf = param_state['D_buffer']
                        g_buf = param_state['grad_buffer']
                        D_buf.mul_(momentum).add_(1 - momentum, d_p - g_buf)
                        self.state[p]['grad_buffer'] = d_p.clone()

                    d_p = d_p.add_(I, I_buf).add_(D, D_buf)

                grad_renew = d_p * -group['lr']
                if 'v_buffer' not in param_state:
                    param_state['v_buffer'] = (1 - beta) * (grad_renew.detach() ** 2)
                    param_state['time_buffer'] = 1
                else:
                    param_state['v_buffer'] = param_state['v_buffer'] * beta + (1 - beta) * (grad_renew.detach() ** 2)
                    param_state['time_buffer'] += 1
                    #print(1 - beta ** param_state['time_buffer'])
                v_buf = param_state['v_buffer'] / (1 - beta ** param_state['time_buffer'])
                grad_renew = grad_renew / (v_buf ** 0.5 + epsilon)

                p.data.add_(1, grad_renew)

        return loss

class SVRGoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, beta=0.99,
                 weight_decay=0,epsilon=0.0000001, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, beta=beta,
                        weight_decay=weight_decay, epsilon=epsilon, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SVRGoptimizer, self).__init__(params, defaults)
        self.get_grad_symbol = False
        self.grad_renew_symbol = False

    def __setstate__(self, state):
        super(SVRGoptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
                    d_p1 = torch.zeros_like(d_p).add_(d_p - self.old_grads[i]).add_(self.basic_grads[i])
                    p.data.add_(-group['lr'], d_p1)

        self.get_grad_symbol = False

        return loss


class SARAHoptimizer(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, beta=0.99,
                 weight_decay=0, epsilon=0.0000001, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, beta=beta,
                        weight_decay=weight_decay, epsilon=epsilon, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SARAHoptimizer, self).__init__(params, defaults)
        self.get_grad_symbol = False
        self.grad_renew_symbol = False

    def __setstate__(self, state):
        super(SARAHoptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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


