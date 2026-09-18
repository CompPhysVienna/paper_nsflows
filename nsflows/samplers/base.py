class base_sampler(object):

    def __init__(self, system, n_cycles, step_size):
        
        self.system = system
        self.n_cycles = n_cycles
        self.step_size = step_size

        self.x0 = None

    def cycle(self, x, u_x, dx, arg):
        raise NotImplementedError
    
    def propagate(self, N, arg):
        raise NotImplementedError