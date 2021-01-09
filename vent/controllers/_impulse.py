from lung.controllers.core import Controller

class Impulse(Controller):
    def __init__(self, impulse=50, start=0.5, end=0.65):
        self.impulse = impulse
        self.start = start
        self.end = end
        
    def compute_action(self, state, t):
        u_out = self.u_out(t)

        impulse = 0
        if t >= self.start and t <= self.end:
            impulse = self.impulse

        impulse = impulse if u_out == 0 else 0
            
        return impulse, 0
