import numpy as np
from lung.controllers.core import Controller


class Explorer(Controller):
    def __init__(self, period=0.5, u_in_max=5, p_switch=0.1, max_inhale=2.5):
        self.period = period
        self.u_in_max = u_in_max
        self.p_switch = p_switch
        self.max_inhale = max_inhale

        self.prev_cycle = -1
        self.last_exhale_time = 0

        self.u_out = 0

    def compute_action(self, state, t):
        cur_cycle = int(t / self.period)
        if self.prev_cycle != cur_cycle:
            self.prev_cycle = cur_cycle
            self.u_in = np.random.randint(0, self.u_in_max + 1)

            if np.random.random() < self.p_switch:
                self.u_out = 1 - self.u_out

            if state > 45:
                self.u_out = 1

        if self.u_out == 1:
            self.last_exhale_time = t
        else:
            if t - self.last_exhale_time > self.max_inhale:
                self.u_out = 1
                self.last_exhale_time = t

        return (self.u_in, self.u_out)
