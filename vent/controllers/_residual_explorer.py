import numpy as np
from vent.controllers.core import Controller

class ResidualExplorer(Controller):
    def __init__(self, base_controller, **kwargs):
        self.base_controller = base_controller
        self.prev_u_out = 1

        self.delta_range = (-30, 30)
        self.duration_range = (0.1, 0.5)

        self.needs_resample = False
        self.resample_delta()

    # def feed(self, state, t):
        # base_u_in, base_u_out = self.base_controller.feed(state, t)
        # return super().feed((state, base_u_in, base_u_out), t)

    def resample_delta(self):
        # random triangular bump
        height = np.random.uniform(*self.delta_range)
        duration = np.random.uniform(*self.duration_range)
        # TODO(cyril): tune magic numbers, add/interpolate step function residual

        t_min, t_max = 0, self.waveform._keypoints[2]

        t_begin = np.random.uniform(t_min, t_max - duration)
        t_mid = t_begin + 0.5 * duration
        t_end = t_begin + duration

        self.xp = np.array([t_min, t_begin, t_mid, t_end, t_max])
        self.fp = np.array([0., 0., height, 0., 0.])


    def compute_action(self, state, t):
        base_u_in, base_u_out = self.base_controller.feed(state, t)

        if base_u_out == 1: # if base exhales, just follow base
            self.u_in = base_u_in
            self.u_out = 1
            if self.needs_resample:
                self.resample_delta()
                self.needs_resample = False
        else: # if base inhales, add residual to cycle phase
            self.needs_resample = True
            self.u_in = base_u_in + np.interp(self.cycle_phase(t), self.xp, self.fp)
            self.u_out = 0
            
        return (np.clip(self.u_in, 0, 100), self.u_out)
