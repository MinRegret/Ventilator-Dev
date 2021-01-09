import numpy as np
from vent.controllers.core import Controller


class SpikyExplorer(Controller):
    def __init__(
        self,
        period=0.5,
        u_in_max=5,
        p_switch=0.1,
        max_inhale=2.5,
        spike_duration=0.1,
        spike_prob=0.2,
        spike_min=20,
    ):
        self.period = period
        self.u_in_max = u_in_max
        self.p_switch = p_switch
        self.max_inhale = max_inhale
        self.spike_duration = spike_duration
        self.spike_prob = spike_prob
        self.spike_min = spike_min

        self.prev_cycle = -1
        self.last_exhale_time = 0
        self.last_cycle_start = 0
        self.spike_value = 0

        self.u_out = 0

    def compute_action(self, state, t):
        cur_cycle = int(t / self.period)

        if self.prev_cycle != cur_cycle:
            self.prev_cycle = cur_cycle
            self.last_cycle_start = t

            self.u_in = np.random.randint(0, self.u_in_max + 1)

            if np.random.random() < self.p_switch:
                self.u_out = 1 - self.u_out

            if np.random.random() < self.spike_prob and state < self.spike_min:
                self.spike_value = np.random.randint(self.u_in_max + 1, 101)
            else:
                self.spike_value = self.u_in

        if self.u_out == 1:
            self.last_exhale_time = t
        else:
            if t - self.last_exhale_time > self.max_inhale:
                self.u_out = 1
                self.last_exhale_time = t

        if t - self.last_cycle_start < self.spike_duration:
            u_in = self.spike_value
        else:
            u_in = self.u_in

        u_out = self.u_out

        if state > 45:  # safety trigger until next phase
            u_in = self.u_in = self.spike_value = 0
            u_out = self.u_out = 1

        return (u_in, u_out)
