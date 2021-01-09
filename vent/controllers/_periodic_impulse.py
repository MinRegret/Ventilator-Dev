import numpy as np
from vent.controllers.core import Controller


class PeriodicImpulse(Controller):
    def __init__(self, period=2, duration=0.1, hold=0.5, amplitude=20, hold_amplitude=0):
        self.period = period
        self.duration = duration
        self.hold = hold
        self.amplitude = amplitude
        self.hold_amplitude = hold_amplitude

    def compute_action(self, state, t):
        phase = t % self.period

        if phase < self.duration:
            u_in = self.amplitude
            u_out = 0
        elif phase < self.hold:
            u_in = self.hold_amplitude
            u_out = 0
        else:
            u_in = 0
            u_out = 1

        return (u_in, u_out)
