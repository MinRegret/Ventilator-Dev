import time

from vent.environments.core import Environment


class PhysicalLung(Environment):
    def __init__(self, hal=None, rest=3.0):
        if hal is None:
            from vent import Hal

            hal = Hal()
        self.hal = hal
        self.rest = rest

    @property
    def _pressure(self):
        return self.hal.pressure

    @property
    def time(self):
        return time.time()

    @property
    def is_real(self):
        return True

    def step(self, u_in, u_out, t):
        self.hal.setpoint_in = u_in
        self.hal.setpoint_ex = u_out

    def run_cleanup(self):
        self.hal.setpoint_in = 0
        self.hal.setpoint_ex = 1
        time.sleep(self.rest)
        self.hal.setpoint_ex = 0
