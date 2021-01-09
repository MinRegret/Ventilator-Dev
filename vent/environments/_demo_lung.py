import numpy as np

from lung.environments.core import Environment

# balloon physics lung ported from Cohen lab's repository
# Sources:
# https://github.com/CohenLabPrinceton/Ventilator-Dev/blob/master/sandbox/HOWTO_RunController.ipynb
# https://github.com/CohenLabPrinceton/Ventilator-Dev/blob/master/vent/controller/control_module.py
class DemoLung(Environment):
    def __init__(self, leak=False, peep_valve=5, PC=40, RP=1, **kwargs):
        # dynamics hyperparameters
        self.max_volume = 6  ### seems unused from code
        self.min_volume = 1.5
        self.PC = PC
        self.RP = RP
        self.P0 = 0
        self.leak = leak
        self.peep_valve = peep_valve

        # reset states
        self.reset()

    def reset(self):
        # keep volume as the only free parameter
        self.volume = self.min_volume
        self.compute_aux_states()

        self.pips, self.peeps = [], []
        self.volumes, self.pressures = [], []

    def compute_aux_states(self):
        # compute all other state vars, which are just functions of volume
        r = (3 * self.volume / (4 * np.pi)) ** (1 / 3)
        r0 = (3 * self.min_volume / (4 * np.pi)) ** (1 / 3)
        self.pressure = self.P0 + self.PC * (1 - (r0 / r) ** 6) / (r0 ** 2 * r)

    def step(self, u_in, u_out, t):
        self.pips.append(u_in)
        self.peeps.append(u_out)

        dt = self.dt

        # 2-dimensional action per timestep: PIP/PEEP voltages

        def PropValve(x):  # copied from Controller.__SimulatedPropValve
            y = 3 * x
            flow_new = 1.0 * (np.tanh(0.03 * (y - 130)) + 1)
            # if y > 170:
                # flow_new = 1.72
            # if y < 0:
                # flow_new = 0
            return np.clip(flow_new, 0.0, 1.72)

        def Solenoid(x):  # copied from Controller.__SimulatedSolenoid
            if x > 0:
                return 1
            else:
                return 0

        flow = np.clip(PropValve(u_in), 0, 2) * self.RP
        if self.pressure > self.peep_valve:
            # print(Solenoid(u_out))
            flow -= np.clip(Solenoid(u_out), 0, 2) * 0.05 * self.pressure

        # update by flow rate
        self.volume += flow * dt

        # simulate leakage
        if self.leak:
            RC = 5
            s = dt / (RC + dt)
            self.volume += s * (self.min_volume - self.volume)

        # compute and record state
        self.compute_aux_states()
        self.volumes.append(self.volume)
        self.pressures.append(self.pressure)

