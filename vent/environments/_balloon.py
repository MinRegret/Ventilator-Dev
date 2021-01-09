import numpy as np

from lung.environments.core import Environment


class Balloon(Environment):
    """
    Physics simulator for inflating balloon with a PEEP valve
    For math, see https://en.wikipedia.org/wiki/Two-balloon_experiment
    """

    def __init__(self, peep_valve):
        # Hard parameters for the simulation
        self.max_volume = 6  # Liters  - 6?
        self.min_volume = 1.5  # Liters - baloon starts slightly inflated.
        self.PC = 40  # Proportionality constant that relates pressure to cm-H2O
        self.P0 = 0  # Baseline/Minimum pressure.
        self.leak = True

        self.fio2 = 60

        # Dynamical parameters - these are the initial conditions
        self.current_flow = 0  # in unit  liters/sec

        self.set_Qin = 0  # set flow of a prop valve on inspiratory side      -- liters/second
        self.Qin = 0  # exact flow of a prop valve on inspiratory side    -- liters/second

        self.set_Qout = 0  # 0|max - setting of an solenoid on expiratory side -- liters/second
        self.Qout = 0  # exact flow through the solenoid

        self.current_pressure = 0  # in unit  cm-H2O
        self.r_real = (3 * self.min_volume / (4 * np.pi)) ** (1 / 3)  # size of the lung
        self.current_volume = self.min_volume  # in unit  liters
        self.peep_valve = peep_valve

    def get_pressure(self):
        return self.current_pressure

    def get_volume(self):
        return self.current_volume

    def step(self, u_in, u_out, t):  # Performs an update of duration dt [seconds]
        dt = self.dt(t)

        # set_flow_in
        self.set_Qin = u_in

        Qin_clip = np.min(
            [Qin, 2]
        )  # Flows have to be positive, and reasonable. Nothing here is faster that 2 l/s
        Qin_clip = np.max([Qin_clip, 0])
        self.Qin = Qin_clip  # Assume the set-value is also the true value for prop

        # set_flow_out
        self.set_Qout = u_out

        Qout_clip = np.min(
            [Qout, 2]
        )  # Flows have to be positive, and reasonable. Nothing here is faster that 2 l/s
        Qout_clip = np.max([Qout_clip, 0])
        difference_pressure = self.current_pressure - 0  # Convention: outside is "0"
        conductance = (
            0.05 * Qout_clip
        )  # This should be in the range of ~1 liter/s for typical max pressure differences
        if self.current_pressure > self.peep_valve:  # Action of the PEEP valve
            self.Qout = difference_pressure * conductance  # Target for flow out
        else:
            self.Qout = 0

        if dt < 1:
            # self.current_flow = self.Qin - self.Qout     # But no update should take longer than that
            s = dt / (0.050 * np.abs(self.current_flow) + dt)
            self.current_flow = self.current_flow + s * ((self.Qin - self.Qout) - self.current_flow)

            self.current_volume += self.current_flow * dt

            # This is from the baloon equation, uses helper variable (the baloon radius)
            self.r_real = (3 * self.current_volume / (4 * np.pi)) ** (1 / 3)
            r0 = (3 * self.min_volume / (4 * np.pi)) ** (1 / 3)

            new_pressure = self.P0 + (self.PC / (r0 ** 2 * self.r_real)) * (
                1 - (r0 / self.r_real) ** 6
            )
            self.current_pressure = new_pressure

            # o2 fluctuations modelled as OUprocess
            self.fio2 = self.OUupdate(self.fio2, dt=dt, mu=60, sigma=5, tau=1)
        else:
            self._reset()
            print(self.current_pressure)

    def OUupdate(self, variable, dt, mu, sigma, tau):
        """
        This is a simple function to produce an OU process.
        It is used as model for fluctuations in measurement variables.
        inputs:
        variable:   float     value at previous time step
        dt      :   timestep
        mu      :   mean
        sigma   :   noise amplitude
        tau     :   time scale
        returns:
        new_variable :  value of "variable" at next time step
        """
        dt = max(dt, 0.05)  # Make sure this doesn't go haywire if anything hangs. Max 50ms
        sigma_bis = sigma * np.sqrt(2.0 / tau)
        sqrtdt = np.sqrt(dt)
        new_variable = (
            variable + dt * (-(variable - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()
        )
        return new_variable

    def reset(self):
        """ resets the ballon to standard parameters. """
        self.set_Qin = 0
        self.Qin = 0
        self.set_Qout = 0
        self.Qout = 0
        self.current_pressure = 0
        self.r_real = (3 * self.min_volume / (4 * np.pi)) ** (1 / 3)
        self.current_volume = self.min_volume
        self.time = float("inf")

