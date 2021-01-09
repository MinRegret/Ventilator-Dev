from vent.controllers.core import Controller


# generic PID controller
class PID(Controller):
    def __init__(self, K=[3, 4, 0], RC=0.5, decay=False, **kwargs):
        # controller coeffs
        self.K_P, self.K_I, self.K_D = K

        # controller states
        self.P, self.I, self.D = 0, 0, 0

        self.RC = RC
        self.decay = decay


    def compute_action(self, state, t):
        err = self.waveform.at(t) - state

        dt = self.dt(t)

        decay = dt / (dt + self.RC)

        self.I += decay * (err - self.I)
        self.D += decay * (err - self.P - self.D)
        self.P = err

        u_in = self.K_P * self.P + self.K_I * self.I + self.K_D * self.D
        u_in = max(0, min(u_in, 100))

        if self.decay:
            decay_u_in = self.waveform.decay(t)
            if decay_u_in is not None:
                u_in = decay_u_in

        return (u_in, self.u_out(t))
