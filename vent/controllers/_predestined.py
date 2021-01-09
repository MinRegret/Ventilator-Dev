from lung.controllers.core import Controller


class Predestined(Controller):
    def __init__(self, u_ins, u_outs):
        self.u_ins = u_ins
        self.u_outs = u_outs

        self.reset()

    def reset(self):
        self.t = 0

    def compute_action(self, state, t):
        u_in, u_out = self.u_ins[self.t], self.u_outs[self.t]
        self.t += 1

        return (u_in, u_out)
