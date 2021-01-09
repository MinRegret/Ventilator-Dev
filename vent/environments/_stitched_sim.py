import numpy as np
import torch

from lung.environments.core import Environment

class StitchedSim(Environment):
    def __init__(self, inspiratory_model, u_window, p_window, u_scaler, p_scaler, peep=5., odd_indexing=True):
        super().__init__()

        self.inspiratory_model = inspiratory_model

        self.u_window = u_window
        self.p_window = p_window

        self.u_scaler = u_scaler
        self.p_scaler = p_scaler

        self.peep = peep

        self.odd_indexing, self.even_indexing = odd_indexing, 1 - odd_indexing

        # reset states
        self.reset()

    def reset(self):
        self.time_since_exhalation = 0

        self.u_history = []
        self.p_history = [] if self.odd_indexing else [self.p_scaler.transform([[self.peep]])[0,0]]

        self.pressure = self.peep
        self.hardcoded_pressure = self.peep  # for hardcoded exhalation dynamics

    def step(self, u_in, u_out, t):

        u_in_scaled = self.u_scaler.transform([[u_in]])[0,0]
        self.u_history.append(u_in_scaled)

        if u_out == 1:
            self.time_since_exhalation = 0
            self.u_history.clear()
            self.p_history.clear()

            if self.even_indexing:
                self.p_history.append(self.p_scaler.transform([[self.peep]])[0,0])

            self.hardcoded_pressure = self.peep
            self.pressure = self.hardcoded_pressure
        else:
            self.time_since_exhalation += 1
            t_key = str(self.time_since_exhalation)

            if t_key in self.inspiratory_model.boundary_dict:  # predict from boundary model
                features = np.concatenate([self.u_history, self.p_history])
                features = torch.tensor(features, dtype=torch.float)
                self.scaled_pressure = self.inspiratory_model.boundary_dict[t_key](features).item()
            else:   # predict from default model
                features = np.concatenate([self.u_history[-self.u_window:], self.p_history[-self.p_window:]])
                features = torch.tensor(features, dtype=torch.float)
                self.scaled_pressure = self.inspiratory_model.default_model(features).item()

            self.p_history.append(self.scaled_pressure)
            self.pressure = self.p_scaler.inverse_transform([[self.scaled_pressure]])[0,0]
