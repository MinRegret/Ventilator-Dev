import os
from enum import Enum
import numpy as np
import dill as pickle

from lung.utils import BreathWaveform
from lung.core import LungObj


ControllerRegistry = []


class Phase(Enum):
    RAMP_UP = 1
    PIP = 2
    RAMP_DOWN = 3
    PEEP = 4


class Controller(LungObj):
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.__setattr__("name", cls.__name__)
        obj.__setattr__("time", float("inf"))
        obj.__setattr__("waveform", BreathWaveform())
        obj.__setattr__("timeseries", ["t", "state", "target", "u_in", "u_out"])
        obj.__setattr__("log_initialized", False)

        for kw, arg in kwargs.items():
            obj.__setattr__(kw, arg)

        return obj

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if cls.__name__ not in [ctrl.__name__ for ctrl in ControllerRegistry]:
            ControllerRegistry.append(cls)

    def compute_action(self, state, t):
        pass

    def __call__(self, *args, **kwargs):
        return self.feed(*args, **kwargs)

    def feed(self, state, t):
        u_in, u_out = self.compute_action(state, t)
        self.time = t
        if hasattr(self, "log_directory"):
            self.update_log(t, state, self.waveform.at(t), u_in, u_out)
        return u_in, u_out

    def u_out(self, t):
        phase = self.phase(t)
        u_out = np.zeros_like(phase)
        u_out[np.equal(phase, Phase.RAMP_DOWN.value)] = 1
        u_out[np.equal(phase, Phase.PEEP.value)] = 1

        return u_out

    def dt(self, t):
        dt = max(0, t - self.time)
        return dt

    def cycle_phase(self, t):
        return t % self.waveform.period

    def phase(self, t):
        return self.waveform.phase(t)

    def set_log_directory(self, directory):
        setattr(self, "log_directory", directory)

    def update_log(self, t, state, target, u_in, u_out):
        if not self.log_initialized:
            self.__setattr__(
                "log_file", open(os.path.join(self.log_directory, "controller_timeseries.csv"), "w")
            )

            cols = ",".join(self.timeseries)
            self.log_file.write(f"{cols}\n")
            self.log_initialized = True

        self.log_file.write(f"{t},{float(state)},{target},{float(u_in)},{u_out}\n")


class LinearForecaster:
    def __init__(self, history_length):
        self.history = np.zeros(history_length)
        self._update_lin_fit()

    def update(self, value):
        self.history[0] = value
        self.history = np.roll(self.history, -1)
        self._update_lin_fit()

    def predict(self, steps_ahead):
        return self.lin_fit(len(self.history) + steps_ahead)

    def _update_lin_fit(self):
        self.lin_fit = np.poly1d(np.polyfit(range(len(self.history)), self.history, 1))
