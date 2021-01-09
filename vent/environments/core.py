import os
import time
import datetime
import tqdm
import dill as pickle
import torch
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

import logging

from vent.core import VentObj
from vent.utils.analyzer import Analyzer
from vent.utils.munger import Munger

from vent.controllers._impulse import Impulse
from vent.controllers._predestined import Predestined

EnvironmentRegistry = []

logger = logging.getLogger(__name__)


class Environment(VentObj):
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.__setattr__("name", cls.__name__)
        obj.__setattr__("pressure", 0)
        obj.__setattr__("flow", 0)
        obj.__setattr__("tt", 0)
        obj.__setattr__("dt", 0.03)

        for kw, arg in kwargs.items():
            obj.__setattr__(kw, arg)

        return obj

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if cls.__name__ not in [env.__name__ for env in EnvironmentRegistry]:
            EnvironmentRegistry.append(cls)

    # Define overrideable properties for run function. We could omit the
    # underscore, but since run is a function we won't touch and Environment
    # writers will likely want to use self.pressure as a float, we do this way
    @property
    def _pressure(self):
        return self.pressure

    @property
    def _flow(self):
        return self.flow

    @property
    def time(self):
        return self.tt * self.dt

    @property
    def is_real(self):
        return False

    def reset(self):
        self.pressure = 0
        self.flow = 0
        self.tt = 0

    def step(self, u_in, u_out, t):
        raise NotImplementedError()

    def run_cleanup(self):
        pass

    def run(
        self,
        controller,
        R=20,
        C=20,
        PEEP=5,
        T=3000,
        abort=50,
        sleep=None,
        directory=None,
        device=None,
        dt=None,
        use_tqdm=True,
    ):
        if dt is not None:
            self.dt = dt

        result = {
            "controller": controller,
            "R": R,
            "C": C,
            "PEEP": PEEP,
            "dt": self.dt,
            "T": T,
            "abort": abort,
            "sleep": sleep,
            "directory": directory,
        }

        try:
            self.reset()

            self.tt = 0
            tt = range(T)

            timestamps = np.zeros(T)
            pressures = np.zeros(T)
            flows = np.zeros(T)
            u_ins = np.zeros(T)
            u_outs = np.zeros(T)

            begin = datetime.datetime.now().timestamp() if self.is_real else 0

            pressure = 0

            if use_tqdm:
                tt = tqdm.tqdm(tt, leave=False)

            for i, t in enumerate(tt):
                pressure = self._pressure
                flow = self._flow

                if device is not None:
                    pressure = torch.tensor(pressure, device=device)

                timestamp = self.time - begin
                u_in, u_out = controller.feed(pressure, timestamp)

                if device is not None:
                    u_in = u_in.item()
                    u_out = u_out.item()

                self.step(u_in, u_out, timestamp)

                timestamps[i] = timestamp
                pressures[i] = pressure
                flows[i] = flow
                u_ins[i] = u_in
                u_outs[i] = u_out

                if pressure > abort:
                    print(f"Pressure of {pressure} > {abort}; quitting")
                    break

                if self.is_real:
                    time.sleep(self.dt)

                self.tt += 1
        finally:
            self.run_cleanup()

        timeseries = {
            "timestamp": np.array(timestamps),
            "pressure": np.array(pressures),
            "flow": np.array(flows),
            "target": controller.waveform.at(timestamps),
            "u_in": np.array(u_ins),
            "u_out": np.array(u_outs),
        }

        for key, val in timeseries.items():
            timeseries[key] = val[:self.tt + 1]

        result["timeseries"] = timeseries

        if directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            pickle.dump(result, open(f"{directory}/{timestamp}.pkl", "wb"))

        return Analyzer(result)

    def plot_impulse_responses(
        self,
        impulses=np.arange(0, 101, 10),
        zero=5,
        start=0.5,
        end=0.65,
        ylim=100,
        dt=0.03,
        T=100,
        use_tqdm=False,
        abort=70,
        **kwargs,
    ):
        analyzers = []

        for impulse in impulses:
            ir = Impulse(impulse, start, end)

            analyzers.append(
                self.run(ir, dt=dt, T=T, PEEP=zero, use_tqdm=use_tqdm, abort=abort, **kwargs)
            )

        loss = 0
        colors = plt.cm.winter(np.linspace(0, 1, len(analyzers)))
        for i, analyzer in enumerate(analyzers):
            plt.plot(analyzer.tt, analyzer.pressure, color=colors[-i - 1])
            if impulses[0] == 0 and i == 0:
                loss += np.abs(analyzer.pressure - zero).mean()

        print(f"MAE for zero response: {loss}")

        plt.ylim(0, ylim)
        plt.show()

        return analyzers

    def plot_open_loop_residuals(self, munger_path, key="test", abort=70, use_tqdm=False, **kwargs):
        munger = Munger.load(munger_path)
        analyzers = []

        for test_trajectory in munger.splits["test"]:
            test_u_ins, test_pressures = test_trajectory
            controller = Predestined(test_u_ins, np.zeros_like(test_u_ins))

            analyzer = self.run(
                controller, T=len(test_u_ins), abort=abort, use_tqdm=use_tqdm, **kwargs
            )
            analyzers.append(analyzer)

            plt.plot(analyzer.tt, test_pressures - analyzer.pressure, c="k", alpha=0.1)

        return analyzers

    def plot_teacher_forcing_residuals(
        self, munger_path, key="test", abort=70, use_tqdm=False, **kwargs
    ):
        munger = Munger.load(munger_path)

        for test_trajectory in munger.splits["test"]:
            test_u_ins, test_pressures = test_trajectory

            T = len(test_u_ins)

            for k in range(T):  # feed k steps of pressure history
                self.reset()
                self.time_since_exhalation = k
                if k > 0:
                    self.u_history = list(self.u_scaler.transform([test_u_ins[:k]])[0])
                    self.p_history = list(self.p_scaler.transform([test_pressures[:k]])[0])
                    self.pressure = test_pressures[k - 1]

                pressures = []
                u_ins = []

                for t in range(k, T):  # feed the rest from generation
                    p = self.pressure
                    pressures.append(p)
                    u_in = test_u_ins[t]
                    self.step(u_in, 0, t)
                    u_ins.append(u_in)

                plt.plot(test_pressures[k:] - pressures, c="k", alpha=0.05)
