import datetime
import os
import sys
import time
import pickle
from jupyterplot import ProgressPlot

import vent
from vent import prefs
from vent.common.message import ControlSetting
from vent.common.values import CONTROL, ValueName
from vent.coordinator.coordinator import get_coordinator


class JupyterGUI:
    def __init__(self, **kwargs):
        if "directory" not in kwargs:
            kwargs["directory"] = os.path.join(
                os.path.expanduser("~"),
                "vent/logs/hazan",
                datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            )

        print("Logging to {}".format(kwargs["directory"]))

        if not os.path.exists(kwargs["directory"]):
            os.makedirs(kwargs["directory"])

        pickle.dump(kwargs, open(os.path.join(kwargs["directory"], "kwargs.pkl"), "wb"))
        print(kwargs)

        self.coordinator = get_coordinator(single_process=True, **kwargs)

        for key, val in CONTROL.items():
            value = val.default
            if key == ValueName.PIP_TIME:
                value = kwargs["pip_time"]
            elif key == ValueName.INSPIRATION_TIME_SEC:
                value = kwargs["inspiration_time"]
            elif key == ValueName.PIP:
                value = kwargs["pip"]
            elif key == ValueName.PEEP:
                value = kwargs["peep"]
            elif key == ValueName.PEEP_TIME:
                value = kwargs["peep_time"]
            elif key == ValueName.BREATHS_PER_MINUTE:
                value = kwargs["bpm"]

            control_setting = ControlSetting(
                name=key,
                value=value,
                min_value=val.safe_range[0],
                max_value=val.safe_range[1],
                timestamp=time.time(),
            )
            self.coordinator.set_control(control_setting)

    def start(self):
        self.coordinator.start()

    def stop(self):
        self.coordinator.stop()

    def show(self, update_frequency=0.1):
        pp = ProgressPlot(plot_names=["pressure", "u_in", "u_out"], line_names=["val"])

        try:
            while True:
                vals = self.coordinator.get_sensors()
                pp.update([[vals["PRESSURE"]], [vals["FLOWOUT"]], [vals["FIO2"]]])
                time.sleep(update_frequency)
        except KeyboardInterrupt:
            pp.finalize()
            sys.exit(1)
