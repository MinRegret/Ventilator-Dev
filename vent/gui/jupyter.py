import time
from jupyterplot import ProgressPlot

import vent
from vent import prefs
from vent.common.message import ControlSetting
from vent.common.values import CONTROL, ValueName
from vent.coordinator.coordinator import get_coordinator


class JupyterGUI:
    def __init__(self):
        self.coordinator = get_coordinator()

        for key, val in CONTROL.items():
            control_setting = ControlSetting(name=key,
                                             value=val.default,
                                             min_value=val.safe_range[0],
                                             max_value=val.safe_range[1],
                                             timestamp=time.time())
            self.coordinator.set_control(control_setting)

    def start(self):
        self.coordinator.start()

    def show(self, update_frequency=0.1):
        pp = ProgressPlot()

        while True:
            vals = self.coordinator.get_sensors()
            pp.update(vals["PRESSURE"])
            time.sleep(update_frequency)
