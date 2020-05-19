import time
from typing import List
import threading
import numpy as np
import copy
from collections import deque
import pdb
import vent.io as io
import logging
import tables as pytb

from vent.common.message import SensorValues, ControlSetting
from vent.common.values import CONTROL, ValueName


class DataSample(pytb.IsDescription):
    timestamp    = pytb.Float64Col()    # current time of the measurement
    pressure     = pytb.Float32Col()    # float  (single-precision)  -- because we'll have a whole bunch of these
    cycle_number = pytb.Int32Col()      # Breath Cycle No.


class ControlCommand(pytb.IsDescription):
    name      = pytb.StringCol(16)   # Control setting name
    min_value = pytb.Float64Col()    # double (double-precision)
    max_value = pytb.Float64Col()    # double (double-precision)
    timestamp = pytb.Float64Col()    # double (double-precision)


class DataLogger:
    """
    Class for logging numerical respiration data and control settings.
    
    Creates a hdf5 file with the general structure:
        / root
        |--- waveforms (group)
        |    |--- time | pressure_data | volume | Cycle No.    --- DISCUSS THIS
        |
        |--- controls (group)                                  --- DISCUSS THIS
        |    |--- (time, controllsignal)
        |

    Attributes:
        name (str): Subject ID
        file (str): Path to hdf5 file - usually `{prefs.DATADIR}/{self.name}.h5`
        current (dict): current task parameters. loaded from
            the 'current' :mod:`~tables.filenode` of the h5 file
        step (int): current step
        current_cycle (int): current breath cycle
    """

    def __init__(self, filename):
        self.file = filename    # Filename
        
    def open_logfile(self):
        """
        Opens the hdf5 file.
        """
        try:
            if not self.h5file.isopen:
                self.h5file = pytb.open_file(self.file, mode = "a")
            else:
                print("This should't happen. hdf5 file was open.")
        except:
            self.h5file = pytb.open_file(self.file, mode = "w")
    
        try:
            group = self.h5file.create_group("/", 'waveforms', 'Respiration waveforms')
            self.data_table    = self.h5file.create_table(group, 'readout', DataSample, "Breath Cycles")
            
            group = self.h5file.create_group("/", 'controls', 'Control signal history')
            self.control_table = self.h5file.create_table(group, 'readout', ControlCommand, "Control Commands")


        except:
            self.data_table = self.h5file.root.waveforms.readout
            self.control_table = self.h5file.root.controls.readout
            
    def close_logfile(self):
        """
        Flushes & closes the open hdf file.
        """
        self.data_table.flush()
        self.h5file.close()

        
    def store_waveform_data(self, sensor_values: SensorValues):
        """
        Appends a datapoint to the file.
        """

        #Make sure hdf5 file is open
        if not self.h5file.isopen:
            self.open_logfile()
        
        datapoint                 = self.data_table.row

        datapoint['timestamp'] = sensor_values.timestamp
        datapoint['pressure'] = sensor_values.PRESSURE
        datapoint['cycle_number'] = sensor_values.breath_count

        datapoint.append()


    
    def flush_logfile(self):
        #Should be done sporadically
         self.data_table.flush()
        
        
    def save_controls(self):
        """
        Saves a new controlsetting to the file.
        """
        
        self.open_logfile()
        
        datapoint                 = self.data_table.row
        datapoint['name']         = ValueName.PEEP
        datapoint['value']        = 3
        datapoint['min_value']    = 1
        datapoint['max_value']    = 4
        datapoint['timestamp']    = time.time()
        datapoint.append()
    
        self.close_logfile()
                
    def check_size(self):
        """
        make sure that the file's are not getting too large.
        """
        pass