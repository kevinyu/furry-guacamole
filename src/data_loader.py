import glob
import os
import re

import h5py
import numpy as np
import pandas as pd

import config


class SessionDataLoader(object):
    """Load data from 'zebra finch neural response to semantic call categories' dataset

    The dataset involves neural population responses to several renditions of
    conspecific calls, over ~10 trials per call. The calls are labeled by their
    semantic call category. The data comes in the form of spike arrival times, and
    t=0.0 is aligned to the stimulus onset.
    """

    def __init__(self, bird_name, site_num):
        self.bird_dir = os.path.join(config.DATA_DIR, bird_name)
        self._site_prefix = "Site{}".format(site_num)

        # Regex for locating files corresponding to single unit spike trains
        self._single_unit_re = re.compile(os.path.join(self.bird_dir,
            "{}_.+_e"
            "(?P<electrode_id>\d+)"
            "_s\d+_ss"
            "(?P<unit_id>\d+).h5".format(self._site_prefix)))

        # Regex for locating files corresponding to multi unit spike trains (single electrode)
        self._multi_unit_re = re.compile(os.path.join(self.bird_dir, 
            "{}_.+_e"
            "(?P<electrode_id>\d+)"
            "_s\d+.h5".format(self._site_prefix)))

        self._call = None
        for key in list(self.single_units.values())[0].keys():
            if key.startswith("Call"):
                self._call = key
                break

    def _match_unit_files(self, regex):
        """Build a dictionary associating each (electrode, unit) pair with a h5py file

        Find all files that are either single-unit or multi-unit and load them as h5py.
        Won't actually load anything into memory yet.

        Returns:
        unit_data (dict):
            (electrode_id, unit_num) -> h5py.File
        """
        unit_data = {}
        for filename in glob.glob(os.path.join(self.bird_dir, "{}*.h5".format(self._site_prefix))):
            match = regex.match(filename)
            if not match:
                continue
            electrode_unit_tuple = tuple(map(int, match.groups()))  # (electrode_id, unit_num)
            unit_data[electrode_unit_tuple] = h5py.File(
                    os.path.join(self.bird_dir, match.group()), "r", libver="latest")

        return unit_data

    @property
    def _stim_id_to_call_type_and_stim_type(self):
        """Dict mapping stim id to call type and stim type

        Keys are stim id strings ("100", "101", ...)
        Values are tuples
            first element is call type ("Ag", "DC", "Di", etc)
            second element is stim type ("call", "song", "mlnoise")
        """
        return dict(
                (stim_id, (stim_data.attrs.get("callid"), stim_data.attrs.get("stim_type")))
                for stim_id, stim_data
                in self.single_units.values()[0][self._call].items())

    @property
    def single_units(self):
        """Return all h5py.File's for spike sorted units for this bird and site"""
        return self._match_unit_files(self._single_unit_re)

    def load_table(self):
        """Load up a Pandas dataframe for the recording session

        Includes every response for every unit

        Columns
            unit (tuple): (electrode_id, unit_id) pair identifying the unit, e.g. (29, 1)
            stim (str): string identifying the stimulus (the unique recording), e.g. "121"
            call_type (str): string identifying the call (semantic) category, e.g. "Ag"
            stim_type (str): string identifying the stimulus (broader) category, e.g. "song" or "call"
            trial (str): trial id number for the recording, e.g. "1"
        """
        units = []
        stim_ids = []
        call_types = []
        stim_types = []
        trial_ids = []
        spike_times = []

        for stim_id, (call_type, stim_type) in self._stim_id_to_call_type_and_stim_type.items():
            for (electrode_id, unit_id), unit_file in self.single_units.items():
                for trial_num, trial_file in unit_file[self._call][stim_id].items():
                    units.append((electrode_id, unit_id))
                    stim_ids.append(stim_id)
                    call_types.append(call_type or "None")
                    stim_types.append(stim_type)
                    trial_ids.append(trial_num)

                    if trial_file["spike_times"][0] == -999:
                        spike_times.append(np.array([]))
                    else:
                        spike_times.append(trial_file["spike_times"][:].flatten())

        return pd.DataFrame({
            "unit": pd.Series(units),
            "stim": pd.Series(stim_ids),
            "call_type": pd.Series(call_types),
            "stim_type": pd.Series(stim_types),
            "trial": pd.Series(trial_ids),
            "spike_times": pd.Series(spike_times)
        })

