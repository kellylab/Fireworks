import os
import sqlite3

"""
This module contains classes and functions for saving and loading data collected during experiments.
"""

class Listener:

    def __init__(self, name, columns, experiment = None):
        self.name = name
        self.columns = columns # Should be a list of column names. If None, the listener will save all columns.
        if experiment:
            self.attach(experiment)
        else:
            self.experiment = None

    def attach(self, experiment):
        """
        Attaches listener to experiment for i/o operations.
        """
        self.experiment = experiment
        experiment.attach(self)

    def save(self, batch):
        """
        Triggers attached
        """
        if not self.experiment:
            raise AttributeError("This listener is not attached to an experiment, hence saving is not possible.")
        batch = filter_columns(batch, self.columns)
        self.experiment.save(self.name, batch)

    def load(self):
        """
        Triggers attached
        """
        if not self.experiment:
            raise AttributeError("This listener is not attached to an experiment, hence saving is not possible.")
        return self.experiment.load(self.name)

class IgniteListener(Listener):
    """
    Listener that can be attached to an Ignite Engine to log results.
    """

    def attach(self, target):
        """
        Attaches to target, which can be an engine or an experiment object.
        Note that an IgniteListener must be attached to both an engine and an experiment in order to save data.
        """
        pass

class Experiment:

    def __init__(self, db_path):
        self.listeners = {}
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        # Create/open save directory
        # if not os.path.exists(save_dir):
        #     try:
        #         os.makedirs(save_dir)
        #     except Error as e:
        #         print("Could not create save directory {save_dir}. Please check permissions and try again: {error}".format(save_dir=save_dir, error=e))
        # self.save_dir = save_dir

    def attach(self, listener):
        # Attach a listener object
        self.listeners[listener.name] = listener
        # Create table if not exists
        if not os.path.exists(os.path.join(save_dir, listener.name)):
            # self.files[listener.name] = open()
            pass

    def save(self, key, batch):
        # Append to the correct table 
        pass

def filter_columns(message, columns = None):
    """
    Returns only the given columns of message or everything if columns is None.
    If tensor columns are requested, they are converted to ndarray first.
    """
    return message # TODO: Implement
