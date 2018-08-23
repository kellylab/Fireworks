import os
from sqlalchemy import create_engine, Column, Float, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
import datetime
from Fireworks import Message
from Fireworks import database as db

"""
This module contains classes and functions for saving and loading data collected during experiments.
"""

# class Listener:
#
#     def __init__(self, name, columns, experiment = None):
#         self.name = name
#         self.columns = columns # Should be a list of column names. If None, the listener will save all columns.
#         if experiment:
#             self.attach(experiment)
#         else:
#             self.experiment = None
#
#     def attach(self, experiment):
#         """
#         Attaches listener to experiment for i/o operations.
#         """
#         self.experiment = experiment
#         experiment.attach(self)
#
#     def save(self, batch):
#         """
#         Triggers attached
#         """
#         if not self.experiment:
#             raise AttributeError("This listener is not attached to an experiment, hence saving is not possible.")
#         batch = filter_columns(batch, self.columns)
#         self.experiment.save(self.name, batch)
#
#     def load(self):
#         """
#         Triggers attached
#         """
#         if not self.experiment:
#             raise AttributeError("This listener is not attached to an experiment, hence saving is not possible.")
#         return self.experiment.load(self.name)

# class IgniteListener(Listener):
#     """
#     Listener that can be attached to an Ignite Engine to log results.
#     """
#
#     def attach(self, target):
#         """
#         Attaches to target, which can be an engine or an experiment object.
#         Note that an IgniteListener must be attached to both an engine and an experiment in order to save data.
#         """
#         pass

metadata_columns = [
    Column('name', String),
    Column('iteration', Integer),
    Column('description', String),
    Column('timestamp', DateTime),
]
metadata_table = db.create_table('metadata', columns=metadata_columns)

def load_experiment(experiment_path):
    """
    Returns an experiment object corresponding to the database in the given path.
    """
    pass

class Experiment:
    # NOTE: For now, we assume that the underlying database is sqlite on local disk
    def __init__(self, experiment_name, db_path, description=None):
        self.listeners = {}
        self.db_path = db_path
        self.name = experiment_name
        self.description = description or ''
        self.timestamp = datetime.datetime.now()
        self.create_dir()
        self.init_metadata()
        self.engines = {}
        # Create/open save directory
        # if not os.path.exists(save_dir):
        #     try:
        #         os.makedirs(save_dir)
        #     except Error as e:
        #         print("Could not create save directory {save_dir}. Please check permissions and try again: {error}".format(save_dir=save_dir, error=e))
        # self.save_dir = save_dir

    def create_dir(self):
        """
        Creates a folder in db_path directory corresponding to this Experiment.
        """
        dirs = os.listdir(self.db_path)
        previous_experiments = [d for d in dirs if d.startswith(self.name)]
        self.iteration = len(previous_experiments)
        os.makedirs(os.path.join(self.db_path, "{name}_{iteration}".format(name=self.name, iteration=self.iteration))) # TODO: Upgrade to 3.6 and use f-strings
        self.save_path = "{name}_{iteration}".format(name=self.name, iteration=self.iteration)
        self.engine = create_engine("sqlite:///{save_path}".format(save_path=os.path.join(self.save_path,'metadata.sqlite')))

    def init_metadata(self):
        self.metadata = db.TableSource(metadata_table, self.engine, columns=['name', 'iteration', 'description', 'timestamp'])
        self.metadata.insert(Message({'name': [self.name], 'iteration': [self.iteration], 'description': [self.description], 'timestamp': [self.timestamp]}))
        self.metadata.commit()

    # def attach(self, listener):
    #     # Attach a listener object
    #     self.listeners[listener.name] = listener
    #     # Create table if not exists
    #     if not os.path.exists(os.path.join(save_dir, listener.name)):
    #         # self.files[listener.name] = open()
    #         pass

    def create_engine(self, name):
        """
        Creates an engine corresponding to a database with the given name. In particular, this creates a file called {name}.sqlite
        in this experiment's save directory, and makes an engine to connect to it.
        """
        if name in self.engines:
            raise ValueError("Engine with name {name} already exists in this experiment.".format(name=name))
        self.engines[name] = create_engine("sqlite:///{filename}".format(filename=os.path.join(self.save_path, name+'.sqlite')))
        return self.engines[name]

    def get_session(self, name):
        """
        Creates an SQLalchemy session corresponding to the engine with the given name that can be used to interact with the database.
        """
        if name in self.engines:
            engine = self.engines[name]
        else: # QUESTION: Should this raise an error or autocreate a new engine?
            engine = self.create_engine(name)
        Session = sessionmaker(bind=engine)
        session = Session()
        return session

def filter_columns(message, columns = None):
    """
    Returns only the given columns of message or everything if columns is None.
    If tensor columns are requested, they are converted to ndarray first.
    """
    return message # TODO: Implement
