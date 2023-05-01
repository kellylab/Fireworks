from .message import Message
import os

class Scaffold:
    """
    A Scaffold can keep track of the internal state of objects in a pipeline. This can be used to save and load the entire state of a
    pipeline, allowing one to pause and resume a project, take snapshots, and log the internal state of components as an experiment proceeds.
    The current implementation of Scaffold is very simple; you attach objects to the Scaffold while providing a name that serves as an
    identifier for that object. You can then call the serialize method to get a dictionary of the current states of all attached objects,
    the save method to save those serialized states to a given folder, or load to update the states of attached objects using data in
    a provided directory.

    ::
    
        scaffold = Scaffold()
        # Attach components as desired
        scaffold.attach('a', A)
        scaffold.attach('b', B)
        .
        .
        .
        # This will save the current state of attached objects in folder 'save_directory'. The filenames will be based on the identifiers
        # eg. 'a.json', 'b.json', etc.
        scaffold.save(path='save_directory', method='json')
        # This will read files in folder 'save_directory2' and set the state of attached objects based on identifiers.
        # eg. the components attached with identifier 'a' will load from 'a.json' and so on. Note that the provided method must be
        # consistent with the filetypes in the save directory.
        scaffold.load(path='save_directory2', method='json')

        # This will produce a dictionary of identifiers to state object serializations (see Component_Map documentation for details).
        # You could use this dictionary to log state information however you want. For example, you could log the current weights of
        # neural network layers in your model for later plotting.

        state = scaffold.serialize()
    """

    def __init__(self, to_attach=None):
        """
        Args:
            to_attach: A dictionary of keys mapping to objects that you want the Scaffold to track. Note that each object must implement
                       a get_state method which returns a dictionary of the form {'external': {...}, 'internal': {...}}.
                       Pipes, Junctions, and Models satisfy this criteria.
        """
        self.attached = to_attach or {}

    def attach(self, name, obj):
        """
        Attaches an object to the Scaffold with a provided identifier. The Scaffold can then track the object's internal state, enabling one
        to access, save, and load the serialized states of all tracked objects at once.

        Args:
            name: The identifier for the object.
            obj: The object to attach. Note that each object must implement a get_state method which returns a dictionary of the
                 form {'external': {...}, 'internal': {...}}. Pipes, Junctions, and Models satisfy this criteria.
        """

        if not hasattr(obj, 'get_state'):
            raise AttributeError(
            "Objects attached to a Scaffold must have a 'get_state' method that returns a serialized representation of the object."
            )
        self.attached[name] = obj

    def serialize(self):
        """
        Returns a dictionary containing serialized representations of all objects tracked by the scaffold. See Component_Map documentation
        for more information on these serializations.

        Args:
            None

        Returns:
            state: A dict of the form {key: state}, where state is a dict of the form {'external': {...}, 'internal': {...}} corresponding to
            the internal and external state of objects tracked by the Scaffold. See Component_Map documentation for more information on state.
        """
        return {name: obj.get_state() for name, obj in self.attached.items()}

    def save(self, path, method='json', **kwargs):
        """
        Saves serialized representation of all objects linked to Scaffold using a desired method (json, csv, pickle, etc.)

        Args:
            path: The folder to save serializations to. This folder must exist and be writable by the program.
            method: The method for saving. This must be one of the methods support by the Message.to(...) method (see Message documentation)
                    , as state dicts are converted to Messages and saved using Message.to(...).
        """
        serialized = self.serialize()

        # Logic for saving
        for name, component_state in serialized.items():

            serialized_internal_state = Message.from_objects(component_state['internal'])
            component_path = os.path.join(path, '{0}.{1}'.format(name, method))
            kwargs['path'] = component_path
            saved_internal_state = serialized_internal_state.to(method=method, **kwargs)

    def load(self, path, method='json', reset=False):
        """
        Loads serialized representations of all objects linked to Scaffold using the given names in the given path.

        Args:
            path: The folder to load serializations from. This folder must exist and be readable by the program.
            method: The method for loading. This must be one of the methods support by the Message.load(...) method (see Message
                    documentation), as that method is used to read the files.
                    Note that load will only look for files corresponding to the provided method that also have the correspponding suffix
                    (eg. json filenames must end with '.json', pickles files with '.pickle', etc.). So if you have files in the foler that
                    were not saved as the given method, or have different filename suffixes, they will be ignored.
        """
        for filename in os.listdir(path):
            # state = Message.from_objects((os.path.join(path, filename))
            state = Message.read(method=method, path=os.path.join(path, filename)).to_dict()
            # NOTE: The saving process adds an extra dimension to the values, which we squash out here.
            state = {key: value[0] for key, value in state.items()}
            key = '.'.join(filename.split('.')[:-1]) # Drop the file suffix for mapping filenames to identifiers.

            if key in self.attached: # Files in the folder that do not correspond to attached objects will be ignored.

                self.attached[key].set_state({'external': {}, 'internal': state}, reset=reset)
