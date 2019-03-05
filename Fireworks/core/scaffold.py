from .message import Message
import os

class Scaffold:

    def __init__(self, to_attach=None):

        self.attached = to_attach or {}

    def attach(self, name, obj):

        if not hasattr(obj, 'get_state'):
            raise AttributeError(
            "Objects attached to a Scaffold must have a 'get_state' method that returns a serialized representation of the object."
            )
        self.attached[name] = obj

    def serialize(self):

        return {name: obj.get_state() for name, obj in self.attached.items()}

    def save(self, path, method='json', **kwargs):
        """
        Saves serialized representation of all objects linked to Scaffold.
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
        """
        for filename in os.listdir(path):
            # state = Message.from_objects((os.path.join(path, filename))
            state = Message.read(method=method, path=os.path.join(path, filename)).to_dict()
            state = {key: value[0] for key, value in state.items()}
            key = '.'.join(filename.split('.')[:-1])
            if key in self.attached:

                self.attached[key].set_state({'external': {}, 'internal': state}, reset=reset)
