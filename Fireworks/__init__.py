import os
base_path = os.getcwd()
test_dir =  os.path.join(base_path, 'test')

from .message import cat, merge
from .pipeline import Pipe
from .junction import Junction
from .message import Message
