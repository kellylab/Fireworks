Installation
=================

Install from PyPi
-----------------------

Fireworks was written using Python 3.5 and has only been tested on Python 3.5+.
You can install Fireworks using pip/pip3:

::

    pip install fireworks-ml

It is recommended to do this from within a virtual environment.

Install from source
-----------------------

For development purposes, you can download the source repository and install the latest version directly.
There are additional libraries required for some of the tests. They are indicated in full-requirements.txt.

::

    git clone https://github.com/smk508/Fireworks.git
    cd Fireworks
    (optional) virtualenv fireworks / vf create fireworks / etc.
    pip install .
    pip install -r full-requirements.txt
