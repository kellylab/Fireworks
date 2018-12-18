

class Model:
    """
    Represents a statistical model which has a set of parameters, and a
    means for converting inputs into outputs. The model functions like a Pipe
    with respect to the input/output stream, and it functions like a Junction
    with respect to the parameterization. Parameters can be provided via multiple
    different sources in this way, providing flexibility in model configuration.
    Models can also provide parameters for other Models, enabling one to create
    complex graphs of Models that can be trained simultaneously or individually.
    """
