class EndHyperparameterOptimization(RuntimeError):
    """
    This exception can be raised to signal a factory to stop looping.
    """
    pass

class ParameterizationError(KeyError):
    """
    This exception is raised to indicate that a Model is missing required parameters that it needs to function.
    """
    pass 
