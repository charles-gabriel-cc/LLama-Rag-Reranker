import inspect

def get_valid_kwargs(cls, kwargs):
    """
    Extract args from a class
    """
    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters.keys()) - {'self'}
    return {k: v for k, v in kwargs.items() if k in valid_keys}