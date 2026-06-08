""" utility functions """
import scipy.io
import numpy as np

def mat_to_dict(obj):
    """Recursively convert scipy mat_struct to nested dicts."""
    if isinstance(obj, scipy.io.matlab.mat_struct):
        return {field: mat_to_dict(getattr(obj, field))
                for field in obj._fieldnames}
    elif isinstance(obj, np.ndarray) and obj.dtype.names:
        # structured array
        return {name: mat_to_dict(obj[name]) for name in obj.dtype.names}
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        # array of structs
        return [mat_to_dict(item) for item in obj.flat]
    else:
        return obj  # base case: numbers, strings, plain np arrays