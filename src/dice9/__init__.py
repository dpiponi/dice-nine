import importlib
import dice9.config as config
import os

backend_name = os.environ.get("PL_PLATFORM")
if backend_name == 'tf':
    print("Selected tf")
    tf = importlib.import_module('dice9.backends.tensorflow_impl')
    config.sx = tf
    impl = importlib.import_module('dice9.implementation')
    for name in dir(impl):
      globals()[name] = getattr(impl, name)
elif backend_name == 'np':
    print("Selected np")
    np = importlib.import_module('dice9.backends.numpy_impl')
    config.sx = np
    impl = importlib.import_module('dice9.implementation')
    for name in dir(impl):
      globals()[name] = getattr(impl, name)
elif backend_name == 'torch':
    print("Selected torch")
    torch = importlib.import_module('dice9.backends.torch_impl')
    config.sx = torch
    impl = importlib.import_module('dice9.implementation')
    for name in dir(impl):
      globals()[name] = getattr(impl, name)
else:
    raise NameError(f"I don't recognize the back end `{backend_name}`. Set PL_PLATFORM to np, tf or torch.")

config.profile = False

def use(backend_name, verbose=False, profile=False):
    pass
#     import config
#     config.profile = profile
#     if backend_name == 'tf':
#         if verbose:
#           print("Selected tf")
#         tf = importlib.import_module('dice9.backends.tensorflow_impl')
#         config.sx = tf
#         impl = importlib.import_module('dice9.implementation')
#         for name in dir(impl):
#           globals()[name] = getattr(impl, name)
#     elif backend_name == 'np':
#         if verbose:
#           print("Selected np")
#         np = importlib.import_module('dice9.backends.numpy_impl')
#         config.sx = np
#         impl = importlib.import_module('dice9.implementation')
#         for name in dir(impl):
#           globals()[name] = getattr(impl, name)
#     elif backend_name == 'torch':
#         if verbose:
#           print("Selected torch")
#         torch = importlib.import_module('dice9.backends.torch_impl')
#         config.sx = torch
#         impl = importlib.import_module('dice9.implementation')
#         for name in dir(impl):
#           globals()[name] = getattr(impl, name)
#     else:
#         raise "bad backend"

def only_use_inside_dist(**kwargs):
   raise ValueError('Function only usable within dist')

constant = only_use_inside_dist
d = only_use_inside_dist
one_hot = only_use_inside_dist
argmin = only_use_inside_dist
