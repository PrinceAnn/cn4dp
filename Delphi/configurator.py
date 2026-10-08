"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval


def _coerce_value(val, key):
    if key not in globals():
        raise ValueError(f"Unknown config key: {key}")
    try:
        attempt = literal_eval(val)
    except (SyntaxError, ValueError):
        attempt = val
    # only enforce type if original type exists and is not None
    if key in globals() and globals()[key] is not None:
        assert type(attempt) == type(globals()[key])
    return attempt


args = sys.argv[1:]
i = 0
while i < len(args):
    arg = args[i]
    if '=' not in arg:
        if arg.startswith('--'):
            # support "--key value" form
            key = arg[2:]
            if i + 1 >= len(args):
                raise ValueError(f"Missing value for argument: {arg}")
            val = args[i + 1]
            attempt = _coerce_value(val, key)
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
            i += 2
        else:
            # assume it's the name of a config file
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read())
            i += 1
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=', 1)
        key = key[2:]
        attempt = _coerce_value(val, key)
        print(f"Overriding: {key} = {attempt}")
        globals()[key] = attempt
        i += 1
