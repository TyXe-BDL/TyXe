import functools

from pyro.poutine.messenger import Messenger


__all__ = [
    "ClampParamMessenger"
]


def _clamp_returnval(fn, min_val, max_val):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        return fn(*args, **kwargs).clamp(min_val, max_val)
    return _fn


class ClampParamMessenger(Messenger):

    def __init__(self, min_val=None, max_val=None, **kwargs):
        super(ClampParamMessenger, self).__init__(**kwargs)
        if min_val is None and max_val is None:
            raise ValueError("Either min_val or max_val must be not None")
        if min_val is not None and max_val is not None and min_val >= max_val:
            raise ValueError("min_val >= max_val")
        self.min_val = min_val
        self.max_val = max_val

    def _pyro_param(self, msg):
        # for some reason just doing the following as a postprocessing does no work:
        # msg["value"] = msg["value"].clamp(self.min_val, self.max_val)
        # but simply wrapping the function to clamp the return value seems to do the job
        msg["fn"] = _clamp_returnval(msg["fn"], self.min_val, self.max_val)
        return None

