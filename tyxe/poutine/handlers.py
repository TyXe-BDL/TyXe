from .reparameterization_messengers import LocalReparameterizationMessenger, FlipoutMessenger
from .selective_messengers import SelectiveMaskMessenger, SelectiveScaleMessenger


__all__ = [
    "local_reparameterization",
    "flipout",
    "selective_mask",
    "selective_scale"
]


# automate the following as in pyro.poutine.handlers
def local_reparameterization(fn=None, reparameterizable_functions=None):
    msngr = LocalReparameterizationMessenger(reparameterizable_functions=reparameterizable_functions)
    return msngr(fn) if fn is not None else msngr


def flipout(fn=None, reparameterizable_functions=None):
    msngr = FlipoutMessenger(reparameterizable_functions=reparameterizable_functions)
    return msngr(fn) if fn is not None else msngr


def selective_mask(fn=None, mask=None, **block_kwargs):
    msngr = SelectiveMaskMessenger(mask, **block_kwargs)
    return msngr(fn) if fn is not None else msngr


def selective_scale(fn=None, scale=1.0, **block_kwargs):
    msngr = SelectiveScaleMessenger(scale, **block_kwargs)
    return msngr(fn) if fn is not None else msngr
