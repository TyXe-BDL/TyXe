from .clamp_param_messenger import ClampParamMessenger
from .local_reparameterization_messenger import LocalReparameterizationMessenger
from .selective_messengers import SelectiveClampParamMessenger, SelectiveScaleMessenger


__all__ = [
    "clamp_param",
    "local_reparameterization",
    "selective_clamp_param",
    "selective_scale"
]


# automate the following as in pyro.poutine.handlers
def local_reparameterization(fn=None):
    msngr = LocalReparameterizationMessenger()
    return msngr(fn) if fn is not None else msngr


def clamp_param(fn=None, min_val=None, max_val=None):
    msngr = ClampParamMessenger(min_val, max_val)
    return msngr(fn) if fn is not None else msngr


def selective_clamp_param(fn=None, min_val=None, max_val=None, **kwargs):
    msngr = SelectiveClampParamMessenger(min_val, max_val, **kwargs)
    return msngr(fn) if fn is not None else msngr


def selective_scale(fn=None, scale=1., **kwargs):
    msngr = SelectiveScaleMessenger(scale, **kwargs)
    return msngr(fn) if fn is not None else msngr
