from .reparameterization_messengers import LocalReparameterizationMessenger, FlipoutMessenger


__all__ = [
    "local_reparameterization",
    "flipout"
]


# automate the following as in pyro.poutine.handlers
def local_reparameterization(fn=None, reparameterizable_functions=None):
    msngr = LocalReparameterizationMessenger(reparameterizable_functions=reparameterizable_functions)
    return msngr(fn) if fn is not None else msngr


def flipout(fn=None, reparameterizable_functions=None):
    msngr = FlipoutMessenger(reparameterizable_functions=reparameterizable_functions)
    return msngr(fn) if fn is not None else msngr
