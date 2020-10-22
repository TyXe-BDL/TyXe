from .local_reparameterization_messenger import LocalReparameterizationMessenger


__all__ = [
    "local_reparameterization",
]


# automate the following as in pyro.poutine.handlers
def local_reparameterization(fn=None):
    msngr = LocalReparameterizationMessenger()
    return msngr(fn) if fn is not None else msngr
