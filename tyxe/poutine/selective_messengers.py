from pyro.poutine.block_messenger import _make_default_hide_fn
from pyro.poutine.scale_messenger import ScaleMessenger


from .clamp_param_messenger import ClampParamMessenger


__all__ = [
    "SelectiveClampParamMessenger",
    "SelectiveScaleMessenger"
]


class HideMixin:

    def __init__(self, hide_fn=None, expose_fn=None,
                 hide_all=True, expose_all=False,
                 hide=None, expose=None,
                 hide_types=None, expose_types=None,
                 **kwargs):
        if not (hide_fn is None or expose_fn is None):
            raise ValueError("Only specify one of hide_fn or expose_fn")
        if hide_fn is not None:
            self.hide_fn = hide_fn
        elif expose_fn is not None:
            self.hide_fn = lambda msg: not expose_fn(msg)
        else:
            self.hide_fn = _make_default_hide_fn(hide_all, expose_all,
                                                 hide, expose,
                                                 hide_types, expose_types)
        super().__init__(**kwargs)


# TODO figure out how to generate the following classes dynamically, in principle the __init__ function could only take
# **kwargs and all the hide/expose related ones will be intercepted by the HideMixin, the rest will be passed on to the
# actual base class. Then we would only need the _process_message and _postprocess_message methods to call the
# super()'s corresponding methods if hide_fn(msg) is False. A more flexible alternative would be to somehow have
# hidden_process and exposed_procces/postprocess methods that get called as appropriate (and do nothing by default?).
class SelectiveScaleMessenger(HideMixin, ScaleMessenger):

    def __init__(self, scale, **kwargs):
        super().__init__(scale=scale, **kwargs)

    def _process_message(self, msg):
        if not self.hide_fn(msg):
            return super()._process_message(msg)


class SelectiveClampParamMessenger(HideMixin, ClampParamMessenger):

    def __init__(self, min_val=None, max_val=None, **kwargs):
        super().__init__(min_val=min_val, max_val=max_val, **kwargs)

    def _process_message(self, msg):
        if not self.hide_fn(msg):
            return super()._process_message(msg)
