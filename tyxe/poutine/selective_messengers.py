from pyro.poutine.mask_messenger import MaskMessenger
from pyro.poutine.scale_messenger import ScaleMessenger
from pyro.poutine.block_messenger import _make_default_hide_fn


__all__ = [
    "SelectiveMaskMessenger",
    "SelectiveScaleMessenger"
]


class SelectiveMixin(object):

    def __init__(self, *args,
                 hide_fn=None, expose_fn=None,
                 hide_all=True, expose_all=False,
                 hide=None, expose=None,
                 hide_types=None, expose_types=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
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

    def _process_message(self, msg):
        if not self.hide_fn(msg):
            super()._process_message(msg)


class SelectiveMaskMessenger(SelectiveMixin, MaskMessenger): pass
class SelectiveScaleMessenger(SelectiveMixin, ScaleMessenger): pass
