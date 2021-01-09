import os
import torch
import dill as pickle


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("zero_index", torch.tensor(0))

    @property
    def device(self):
        return self.zero_index.device

    def tensor(self, val, dtype=torch.float32):
        return torch.tensor(val, dtype=dtype)

    def update_history(self, attr, val):
        history = getattr(self, attr)
        history = history.index_copy(0, self.zero_index, val)
        setattr(self, attr, torch.roll(history, -1, 0))

    def update_history_batch(self, attr, val):
        history = getattr(self, attr)
        history[:, 0] = val.squeeze()
        setattr(self, attr, torch.roll(history, -1, 1))

    def save(self, path):
        self.cpu()
        pickle.dump(self, open(path, "wb"))
        # torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, *args, **kwargs):
        return pickle.load(open(path, "rb"))
        # module = cls(*args, **kwargs)
        # module.load_state_dict(torch.load(path))
        # module.eval()
        # return module


def reclass_helper(obj):
    for key in dir(obj):
        if key[:2] == "__":
            continue
        attr = getattr(obj, key)
        if hasattr(attr, "reclass") and not isinstance(attr, type):
            setattr(obj, key, attr.reclass())
        elif isinstance(attr, dict):
            for key2 in attr:
                attr[key2] = reclass_helper(attr[key2])
                setattr(obj, key, attr)
        elif isinstance(attr, list):
            for i in range(len(attr)):
                attr[i] = reclass_helper(attr[i])
                setattr(obj, key, attr)

    return obj


def full_classname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    else:
        return module + "." + o.__class__.__name__


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class LungObj:
    def reclass(self):
        self.__class__ = import_class(full_classname(self))

    def save(self, path, thing=None):
        obj = reclass_helper(thing or self)
        obj.reclass()
        dirname = os.path.abspath(os.path.dirname(path))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        pickle.dump(obj, open(path, "wb"))

    @classmethod
    def load(cls, path):
        return pickle.load(open(path, "rb"))
