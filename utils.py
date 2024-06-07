import importlib
import pkgutil
def get_instance(module, config, *args, **kwargs):
    return getattr(module, config['type'])(*args, **kwargs, **config['args'])


def get_class(module, name):
    package = importlib.import_module(module)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        if module_name.split(".")[-1].replace("_","").lower()== name.lower():
            return getattr(importlib.import_module(module_name), name)
    raise Exception(f"the name {name} is not in the module {module}")
