import importlib
import pkgutil
import types
def get_instance(module, config, *args, **kwargs):
    return getattr(module, config['type'])(*args, **kwargs, **config['args'])

def get_attr(module_name, name, ignore_part):
    if isinstance(module_name, types.ModuleType):
        pass
    else:
        module_name = importlib.import_module(module_name)
    all = dir(module_name)
    for i in all:
        if i.replace(ignore_part,"").replace("_","").lower() == name.lower():
            return getattr(module_name, i)
    raise Exception(f"cannot find {name} with ignore part {ignore_part} in module {module_name} ")

def get_class(package, name):
    if isinstance(package, types.ModuleType):
        pass
    else:
        package = importlib.import_module(package)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        cls = module_name.split(".")[-1].replace("_","")
        if cls.lower()== name.lower():
            return get_attr(module_name, cls,"")
    raise Exception(f"the name {name} is not in the module {package}")

