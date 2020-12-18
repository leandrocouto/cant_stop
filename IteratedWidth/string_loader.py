import importlib
from importlib import abc
import sys
import types


class StringLoader(importlib.abc.Loader):

    def __init__(self, modules):
        self._modules = modules

    def has_module(self, fullname):
        return (fullname in self._modules)

    def create_module(self, spec):
        if self.has_module(spec.name):
            module = types.ModuleType(spec.name)
            exec(self._modules[spec.name], module.__dict__)
            return module

    def exec_module(self, module):
        pass


class StringFinder(importlib.abc.MetaPathFinder):

    def __init__(self, loader):
        self._loader = loader

    def find_spec(self, fullname, path, target=None):
        if self._loader.has_module(fullname):
            return importlib.machinery.ModuleSpec(fullname, self._loader)


if __name__ == '__main__':
    modules = {
    'my_module': 
"""class Foo:
    def __init__(self, *args: str):
        self.args = args
    def bar(self):
        return ', '.join(self.args)
"""}

    finder = StringFinder(StringLoader(modules))
    sys.meta_path.append(finder)

    from my_module import Foo
    foo = Foo('Hello', 'World!')
    print(foo.bar())
    #print(my_module.BAZ)