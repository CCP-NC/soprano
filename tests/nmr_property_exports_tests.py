"""The public NMR AtomsProperty classes are importable from soprano.properties.nmr."""

import inspect
import unittest

import soprano.properties.nmr as nmr_pkg
from soprano.properties.atomsproperty import AtomsProperty
from soprano.properties.nmr import efg, isc, ms

# Building blocks that are deliberately not re-exported
_PRIVATE = {"ISCDiagonal", "KTensor"}


class TestNMRPropertyExports(unittest.TestCase):
    def test_all_properties_reexported(self):
        for module in (ms, efg, isc):
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, AtomsProperty)
                    and obj is not AtomsProperty
                    and obj.__module__ == module.__name__
                    and name not in _PRIVATE
                ):
                    with self.subTest(name=name):
                        self.assertIs(getattr(nmr_pkg, name, None), obj)


if __name__ == "__main__":
    unittest.main()
