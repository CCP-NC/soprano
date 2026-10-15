"""AtomsProperty.get honours instance parameters when called on an instance."""

import os
import unittest

import numpy as np
from ase.io import read

from soprano.collection import AtomsCollection
from soprano.properties.atomsproperty import AtomsProperty
from soprano.properties.nmr import EFGQuadrupolarConstant

DATA = os.path.join(os.path.dirname(__file__), "test_data", "ethanol.magres")


class TestInstanceGet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.atoms = read(DATA)
        cls.h_idx = [i for i, s in enumerate(cls.atoms.get_chemical_symbols()) if s == "H"]

    def test_instance_get_uses_instance_params(self):
        inst = EFGQuadrupolarConstant(isotopes={"H": 2})
        from_instance = inst.get(self.atoms)
        from_class = EFGQuadrupolarConstant.get(self.atoms, isotopes={"H": 2})
        np.testing.assert_allclose(from_instance, from_class)
        self.assertTrue(np.all(np.asarray(from_instance)[self.h_idx] != 0))

    def test_instance_get_matches_call(self):
        inst = EFGQuadrupolarConstant(isotopes={"H": 2})
        np.testing.assert_allclose(inst.get(self.atoms), inst(self.atoms))

    def test_explicit_kwargs_override_instance_params(self):
        inst = EFGQuadrupolarConstant(isotopes={"H": 2})
        overridden = inst.get(self.atoms, isotopes={"H": 1})
        np.testing.assert_allclose(overridden, EFGQuadrupolarConstant.get(self.atoms, isotopes={"H": 1}))
        np.testing.assert_allclose(np.asarray(overridden)[self.h_idx], 0)

    def test_instance_get_with_selection(self):
        inst = EFGQuadrupolarConstant(isotopes={"H": 2})
        selected = inst.get(self.atoms, selection="H")
        self.assertEqual(len(selected), len(self.h_idx))
        np.testing.assert_allclose(selected, np.asarray(inst.get(self.atoms))[self.h_idx])

    def test_class_get_unchanged(self):
        default_run = EFGQuadrupolarConstant.get(self.atoms)
        np.testing.assert_allclose(default_run, EFGQuadrupolarConstant().get(self.atoms))
        np.testing.assert_allclose(np.asarray(default_run)[self.h_idx], 0)

    def test_unknown_kwarg_raises_same_on_class_and_instance(self):
        with self.assertRaises(TypeError):
            EFGQuadrupolarConstant.get(self.atoms, not_a_param=1)
        with self.assertRaises(TypeError):
            EFGQuadrupolarConstant(isotopes={"H": 2}).get(self.atoms, not_a_param=1)



class _Scale(AtomsProperty):

    default_name = "scale"
    default_params = {"factor": 1.0}

    @staticmethod
    def extract(s, factor):
        return len(s) * factor


class TestBaseMean(unittest.TestCase):

    def test_mean_uses_instance_params(self):
        coll = AtomsCollection([read(DATA), read(DATA)])
        self.assertEqual(_Scale(factor=2).mean(coll), 2 * len(read(DATA)))
        self.assertEqual(_Scale().mean(coll), len(read(DATA)))


class TestInstanceGetCollection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.atoms = read(DATA)

    def test_collection_uses_instance_params(self):
        inst = EFGQuadrupolarConstant(isotopes={"H": 2})
        coll = AtomsCollection([self.atoms, self.atoms])
        arr = inst.get(coll)
        self.assertEqual(len(arr), 2)
        np.testing.assert_allclose(arr[0], inst.get(self.atoms))
        np.testing.assert_allclose(arr[1], inst.get(self.atoms))

    def test_collection_store_array_uses_default_name(self):
        inst = EFGQuadrupolarConstant(isotopes={"H": 2})
        coll = AtomsCollection([self.atoms, self.atoms])
        inst.get(coll, store_array=True)
        stored = coll.get_array(EFGQuadrupolarConstant.default_name)
        np.testing.assert_allclose(stored[0], inst.get(self.atoms))


if __name__ == "__main__":
    unittest.main()
