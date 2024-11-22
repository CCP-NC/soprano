#!/usr/bin/env python
"""
Test code for NMR Spin System classes.

A spin system is made up of a collection of Site and Coupling objects. There is only one type of Site object, 
but there are several types of Coupling objects, each representing a different type of interaction between two sites.
"""

import os
import unittest

import numpy as np
from ase import io

from soprano.data.nmr import nmr_gamma
from soprano.nmr.coupling import DipolarCoupling
from soprano.nmr.site import Site
from soprano.nmr.tensor import ElectricFieldGradient, MagneticShielding, NMRTensor
from soprano.properties.nmr import DipolarCoupling as DipolarCouplingProperty
from soprano.properties.nmr import DipolarTensor

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")


class TestSite(unittest.TestCase):

    def test_site(self):
        isotope = "1H"
        site_label = "H1"
        ms_data = np.array([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]).reshape(3, 3)
        efg_data = np.array([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]).reshape(3, 3)
        # Test the Site class
        site = Site(
            isotope=isotope,
            label=site_label,
            magnetic_shielding_tensor=MagneticShielding(
                ms_data,
                isotope,
            ),
            efg_tensor=ElectricFieldGradient(
                efg_data,
                isotope,
            ),
            quadrupolar=False,
            euler_convention='zyz',
        )
        self.assertEqual(site.isotope, isotope)
        self.assertEqual(site.label, site_label)
        np.testing.assert_array_equal(site.magnetic_shielding_tensor.data, ms_data)
        self.assertEqual(site.ms_tensor_convention, 'Haeberlen') # default value
        np.testing.assert_array_equal(site.efg_tensor.data, efg_data)
        self.assertEqual(site.efg_tensor_convention, 'NQR') # default value
        self.assertEqual(site.quadrupolar, False)
        self.assertEqual(site.euler_convention, 'zyz')
        self.assertEqual(site.euler_degrees, False) # default value
        self.assertEqual(site.euler_passive, False) # default value




class TestDipolarCoupling(unittest.TestCase):

    def setUp(self):
        atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        self.atoms = atoms

        self.isotopes = {'H': 1, 'C': 13, 'N': 15, 'O': 17}
        self.isotope_map = {k: f"{v}{k}" for k, v in self.isotopes.items()}


        self.dip = DipolarCouplingProperty.get(self.atoms, isotopes=self.isotopes, self_coupling=False)
        self.dip_tensors = DipolarTensor.get(self.atoms, isotopes=self.isotopes, self_coupling=False)
        self.pairs = list(self.dip.keys())



        dipolar_data = np.array([-2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
        self.coupling = DipolarCoupling(
            site_i=1,
            site_j=2,
            species1='1H',
            species2='13C',
            tensor=NMRTensor(dipolar_data, order='n'),
            tag='test_tag',
            euler_convention='zyz',
        )

    def test_coupling_strength(self):
        self.assertEqual(self.coupling.coupling_constant, -1.0)

    def test_site_ij(self):
        self.assertEqual(self.coupling.site_i, 1)
        self.assertEqual(self.coupling.site_j, 2)


    def test_species(self):
        self.assertEqual(self.coupling.species1, '1H')
        self.assertEqual(self.coupling.species2, '13C')

    def test_gamma(self):
        self.assertEqual(self.coupling.gamma1, nmr_gamma('H', 1))
        self.assertEqual(self.coupling.gamma2, nmr_gamma('C', 13))

    def test_tensor(self):
        np.testing.assert_array_equal(self.coupling.tensor.data, np.array([-2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3))

    def test_tag(self):
        self.assertEqual(self.coupling.tag, 'test_tag')

    def test_euler_convention(self):
        self.assertEqual(self.coupling.euler_convention, 'zyz')

    def test_consistent(self):
        """ Test that DipolarCoupling gives the same result as the property method """

        elements = self.atoms.get_chemical_symbols()
        for p in self.pairs:
            i,j = p
            tensor = self.dip_tensors[p] # NMRTensor object from the property method
            species1 = self.isotope_map[elements[i]]
            species2 = self.isotope_map[elements[j]]

            d = DipolarCoupling(site_i=i, site_j=j, tensor = tensor, species1=species1, species2=species2)

            # Euler angles
            np.testing.assert_array_almost_equal(d.euler_angles(degrees=True), self.dip_tensors[p].euler_angles(degrees=True), decimal=4)

            # Coupling constant
            self.assertAlmostEqual(d.coupling_constant, self.dip[p][0])



# class TestISCoupling(unittest.TestCase):

#     def setUp(self):
#         k = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
#         self.coupling = ISCoupling(
#             site_i=1,
#             site_j=2,
#             species1='1H',
#             species2='13C',
#             gamma1=42.0, # in MHz/T
#             gamma2=24.0, # in MHz/T
#             tensor=NMRTensor(k),
#             tag='test_tag',
#             euler_convention='zyz',
#         )

#     def test_coupling_strength(self):
#         self.assertEqual(self.coupling.coupling_constant, 100.0)

#     def test_euler_angles(self):
#         expected_angles = np.array([0, 0, 0])
#         np.testing.assert_array_equal(self.coupling.euler_angles(), expected_angles)

#     def test_site_i(self):
#         self.assertEqual(self.coupling.site_i, 1)

#     def test_site_j(self):
#         self.assertEqual(self.coupling.site_j, 2)

#     def test_species1(self):
#         self.assertEqual(self.coupling.species1, 'H')

#     def test_species2(self):
#         self.assertEqual(self.coupling.species2, 'C')

#     def test_tensor(self):
#         np.testing.assert_array_equal(self.coupling.tensor.data, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3))

#     def test_tag(self):
#         self.assertEqual(self.coupling.tag, 'test_tag')

#     def test_euler_convention(self):
#         self.assertEqual(self.coupling.euler_convention, 'zyz')


if __name__ == '__main__':
    unittest.main()
