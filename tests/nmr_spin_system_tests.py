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
from soprano.nmr.spin_system import SpinSystem
from soprano.nmr.site import Site
from soprano.nmr.tensor import ElectricFieldGradient, MagneticShielding, NMRTensor
from soprano.properties.nmr import DipolarCoupling as DipolarCouplingProperty
from soprano.properties.nmr import DipolarTensor, NMRSpinSystem, get_sites
from soprano.properties.nmr import DipolarCouplingList

_TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

# def compare_spin_system_dumps(obj1: Any, obj2: Any) -> bool:
#     """
#     Recursively compare two objects, handling various nested data types.
    
#     Args:
#         obj1: First object to compare
#         obj2: Second object to compare
    
#     Returns:
#         bool: True if objects are equivalent, False otherwise
#     """
#     # Direct equality check for primitive types
#     if type(obj1) != type(obj2):
#         return False
    
#     # Handle dictionaries
#     if isinstance(obj1, dict):
#         if obj1.keys() != obj2.keys():
#             return False
        
#         for key in obj1:
#             if not compare_spin_system_dumps(obj1[key], obj2[key]):
#                 return False
#         return True
    
#     # Handle lists
#     elif isinstance(obj1, list):
#         if len(obj1) != len(obj2):
#             return False
        
#         for item1, item2 in zip(obj1, obj2):
#             if not compare_spin_system_dumps(item1, item2):
#                 return False
#         return True
    
#     # Handle NMRTensor objects
#     elif hasattr(obj1, 'data') and hasattr(obj2, 'data'):
#         return np.array_equal(obj1.data, obj2.data)
    
#     # Handle tuples
#     elif isinstance(obj1, tuple):
#         return np.array_equal(obj1, obj2)
    
#     # Handle numpy arrays
#     elif isinstance(obj1, np.ndarray):
#         return np.array_equal(obj1, obj2)
    
#     # Fallback to direct comparison for other types
#     return obj1 == obj2

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
            ms=MagneticShielding(
                ms_data,
                isotope,
            ),
            efg=ElectricFieldGradient(
                efg_data,
                isotope,
            ),
        )
        self.assertEqual(site.isotope, isotope)
        self.assertEqual(site.label, site_label)
        np.testing.assert_array_equal(site.ms.data, ms_data)
        self.assertEqual(site.ms.order, 'h') # default value
        np.testing.assert_array_equal(site.efg.data, efg_data)
        self.assertEqual(site.efg.order, 'n') # default value

    def test_site_from_dict(self):
        isotope = "1H"
        site_label = "H1"
        ms_data = np.array([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]).reshape(3, 3)
        efg_data = np.array([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -20.0]).reshape(3, 3)
        site_dict = {
            "isotope": isotope,
            "label": site_label,
            "ms": {
                "data": ms_data.tolist(),
                "species": isotope,
            },
            "efg": {
                "data": efg_data.tolist(),
                "species": isotope,
            },
        }
        site = Site.from_dict(site_dict)
        self.assertEqual(site.isotope, isotope)
        self.assertEqual(site.label, site_label)
        np.testing.assert_array_equal(site.ms.data, ms_data)
        self.assertEqual(site.ms.order, 'h')
        np.testing.assert_array_equal(site.efg.data, efg_data)
        self.assertEqual(site.efg.order, 'n')

    def test_site_equality(self):
        isotope = "1H"
        site_label = "H1"
        ms_data = np.array([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]).reshape(3, 3)
        efg_data = np.array([10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, -20.0]).reshape(3, 3)
        site1 = Site(
            isotope=isotope,
            label=site_label,
            ms=MagneticShielding(
                ms_data,
                isotope,
            ),
            efg=ElectricFieldGradient(
                efg_data,
                isotope,
            ),
        )
        site1_copy = site1.copy()
        site2 = site1.copy(update={"label": "H2"})
        site3 = site1.copy(update={"label": "H1"})
        site4 = site1.copy(update={"ms": None})
        site5 = site1.copy(update={"efg": None})
        
        self.assertEqual(site1, site1_copy)
        self.assertEqual(site1, site3)
        self.assertNotEqual(site1, site2)
        self.assertNotEqual(site1, site4)
        self.assertNotEqual(site1, site5)



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

    def test_property(self):
        """ Test the DipolarCoupling AtomsProperty method """
        sel_i = [0,1,2]
        sel_j = [3,4,5]

        dip_list = DipolarCouplingList.get(self.atoms, sel_i=sel_i, sel_j=sel_j, isotopes=self.isotopes, self_coupling=False)

        self.assertEqual(len(dip_list), 9) # 3 sites * 3 sites


        pairs = [(d.site_i, d.site_j) for d  in dip_list]
        # Compare the coupling constants to those from the DipolarCouplingProperty method
        for i, pair in enumerate(pairs):
            self.assertAlmostEqual(dip_list[i].coupling_constant, self.dip[pairs[i]][0])

    def test_equality(self):
        dipolar_data = np.array([-2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
        coupling1 = DipolarCoupling(
            site_i=1,
            site_j=2,
            species1='1H',
            species2='13C',
            tensor=NMRTensor(dipolar_data, order='n'),
            tag='test_tag',
        )
        coupling1_copy = coupling1.copy()
        coupling2 = coupling1.copy(update={"site_i": 2})
        coupling3 = coupling1.copy(update={"site_j": 1})
        coupling4 = coupling1.copy(update={"tensor": None})
        coupling5 = coupling1.copy(update={"tag": "new_tag"})
        
        self.assertEqual(coupling1, coupling1_copy)
        self.assertNotEqual(coupling1, coupling2)
        self.assertNotEqual(coupling1, coupling3)
        self.assertNotEqual(coupling1, coupling4)
        self.assertNotEqual(coupling1, coupling5)


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



class TestSpinSystem(unittest.TestCase):
    """
    Test the SpinSystem class

    Manually create a SpinSystem object and check that the attributes are set correctly.
    """

    def setUp(self):
        atoms = io.read(os.path.join(_TESTDATA_DIR, "ethanol.magres"))
        self.atoms = atoms

        self.isotopes = {'H': 1, 'C': 13, 'N': 15, 'O': 17}
        self.isotope_map = {k: f"{v}{k}" for k, v in self.isotopes.items()}

        self.dip = DipolarCouplingProperty.get(self.atoms, isotopes=self.isotopes, self_coupling=False)
        self.dip_tensors = DipolarTensor.get(self.atoms, isotopes=self.isotopes, self_coupling=False)
        self.pairs = list(self.dip.keys())

        self.sites = get_sites(self.atoms, isotopes=self.isotopes)

        self.couplings = DipolarCouplingList.get(self.atoms, isotopes=self.isotopes, self_coupling=False)


    def test_spin_system(self):
        # Test the SpinSystem class
        spin_system = SpinSystem(sites=self.sites, couplings=self.couplings)
        self.assertEqual(len(spin_system.sites), len(self.sites))
        self.assertEqual(len(spin_system.couplings), len(self.couplings))

        for i, site in enumerate(spin_system.sites):
            self.assertEqual(site.isotope, self.sites[i].isotope)
            self.assertEqual(site.label, self.sites[i].label)
            np.testing.assert_array_equal(site.ms.data, self.sites[i].ms.data)
            np.testing.assert_array_equal(site.efg.data, self.sites[i].efg.data)

        for i, coupling in enumerate(spin_system.couplings):
            self.assertEqual(coupling.site_i, self.couplings[i].site_i)
            self.assertEqual(coupling.site_j, self.couplings[i].site_j)
            np.testing.assert_array_equal(coupling.tensor.data, self.couplings[i].tensor.data)

    def test_spin_system_add_site(self):
        
        spin_system = SpinSystem(sites=[], couplings=[])
        for site in self.sites:
            spin_system.add_site(site)
        self.assertEqual(len(spin_system.sites), len(self.sites))
    
    def test_spin_system_update_site(self):
        spin_system = SpinSystem(sites=self.sites, couplings=[])
        site = self.sites[0]
        site.label = 'test_label'
        spin_system.update_site(0, site)
        self.assertEqual(spin_system.sites[0].label, 'test_label')
    
    def test_spin_system_remove_site(self):
        spin_system = SpinSystem(sites=self.sites, couplings=[])
        spin_system.remove_site(0)
        self.assertEqual(len(spin_system.sites), len(self.sites) - 1)

    def test_spin_system_add_coupling(self):
        spin_system = SpinSystem(sites=self.sites, couplings=[])
        for coupling in self.couplings:
            spin_system.add_coupling(coupling)
        self.assertEqual(len(spin_system.couplings), len(self.couplings))

    def test_spin_system_get_couplings(self):
        spin_system = SpinSystem(sites=self.sites, couplings=self.couplings)
        couplings = spin_system.get_couplings(0, 1)
        self.assertEqual(len(couplings), 1)
        self.assertEqual(couplings[0].site_i, 0)
        self.assertEqual(couplings[0].site_j, 1)

    def test_spin_system_update_coupling(self):
        spin_system = SpinSystem(sites=self.sites, couplings=self.couplings)
        coupling = self.couplings[0]
        coupling.gamma1 = 100.0
        spin_system.update_coupling(0, 1, coupling)
        self.assertEqual(spin_system.couplings[0].gamma1, 100.0)

    def test_spin_system_remove_coupling(self):
        spin_system = SpinSystem(sites=self.sites, couplings=self.couplings)
        spin_system.remove_coupling(0, 1, coupling_type='D')
        self.assertEqual(len(spin_system.couplings), len(self.couplings) - 1)

    def test_spin_system_equality(self):
        spin_system = SpinSystem(sites=self.sites, couplings=self.couplings)
        spin_system_copy = spin_system.copy()
        spin_system2 = SpinSystem(sites=self.sites, couplings=[])
        spin_system3 = SpinSystem(sites=[], couplings=self.couplings)
        spin_system4 = SpinSystem(sites=self.sites, couplings=self.couplings[1:])
        spin_system5 = SpinSystem(sites=self.sites[1:], couplings=self.couplings)
        
        self.assertEqual(spin_system, spin_system_copy)
        self.assertNotEqual(spin_system, spin_system2)
        self.assertNotEqual(spin_system, spin_system3)
        self.assertNotEqual(spin_system, spin_system4)
        self.assertNotEqual(spin_system, spin_system5)

        
    def test_spin_system_property(self):
        

        spin_system_manual = SpinSystem(sites = self.sites, couplings = self.couplings)
        spin_system_prop = NMRSpinSystem.get(self.atoms, include_dipolar = True)
        self.assertEqual(spin_system_manual, spin_system_prop)

    def test_spin_system_subsystem(self):
        from soprano.selection import AtomSelection

        spin_system_all = NMRSpinSystem.get(self.atoms, include_dipolar = True)
        sel = AtomSelection.from_element(self.atoms, 'H')
        spin_system_sel = NMRSpinSystem.get(sel.subset(self.atoms), include_dipolar = True)

        # Convert to dictionaries and compare
        all_dict = spin_system_all.model_dump()
        sel_dict = spin_system_sel.model_dump()

        # Check that the sites are all H
        self.assertTrue(all(site['label'] == 'H' for site in sel_dict['sites']))
        # Check all couplings have the same gamma1 and gamma2
        sel_gamma1s = [coupling['gamma1'] for coupling in sel_dict['couplings'] ]
        sel_gamma2s = [coupling['gamma2'] for coupling in sel_dict['couplings'] ]
        self.assertTrue(all(gamma == sel_gamma1s[0] for gamma in sel_gamma1s))
        self.assertTrue(all(gamma == sel_gamma2s[0] for gamma in sel_gamma2s))


if __name__ == '__main__':
    unittest.main()
