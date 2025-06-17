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

from soprano.data.nmr import EFG_TO_CHI, nmr_gamma
from soprano.nmr.coupling import DipolarCoupling, ISCoupling
from soprano.nmr.spin_system import SpinSystem
from soprano.nmr.site import Site
from soprano.nmr.tensor import ElectricFieldGradient, MagneticShielding, NMRTensor
from soprano.properties.nmr import DipolarCoupling as DipolarCouplingProperty
from soprano.properties.nmr import DipolarTensor, NMRSpinSystem, get_sites
from soprano.properties.nmr import DipolarCouplingList, JCouplingList

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
            index=0,
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
            "index": 0,
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
            index=0,
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

    def test_to_mrsimulator(self):
        # Create a site with known tensor values
        isotope = "2H"
        site_label = "H1"
        # Create a ms tensor (First H ms in ethanol.magres)
        ms_data = np.array([
            [30.2981796159, 1.2051069281, 3.67274492938],
            [1.96313294552, 27.5765250451, 2.57545224195],
            [4.21834131673, 2.16271307552, 30.9031525163]])
        ms_reference = 31.7

        # Create an efg tensor that's not just identity
        # (First H efg in ethanol.magres)
        efg_data = np.array([
            [0.12793404309, 0.0514298737569, 0.20226839328],
            [0.0514298737569, -0.133531745662, 0.0414560149276],
            [0.20226839328, 0.0414560149276, 0.00559770257191]
        ])

        site = Site(
            isotope=isotope,
            label=site_label,
            index=0,
            ms=MagneticShielding(ms_data, isotope, reference=ms_reference),
            efg=ElectricFieldGradient(efg_data, isotope)
        )

        # From MRSimulator's to_haeberlen_params function:
        ref_ms_iso = 29.5926190591
        ref_ms_zeta = 5.961009887515737
        ref_ms_eta = 0.1419680007811347
        ref_ms_eulers = [ 2.67320545,  2.35027756, -2.65355747]
        # chemical shift reference
        ref_cs_iso = (ms_reference - ref_ms_iso) / (1.0 - ms_reference*1e-6)

        # From MRSimulator's to_haeberlen_params function:
        ref_efg_Cq = 0.28840675146787703 * EFG_TO_CHI * site.efg.quadrupole_moment
        ref_efg_eta = 0.01819690682375444
        ref_efg_eulers = [-0.60493229,  2.2014038 , -2.9490402 ]

        # Test with default parameters (include everything)
        result = site.to_mrsimulator()

        # Check basic fields
        self.assertEqual(result["isotope"], isotope)
        self.assertEqual(result["label"], site_label)

        # Check MS values present
        self.assertIn("isotropic_chemical_shift", result)
        self.assertIn("shielding_symmetric", result)
        self.assertIn("zeta", result["shielding_symmetric"])
        self.assertIn("eta", result["shielding_symmetric"])
        self.assertIn("alpha", result["shielding_symmetric"])
        self.assertIn("beta", result["shielding_symmetric"])
        self.assertIn("gamma", result["shielding_symmetric"])

        # Check MS values
        self.assertAlmostEqual(result["isotropic_chemical_shift"], ref_cs_iso, places=5)
        self.assertAlmostEqual(result["shielding_symmetric"]["zeta"], ref_ms_zeta, places=5)
        self.assertAlmostEqual(result["shielding_symmetric"]["eta"], ref_ms_eta, places=5)
        # The angles are unlikely to agree due to our choice of normalisations... (TODO)
        # self.assertAlmostEqual(result["shielding_symmetric"]["alpha"], ref_ms_eulers[0], places=5)
        # self.assertAlmostEqual(result["shielding_symmetric"]["beta"], ref_ms_eulers[1], places=5)
        # self.assertAlmostEqual(result["shielding_symmetric"]["gamma"], ref_ms_eulers[2], places=5)

        # Check EFG fields present
        self.assertIn("quadrupolar", result)
        self.assertIn("Cq", result["quadrupolar"])
        self.assertIn("eta", result["quadrupolar"])
        self.assertIn("alpha", result["quadrupolar"])
        self.assertIn("beta", result["quadrupolar"])
        self.assertIn("gamma", result["quadrupolar"])

        # Check EFG values
        self.assertAlmostEqual(result["quadrupolar"]["Cq"], ref_efg_Cq, places=5)
        self.assertAlmostEqual(result["quadrupolar"]["eta"], ref_efg_eta, places=5)
        # The angles are unlikely to agree due to our choice of normalisations... (TODO)
        # self.assertAlmostEqual(result["quadrupolar"]["alpha"], ref_efg_eulers[0], places=5)
        # self.assertAlmostEqual(result["quadrupolar"]["beta"], ref_efg_eulers[1], places=5)
        # self.assertAlmostEqual(result["quadrupolar"]["gamma"], ref_efg_eulers[2], places=5)

        # Test with excluding magnetic shielding
        result = site.to_mrsimulator(include_ms=False)
        self.assertNotIn("isotropic_chemical_shift", result)
        self.assertNotIn("shielding_symmetric", result)

        # Test with excluding electric field gradient
        result = site.to_mrsimulator(include_efg=False)
        self.assertNotIn("quadrupolar", result)

        # Test with excluding angles
        result = site.to_mrsimulator(include_angles=False)
        self.assertIn("shielding_symmetric", result)
        self.assertIn("quadrupolar", result)
        self.assertNotIn("alpha", result["shielding_symmetric"])
        self.assertNotIn("beta", result["shielding_symmetric"])
        self.assertNotIn("gamma", result["shielding_symmetric"])
        self.assertNotIn("alpha", result["quadrupolar"])
        self.assertNotIn("beta", result["quadrupolar"])
        self.assertNotIn("gamma", result["quadrupolar"])

        # Test selective angle inclusion
        result = site.to_mrsimulator(include_ms_angles=True, include_efg_angles=False)
        self.assertIn("alpha", result["shielding_symmetric"])
        self.assertIn("beta", result["shielding_symmetric"])
        self.assertIn("gamma", result["shielding_symmetric"])
        self.assertNotIn("alpha", result["quadrupolar"])
        self.assertNotIn("beta", result["quadrupolar"])
        self.assertNotIn("gamma", result["quadrupolar"])

    def test_to_simpson(self):
        # Create a site with known tensor values
        isotope = "2H"
        site_label = "H1"
        # Create a ms tensor (First H ms in ethanol.magres)
        ms_data = np.array([
            [30.2981796159, 1.2051069281, 3.67274492938],
            [1.96313294552, 27.5765250451, 2.57545224195],
            [4.21834131673, 2.16271307552, 30.9031525163]])
        ms_reference = 31.7

        # Create an efg tensor that's not just identity
        # (First H efg in ethanol.magres)
        efg_data = np.array([
            [0.12793404309, 0.0514298737569, 0.20226839328],
            [0.0514298737569, -0.133531745662, 0.0414560149276],
            [0.20226839328, 0.0414560149276, 0.00559770257191]
        ])

        site = Site(
            isotope=isotope,
            label=site_label,
            index=0,
            ms=MagneticShielding(ms_data, isotope, reference=ms_reference),
            efg=ElectricFieldGradient(efg_data, isotope)
        )

        ms_haeberlen = site.ms.haeberlen_shift
        ref_cs_iso = ms_haeberlen.delta_iso
        ref_cs_zeta = ms_haeberlen.zeta  # Simpson expects reduced anisotropy
        ref_cs_eta = ms_haeberlen.eta
        # ref_cs_zeta = site.ms.shift_reduced_anisotropy

        # From MRSimulator's to_haeberlen_params function:
        ref_efg_Cq = 0.28840675146787703 * EFG_TO_CHI * site.efg.quadrupole_moment
        ref_efg_eta = 0.01819690682375444
        ref_efg_eulers = [-0.60493229,  2.2014038 , -2.9490402 ]

        # Test with default parameters
        ms_block, efg_block = site.to_simpson(include_angles=True)

        # Check basic format for MS block
        self.assertTrue(ms_block.startswith("shift 1"))
        # The MS block should contain 8 elements: "shift", index, iso, aniso, eta, alpha, beta, gamma
        ms_block_elements = ms_block.split()
        self.assertEqual(len(ms_block_elements), 8)
        # Check the isotropic chemical shift is correct (remove the "p" from the string)
        self.assertAlmostEqual(float(ms_block_elements[2][:-1]), ref_cs_iso, places=5)
        # Check the anisotropic chemical shift is correct (remove the "p" from the string)
        self.assertAlmostEqual(float(ms_block_elements[3][:-1]), ref_cs_zeta, places=5)
        # Check the eta is correct
        self.assertAlmostEqual(float(ms_block_elements[4]), ref_cs_eta, places=5)

        # Check basic format for EFG block
        self.assertTrue(efg_block.startswith("quadrupole 1"))
        # The EFG block should contain 8 elements: "quadrupole", index, order, Cq, eta, alpha, beta, gamma
        self.assertEqual(len(efg_block.split()), 8)
        # Check the quadrupole order is correct
        self.assertEqual(efg_block.split()[2], "2")
        # Check the quadrupole Cq is correct
        self.assertAlmostEqual(float(efg_block.split()[3]), ref_efg_Cq, places=5)
        # Check the quadrupole eta is correct
        self.assertAlmostEqual(float(efg_block.split()[4]), ref_efg_eta, places=5)

        # Test excluding MS
        ms_block, efg_block = site.to_simpson(include_ms=False)
        self.assertEqual(ms_block, "")

        # Test excluding EFG
        ms_block, efg_block = site.to_simpson(include_efg=False)
        self.assertEqual(efg_block, "")

        # Test excluding angles
        ms_block, efg_block = site.to_simpson(include_angles=False)
        # Now we should still have 8 elements in MS block
        self.assertEqual(len(ms_block.split()), 8)
        # The last three elements should zero
        self.assertEqual(ms_block.split()[5], "0.0")
        self.assertEqual(ms_block.split()[6], "0.0")
        self.assertEqual(ms_block.split()[7], "0.0")
        
        # And 8 elements in EFG block
        self.assertEqual(len(efg_block.split()), 8)
        # The last three elements should zero
        self.assertEqual(efg_block.split()[5], "0.0")
        self.assertEqual(efg_block.split()[6], "0.0")
        self.assertEqual(efg_block.split()[7], "0.0")


        # Test specific quadrupole order
        ms_block, efg_block = site.to_simpson(q_order=1)
        # Check the quadrupole order is set correctly
        self.assertEqual(efg_block.split()[2], "1")

        # Test selective angle inclusion
        ms_block, efg_block = site.to_simpson(include_ms_angles=True, include_efg_angles=False)
        # Check the MS block has angles that are not zero
        self.assertEqual(len(ms_block.split()), 8)  # not zero angles
        self.assertNotEqual(ms_block.split()[5], "0.0")
        self.assertNotEqual(ms_block.split()[6], "0.0")
        self.assertNotEqual(ms_block.split()[7], "0.0")
        
        self.assertEqual(len(efg_block.split()), 8)  # zero angles
        self.assertEqual(efg_block.split()[5], "0.0")
        self.assertEqual(efg_block.split()[6], "0.0")
        self.assertEqual(efg_block.split()[7], "0.0")


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

    def test_to_mrsimulator(self):
        """Test the to_mrsimulator method of DipolarCoupling"""
        # Create a dipolar coupling with known tensor for testing
        dipolar_data = np.array([-2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 4.0]).reshape(3, 3)
        coupling = DipolarCoupling(
            site_i=0,
            site_j=1,
            species1='1H',
            species2='13C',
            tensor=NMRTensor(dipolar_data, order='n'),
            tag='test_tag',
        )

        # Test with default parameters (include angles)
        result = coupling.to_mrsimulator()

        # Check the structure of the output
        self.assertIn('dipolar', result)
        self.assertIn('D', result['dipolar'])
        self.assertIn('alpha', result['dipolar'])
        self.assertIn('beta', result['dipolar'])
        self.assertIn('gamma', result['dipolar'])

        # Check the coupling constant value
        self.assertEqual(result['dipolar']['D'], 2.0)

        # Test with angles excluded
        result = coupling.to_mrsimulator(include_angles=False)
        self.assertIn('dipolar', result)
        self.assertIn('D', result['dipolar'])
        self.assertNotIn('alpha', result['dipolar'])
        self.assertNotIn('beta', result['dipolar'])
        self.assertNotIn('gamma', result['dipolar'])

    def test_to_simpson(self):
        """Test the to_simpson method of DipolarCoupling"""
        # Create a dipolar coupling with known tensor for testing
        dipolar_data = np.array([-2.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 4.0]).reshape(3, 3)
        coupling = DipolarCoupling(
            site_i=0,
            site_j=1,
            species1='1H',
            species2='13C',
            tensor=NMRTensor(dipolar_data, order='n'),
            tag='test_tag',
        )

        # Test with default parameters (include angles)
        result = coupling.to_simpson()
        
        # Check the format of the string
        self.assertTrue(result.startswith("dipole 1 2"))

        # The coupling constant is scaled by 2*pi in simpson format
        # So we expect 2.0 * 2 * pi = 4*pi Hz
        coupling_constant_part = result.split()[3]
        self.assertAlmostEqual(float(coupling_constant_part), 2.0 * 2 * np.pi, places=5)

        # Check that we have Euler angles (should have 7 parts with angles)
        parts = result.split()
        self.assertEqual(len(parts), 7)

        # Test without angles
        result = coupling.to_simpson(include_angles=False)
        parts = result.split()
        # Should still have 7 parts, but last three should be "0"
        self.assertEqual(len(parts), 7)
        self.assertEqual(parts[4], "0")
        self.assertEqual(parts[5], "0")
        self.assertEqual(parts[6], "0")

class TestISCoupling(unittest.TestCase):

    def setUp(self):
        """
        For this K tensor (from a CASTEP magres file:)
        isc H            1 C            1          2.4613771018822995E+01         -1.7689090415937836E+00         -7.3815468058327873E+00         -1.7342375535431414E+00          3.3870555727450665E+01         -1.5523010487085713E+00         -7.5180123793714939E+00         -1.6000769431832076E+00          2.8229351540541071E+01
        
        We get this magres_old block (again this is the K tensor, not the J tensor!):
        J-coupling Total

            24.6138     -1.7689     -7.3815
            -1.7342     33.8706     -1.5523
            -7.5180     -1.6001     28.2294

        C    1 Isotropic:       28.9046    10^19kg m^-2 s^-2 A^-2

        C    1 Eigenvalue  sigma_xx      18.3983
        C    1 Eigenvector sigma_xx       0.7762      0.1502      0.6123
        C    1 Eigenvalue  sigma_yy      33.9880
        C    1 Eigenvector sigma_yy      -0.5846      0.5353      0.6097
        C    1 Eigenvalue  sigma_zz      34.3274
        C    1 Eigenvector sigma_zz       0.2361      0.8312     -0.5033

        C    1 Anisotropy:       8.1343 # TODO: check this value...

        And from the .castep file (J-coupling values):
        |--------------------------------------------------------------------------------|
        |     Nucleus                               J-coupling values                    |
        |  Species          Ion     FC           SD         PARA       DIA       TOT(Hz) |
        |** H                 1     --.--       --.--       --.--      --.--       --.-- |
        |   C                 1     85.44        0.17        1.18       0.53       87.32 |
        |                                                                                |
        |** Signifies Perturbing Nucleus A of coupling J_AB                              |
        ==================================================================================

        ================================================================================
                    Bond                          Length    1st image    J_iso    J_aniso
                                                    (A)         (A)       (Hz)       (Hz)
        ================================================================================
                    H 1 -- C 1                  1.0944     4.4757      87.32      24.57 # TODO: check the aniso value

        """
        # From the ethanol test case H-C coupling
        k = np.array([[ 2.4613771018822995E+01, -1.7689090415937836E+00, -7.3815468058327873E+00],
                      [-1.7342375535431414E+00, 3.3870555727450665E+01, -1.5523010487085713E+00],
                      [-7.5180123793714939E+00, -1.6000769431832076E+00, 2.8229351540541071E+01]])
        self.k = k
        self.coupling = ISCoupling(
            site_i=0,
            site_j=6,
            species1='1H',
            species2='13C',
            tensor=NMRTensor(k),
            tag='test_tag',
        )

        self.ref_isotropic_J = 87.32260275
        self.ref_anisotropy = -47.61034819209813
        self.ref_reduced_anisotropy = -31.74023212806542
        self.ref_asymmetry = 0.03230181862060256



    # def test_euler_angles(self):
        # expected_angles = np.array([0, 0, 0])
        # np.testing.assert_array_equal(self.coupling.euler_angles(), expected_angles)

    def test_site_i(self):
        self.assertEqual(self.coupling.site_i, 0)

    def test_site_j(self):
        self.assertEqual(self.coupling.site_j, 6)

    def test_species1(self):
        self.assertEqual(self.coupling.species1, '1H')

    def test_species2(self):
        self.assertEqual(self.coupling.species2, '13C')

    def test_tensor(self):
        np.testing.assert_array_equal(self.coupling.tensor.data, self.k)

    def test_tag(self):
        self.assertEqual(self.coupling.tag, 'test_tag')

    def test_coupling_strength(self):
        self.assertAlmostEqual(self.coupling.coupling_constant, self.ref_isotropic_J, places=6)

    def test_asymmetry(self):
        """Test the asymmetry parameter of the ISCoupling"""
        self.assertAlmostEqual(self.coupling.J_asymmetry, self.ref_asymmetry, places=6)

    def test_anisotropy(self):
        """Test the anisotropy parameter of the ISCoupling
        
        Note that CASTEP sorts the K eigenvalues in abs ascending order, so the
        anisotropy is different to the one calculated here, following the Haeberlen convention.
        """
        self.assertAlmostEqual(self.coupling.J_anisotropy, self.ref_anisotropy, places=6)
    def test_reduced_anisotropy(self):
        """Test the reduced anisotropy of the ISCoupling"""
        self.assertAlmostEqual(self.coupling.J_reduced_anisotropy, self.ref_reduced_anisotropy, places=6)

    def test_to_mrsimulator(self):
        """Test the to_mrsimulator method of ISCoupling"""
        mrsimulator_dict = self.coupling.to_mrsimulator()
        # Check the structure of the output
        self.assertIn('isotropic_j', mrsimulator_dict)
        self.assertIn('zeta', mrsimulator_dict['j_symmetric'])
        self.assertIn('eta', mrsimulator_dict['j_symmetric'])
        self.assertIn('alpha', mrsimulator_dict['j_symmetric'])
        self.assertIn('beta', mrsimulator_dict['j_symmetric'])
        self.assertIn('gamma', mrsimulator_dict['j_symmetric'])
        # Check the isotropic J value
        self.assertAlmostEqual(mrsimulator_dict['isotropic_j'], self.ref_isotropic_J, places=6)
        self.assertAlmostEqual(mrsimulator_dict['j_symmetric']['zeta'], self.ref_reduced_anisotropy, places=6)
        self.assertAlmostEqual(mrsimulator_dict['j_symmetric']['eta'], self.ref_asymmetry, places=6)

        # TODO: add Euler angle checks

    def test_to_simpson(self):
        """Test the to_simpson method of ISCoupling"""
        simpson_str = self.coupling.to_simpson()
        # Check the format of the string:
        # jcoupling i j Jiso_ij Janiso_ij eta_ij alpha beta gamma
        # NB Simpson uses 1-based indexing, so site_i=0 becomes 1, site_j=6 becomes 7
        self.assertTrue(simpson_str.startswith("jcoupling 1 7"))
        # The string should have 9 parts
        parts = simpson_str.split()
        self.assertEqual(len(parts), 9)

        # Isotropic J value
        self.assertAlmostEqual(float(parts[3]), self.ref_isotropic_J, places=6)

        # Anisotropy value
        self.assertAlmostEqual(float(parts[4]), self.ref_anisotropy, places=6)

        # TODO: add checks for angles


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

        self.dipcouplings = DipolarCouplingList.get(self.atoms, isotopes=self.isotopes, self_coupling=False)
        self.jcouplings = JCouplingList.get(self.atoms, isotopes=self.isotopes, self_coupling=False)

        self.couplings = self.dipcouplings + self.jcouplings


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
        self.assertEqual(len(couplings), 2) # One dipolar and one J-coupling
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
        spin_system_prop = NMRSpinSystem.get(self.atoms, include_dipolar = True, include_j = True)
        self.assertEqual(spin_system_manual, spin_system_prop)

    def test_spin_system_subsystem(self):
        from soprano.selection import AtomSelection

        sel = AtomSelection.from_element(self.atoms, 'H')
        spin_system_sel = NMRSpinSystem.get(sel.subset(self.atoms), include_dipolar = True, include_j = True)

        # Convert to dictionaries and compare
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
