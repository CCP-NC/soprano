"""
utils.py

Contains the following package-wide useful routines:

[TO ADD]
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def abc2cart(abc):
    """Transforms an axes and angles representation of lattice parameters
       into a Cartesian one

    """

    abc = np.array(abc, copy=False)

    if abc.shape != (2, 3):
        raise ValueError("Invalid abc passed to abc2cart")

    cart = []
    sin = np.sin(abc[1, :])
    cos = np.cos(abc[1, :])
    cart.append([sin[2], cos[2], 0.0])
    cart.append([0.0, 1.0, 0.0])
    cart.append([(cos[1]-cos[0]*cos[2])/sin[2], cos[0], 0.0])
    cart[2][2] = np.sqrt(1.0-cart[2][0]**2.0-cart[2][1]**2.0)
    cart = np.array([np.array(cart[i])*abc[0, i] for i in range(3)])

    return cart


def cart2abc(cart):
    """Transforms a Cartesian representation of lattice parameters
       into an axes and angles one

    """

    cart = np.array(cart, copy=False)

    if cart.shape != (3, 3):
        raise ValueError("Invalid cart passed to cart2abc")

    abc = np.zeros((2, 3))
    abc[0, :] = np.linalg.norm(cart, axis=1)
    # Now for the angles
    cart_roll_1 = np.roll(cart, 1, axis=0)
    cart_roll_2 = np.roll(cart, 2, axis=0)
    cross_cart = np.linalg.norm(np.cross(cart_roll_1, cart_roll_2), axis=1)
    dot_cart = np.diagonal(np.dot(cart_roll_1, cart_roll_2.T))
    abc[1, :] = np.arctan2(cross_cart, dot_cart)

    return abc


def hkl2d2_matgen(abc):
    """Generate a matrix that turns hkl indices into inverse crystal
       plane distances for a given lattice in ABC form

    """

    abc = np.array(abc, copy=False)

    if abc.shape != (2, 3):
        raise ValueError("Invalid abc passed to hkl2d2_matgen")

    sin = np.sin(abc[1, :])
    cos = np.cos(abc[1, :])
    a2b2 = (abc[0, 0]*abc[0, 1])**2.0
    a2c2 = (abc[0, 0]*abc[0, 2])**2.0
    b2c2 = (abc[0, 1]*abc[0, 2])**2.0
    abc_prod = np.prod(abc[0, :])

    hkl2d2 = np.matrix([[b2c2*sin[0]**2.0,
                         abc_prod*abc[0, 2]*(cos[0]*cos[1]-cos[2]),
                         abc_prod*abc[0, 1]*(cos[0]*cos[2]-cos[1])],
                        [abc_prod*abc[0, 2]*(cos[0]*cos[1]-cos[2]),
                         a2c2*sin[1]**2.0,
                         abc_prod*abc[0, 0]*(cos[1]*cos[2]-cos[0])],
                        [abc_prod*abc[0, 1]*(cos[0]*cos[2]-cos[1]),
                         abc_prod*abc[0, 0]*(cos[1]*cos[2]-cos[0]),
                         a2b2*sin[2]**2.0]])

    hkl2d2 /= abc_prod**2.0*(1.0-np.dot(cos, cos)+2.0*np.prod(cos))

    return hkl2d2


def inv_plane_dist(hkl, hkl2d2):
    """Calculate inverse planar distance for a given set of
       Miller indices h, k, l.

    """

    hkl = np.array(hkl, copy=False)

    if hkl.shape != (3,):
        raise ValueError("Invalid hkl passed to inv_plane_dist")

    return np.sqrt(np.dot(np.dot(hkl2d2, hkl), hkl)[0, 0])


def minimum_supcell(latt_cart, max_r):

    """ Generate the bounds for a supercell containing a sphere
        of given radius, knowing the unit cell.

    | Args:
    |   latt_cart (np.ndarray): unit cell in cartesian form
    |   max_r (float): radius of the sphere contained in the supercell
    
    | Returns:
    |   bounds (tuple[int]): bounds of the supercell to be built.
    |                        These are to be interpreted as the 
    |                        thickness of the "shell" one has to build 
    |                        around a core unit cell: for example (1,2,1)
    |                        means the supercell will be 3x5x3
    
    | Raises:
    |   ValueError: if some of the arguments are invalid

    """

    latt_cart = np.array(latt_cart)
    if (latt_cart.shape != (3,3)):
        raise ValueError("Invalid latt_cart passed to minimum_supcell")

    # The logic of this algorithm isn't obvious, so let's make it clearer.
    # 
    # By diagonalizing it we can calculate the transformation matrix which morphs a simple unit sphere into this ellipsoid
    # through scaling (with the eigenvalues) and rotating (with the eigenvectors).
    # We have thus that a point p in unit sphere space becomes q = M*p in our space of interest (that is, hkl space). On the other hand,
    # a point q in hkl space will be transformed into unit sphere space by p = M^-1*q. 
    # Now, the boundaries of the ellipsoids are none other but the points which, in hkl space, have normals aligned with the axes.
    # And NORMALS transform between spaces with a matrix that is the inverse transpose of the regular one.
    # So the logic goes:
    # - take an axis direction in hkl space (for example, [1,0,0])
    # - transform it into unit sphere space by premultiplying (M^-1)^-1^T = M^T
    # - now that we have the direction of the normal in unit sphere space, we just need to normalize it to find the point p which has that normal
    #   (thanks to the properties of the unit sphere)
    # - then we transform back to hkl space with M and there you go, that's your boundary point.
    # - wash, rinse, repeat for all axes directions. Find the maxima in each direction and you've got your box.

    r_Matrix = np.dot(latt_cart, latt_cart.T)
    r_evals, r_evecs = np.linalg.eigh(r_Matrix)

    unit_transf_M = np.dot(r_evecs, np.diag(1.0/np.sqrt(r_evals)))          # Unit sphere - to - ellipsoid transformation matrix
    # To find the boundaries, we need to iterate over the three main directions. We do this implicitly though
    qMatrix = np.dot(unit_transf_M, max_r*(unit_transf_M/np.linalg.norm(unit_transf_M, axis=1)[:,None]).T)
    r_bounds = np.max(np.ceil(abs(qMatrix)), axis=1).astype(int)

    return tuple(r_bounds)