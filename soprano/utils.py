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

import os
import numpy as np
from ase.quaternions import Quaternion


def seedname(path):
    """Get the filename (with no extension) from a full path"""
    return os.path.splitext(os.path.basename(path))[0]


def replace_folder(path, new_folder):
    """Replace the folder of the given path with a new one"""
    return os.path.join(new_folder, os.path.basename(path))


def progbar(i, i_max, bar_len=20, spinner=True, spin_rate=3.0):
    """A textual progress bar for the command line

    | Args:
    |   i (int): current progress index
    |   max_i (int): final progress index
    |   bar_len (Optional[int]): length in characters of the bar (no brackets)
    |   spinner (Optional[bool]): show a spinner at the end
    |   spin_rate (Optional[float]): spinner rotation speed (turns per full
    |                                progress)

    | Returns:
    |   bar (str): a progress bar formatted as requested

    """
    block = {True: '\u2588', False: ' '}
    spin = '|\\-/'
    perc = i/float(i_max)*bar_len
    bar = '[{0}]'.format(''.join([block[i < perc] for i in range(bar_len)]))
    if spinner:
        bar += ' {0}'.format(spin[int(perc*spin_rate) % len(spin)])

    return bar


def parse_intlist(string):
    """Parse a list of ints from a string"""
    return [int(x) for x in string.split()]


def parse_floatlist(string):
    """Parse a list of floats from a string"""
    return [float(x) for x in string.split()]


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


def minimum_supcell(max_r, latt_cart=None, r_matrix=None,
                    pbc=[True, True, True]):
    """
    Generate the bounds for a supercell containing a sphere
    of given radius, knowing the unit cell.

    | Args:
    |   max_r (float): radius of the sphere contained in the supercell
    |   latt_cart (np.ndarray): unit cell in cartesian form
    |   r_matrix (np.ndarray): matrix for the quadratic form returning
    |                          r^2 for this supercell.
    |                          Alternative to latt_cart, for a direct
    |                          space cell would be equal to
    |                          np.dot(latt_cart, latt_cart.T)
    |   pbc ([bool, bool, bool]): periodic boundary conditions - if
    |                             a boundary is not periodic the
    |                             range returned will always be zero
    |                             in that dimension

    | Returns:
    |   shape (tuple[int]):  shape of the supercell to be built.

    | Raises:
    |   ValueError: if some of the arguments are invalid

    """

    if latt_cart is not None:
        latt_cart = np.array(latt_cart, copy=False)
        if latt_cart.shape != (3, 3):
            raise ValueError("Invalid latt_cart passed to minimum_supcell")
        r_matrix = np.dot(latt_cart, latt_cart.T)
    elif r_matrix is not None:
        r_matrix = np.array(r_matrix, copy=False)
        if r_matrix.shape != (3, 3):
            raise ValueError("Invalid r_matrix passed to minimum_supcell")
    else:
        raise ValueError("One between latt_cart and r_matrix has to"
                         "be present")

    # The logic of this algorithm isn't obvious, so let's make it clearer.
    #
    # What we are basically looking for is the AABB bounding box of a sphere
    # in absolute space.
    # This becomes a rotated ellipsoid in fractional coordinates space.
    # r_matrix represents a quadratic form which turns a fractional coordinate
    # into a squared distance in space.
    # By diagonalizing it we can calculate the transformation matrix
    # which morphs a simple unit sphere into this ellipsoid through
    # scaling (with the eigenvalues) and rotating (with the eigenvectors).
    # We have thus that a point p in unit sphere space becomes q = M*p
    # in fractional space. On the other hand, a point q in fractional space
    # will be transformed into absolute space by p = M^-1*q.
    # Now, the boundaries of the ellipsoids are none other but the points
    # which, in regular space, have normals aligned with the axes.
    # And NORMALS transform between spaces with a matrix that is the
    # inverse transpose of the regular one. So the logic goes:
    # - take an axis direction in fractional space (for example, [1,0,0])
    # - transform it into absolute space by premultiplying (M^-1)^-1^T = M^T
    # - now that we have the direction of the normal in absolute space,
    # we just need to normalize it to find the point p which has that normal
    # (thanks to the properties of the unit sphere)
    # - then we transform back to fractional space with M and there you go,
    # that's your boundary point.
    # - wash, rinse, repeat for all axes directions.
    # Find the maxima in each direction and you've got your box.

    r_evals, r_evecs = np.linalg.eigh(r_matrix)

    # Unit sphere - to - ellipsoid transformation matrix
    utransf_matrix = np.dot(r_evecs, np.diag(1.0/np.sqrt(r_evals)))
    # To find the boundaries, we need to iterate over the three main
    # directions. We do this implicitly though.
    qmatrix = np.dot(utransf_matrix,
                     max_r*(utransf_matrix/np.linalg.norm(utransf_matrix,
                                                          axis=1)[:, None]).T)
    r_bounds = np.max(np.ceil(abs(qmatrix)), axis=1).astype(int)
    r_bounds = np.where(pbc, r_bounds, 0)

    return tuple([2*r+1 for r in r_bounds])


def supcell_gridgen(latt_cart, shape):
    """
    Generate a full linearized grid for a supercell with r_bounds
    and a base unit cell in Cartesian form.

    | Args:
    |   latt_cart (np.ndarray): unit cell in cartesian form
    |   shape (tuple[int]):  shape of the supercell to be built,
    |                        as returned by minimum_supcell.

    | Returns:
    |   neigh_i_grid (np.ndarray): supercell grid in fractional coordinates
    |   neigh_grid (np.ndarray): supercell grid in cartesian coordinates

    | Raises:
    |   ValueError: if some of the arguments are invalid

    """

    latt_cart = np.array(latt_cart, copy=False)
    if latt_cart.shape != (3, 3):
        raise ValueError("Invalid latt_cart passed to supcell_gridgen")

    shape = np.array(shape, copy=False).astype(int)
    if shape.shape != (3,):
        raise ValueError("Invalid shape passed to supcell_gridgen")

    min_bounds = (-((shape-1)/2)).astype(int)
    max_bounds = shape+min_bounds
    x_range = range(min_bounds[0], max_bounds[0])
    y_range = range(min_bounds[1], max_bounds[1])
    z_range = range(min_bounds[2], max_bounds[2])

    # We now generate a grid of neighbours to check for contact with
    # First just the supercell indicess
    neigh_i_grid = np.swapaxes(np.meshgrid(x_range, y_range, z_range), 0, 3)
    neigh_i_grid = np.reshape(neigh_i_grid, (-1, 3))
    # Then the actual supercell cartesian shifts
    neigh_grid = np.dot(neigh_i_grid, latt_cart)

    return neigh_i_grid, neigh_grid


def minimum_periodic(v, latt_cart):
    """
    Find the shortest periodic equivalent vector for a list of vectors and a
    given lattice.

    | Args:
    |   v (np.ndarray): list of 3-vectors representing points or vectors to
    |                   reduce to their closest periodic version
    |   latt_cart (np.ndarray): unit cell in cartesian form

    | Returns:
    |   v_period (np.ndarray): array with the same shape as v, containing the
    |                          vectors in periodic reduced form
    |   v_cells (np.ndarray): array of triples of ints, corresponding to the 
    |                         cells from which the various periodic copies of
    |                         the vectors were taken. For an unchanged vector
    |                         will be all [0,0,0]

    """

    max_r = np.amax(np.linalg.norm(v, axis=-1))
    scell_shape = minimum_supcell(max_r, latt_cart)
    neigh_i_grid, neigh_grid = supcell_gridgen(latt_cart, scell_shape)
    v_period = np.array(v, copy=False)[:, None, :] + neigh_grid[None, :, :]
    min_copies = np.argmin(np.linalg.norm(v_period, axis=-1), axis=1)
    v_period = v_period[range(len(v)), min_copies, :]

    return v_period, neigh_i_grid[min_copies]


def is_string(s):
    """Checks whether s is a string, with Python 2 and 3 compatibility"""
    try:
        return isinstance(s, basestring)
    except NameError:
        # It must be Python 3!
        return isinstance(s, str)

# Inspecting arguments of a function, Python 2 and 3 way
import inspect
if hasattr(inspect, 'signature'):
    def inspect_args(f):
        fsig = inspect.signature(f)
        args = fsig.parameters
        nargs = len(args)
        nargs_def = len([p for p in args
                         if args[p].default != inspect.Signature.empty])
        return (nargs, nargs_def)
else:
    def inspect_args(f):
        argspec = inspect.getargspec(f)
        return (len(argspec.args),
                (0 if argspec.defaults is None else len(argspec.defaults)))


def list_distance(l1, l2):
    """Return an integer distance between two lists (number of differing
    elements)"""

    ldiff = list(l2[:])
    d = 0
    d += len(l1)-len([ldiff.remove(el) for el in l1 if el in ldiff])
    d += len(ldiff)
    return d


def swing_twist_decomp(quat, axis):
    """Perform a Swing*Twist decomposition of a Quaternion. This splits the
    quaternion in two: one containing the rotation around axis (Twist), the
    other containing the rotation around a vector parallel to axis (Swing).

    Returns two quaternions: Swing, Twist.
    """

    """
        vector3 ra( rotation.x, rotation.y, rotation.z ); // rotation axis
    vector3 p = projection( ra, direction ); // return projection v1 on to v2  (parallel component)
    twist.set( p.x, p.y, p.z, rotation.w );
    twist.normalize();
    swing = rotation * twist.conjugated();
    """

    # Current rotation axis
    ra = quat.q[1:]
    # Ensure that axis is normalised
    axis_norm = axis/np.linalg.norm(axis)
    # Projection of ra along the given axis
    p = np.dot(ra, axis_norm)*axis_norm
    # Create Twist
    qin = [quat.q[0], p[0], p[1], p[2]]
    twist = Quaternion(qin/np.linalg.norm(qin))
    # And Swing
    swing = quat*twist.conjugate()

    return swing, twist
