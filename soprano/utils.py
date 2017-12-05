# Soprano - a library to crack crystals! by Simone Sturniolo
# Copyright (C) 2016 - Science and Technology Facility Council

# Soprano is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Soprano is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
utils.py

Contains package-wide useful routines that don't fall under any specific
category. Many of these handle common operations involving periodicity,
conversions between different representations etc.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
from contextlib import contextmanager
import inspect
import numpy as np
from scipy.misc import factorial
from itertools import product as iter_product
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


@contextmanager
def silence_stdio(silence_stdout=True, silence_stderr=True):
    """Useful stdout/err silencer"""
    dummy_out = open(os.devnull, "w")
    if silence_stdout:
        old_stdout, sys.stdout = sys.stdout, dummy_out
    if silence_stderr:
        old_stderr, sys.stderr = sys.stderr, dummy_out
    try:
        yield dummy_out
    finally:
        if silence_stdout:
            sys.stdout = old_stdout
        if silence_stderr:
            sys.stderr = old_stderr


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


def minimum_periodic(v, latt_cart, exclude_self=False):
    """
    Find the shortest periodic equivalent vector for a list of vectors and a
    given lattice.

    | Args:
    |   v (np.ndarray): list of 3-vectors representing points or vectors to
    |                   reduce to their closest periodic version
    |   latt_cart (np.ndarray): unit cell in cartesian form
    |   exclude_self (bool): if True, any vector that is equal to zero will be
    |                        excluded, and its closest non-zero periodic
    |                        version will be considered instead. Default is
    |                        False

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
    v_norm = np.linalg.norm(v_period, axis=-1)
    if exclude_self:
        v_norm = np.where(v_norm > 0, v_norm, np.inf)
    min_copies = np.argmin(v_norm, axis=1)
    v_period = v_period[range(len(v)), min_copies, :]

    return v_period, neigh_i_grid[min_copies]


def all_periodic(v, latt_cart, max_r):
    """
    Find all the periodic equivalent vectors for a list of vectors and a
    given lattice falling within a given length.

    | Args:
    |   v (np.ndarray): list of 3-vectors representing points or vectors to
    |                   produce periodic versions of
    |   latt_cart (np.ndarray): unit cell in cartesian form
    |   max_r (float): maximum length of periodic copies of vectors

    | Returns:
    |   v_period (np.ndarray): array with the same shape as v, containing the
    |                          vectors in periodic reduced form
    |   v_index (np.ndarray): indices (referring to the original array v) of
    |                         the array of which the corresponding element of
    |                         v_period is a copy
    |   v_cells (np.ndarray): array of triples of ints, corresponding to the
    |                         cells from which the various periodic copies of
    |                         the vectors were taken. For an unchanged vector
    |                         will be all [0,0,0]

    """

    scell_shape = minimum_supcell(max_r, latt_cart)
    neigh_i_grid, neigh_grid = supcell_gridgen(latt_cart, scell_shape)
    v_period = np.array(v, copy=False)[:, None, :] + neigh_grid[None, :, :]
    r_copies = np.where(np.linalg.norm(v_period, axis=-1) <= max_r)
    v_period = v_period[r_copies[0], r_copies[1], :]

    return v_period, r_copies[0], neigh_i_grid[r_copies[1]]


def is_string(s):
    """Checks whether s is a string, with Python 2 and 3 compatibility"""
    try:
        return isinstance(s, basestring)
    except NameError:
        # It must be Python 3!
        return isinstance(s, str)


def safe_communicate(subproc, stdin=''):
    """Executes a Popen.communicate and returns output in a way that is
    compatible with Python 2 & 3 keeping input and output as strings (since
    Python 3 requires bytes objects otherwise)"""
    if not subproc.universal_newlines:
        stdin = stdin.encode('utf-8') if hasattr(stdin, 'encode') else stdin
    stdout, stderr = map(lambda x: x.decode() if hasattr(x, 'decode') else x,
                         subproc.communicate(stdin))

    return stdout, stderr

# Inspecting arguments of a function, Python 2 and 3 way
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


# Importing a module from filename with compatibility across Python versions

def import_module(mpath):

    mname = seedname(mpath)

    # Python 3.5+ version
    try:
        import importlib.util
    except ImportError:
        pass
    else:
        spec = importlib.util.spec_from_file_location(mname, mpath)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo

    # Python 3.3-3.4 version
    try:
        from importlib.machinery import SourceFileLoader
    except ImportError:
        pass
    else:
        return SourceFileLoader(mname, mpath).load_module()

    # Python 2 version
    try:
        import imp
    except ImportError:
        # Ok, what the hell?
        raise ImportError('Could not import file ' + mpath)
    else:
        return imp.load_source(mname, mpath)

    # We should never get here, but...
    return None


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

### Clebsch-Gordan and Wigner-3j symbols ###


def clebsch_gordan(j, m, j1, m1, j2, m2):
    """ Clebsch-Gordan cohefficients for given quantum numbers:
    j, m, j1, m1, j2, m2

    The numbers passed can be arrays (just make sure they're the
    same size).
    """

    # Is it a single value?
    try:
        n = len(j)
    except TypeError:
        n = 1

    # Safety checks
    jm_all = np.array([j, m, j1, m1, j2, m2]).T
    if n == 1:
        jm_all = jm_all[None, :]
    elif len(jm_all.shape) == 1:
        # Inconsistent lenghts!
        raise ValueError('Not all arrays passed have the same size')
    if (((jm_all*2) % 1 != 0).any() or
            (np.abs(jm_all[:, 1::2]) > jm_all[:, ::2]).any()):
        raise ValueError('Invalid momentum values')

    j, m, j1, m1, j2, m2 = jm_all.T

    # Gotta do this the hard way...
    # Find all valid k

    kmax = np.min([j1+j2-j,          # j1+j2-j-k >= 0
                   j1-m1,            # j1-m1-k >= 0
                   j2+m2],           # j2+m2-k >= 0
                  axis=0)
    kmin = np.max([j*0,              # k >= 0
                   -j+j2-m1,         # j-j2+m1+k >= 0
                   -j+j1+m2],        # j-j1+m2+k >= 0
                  axis=0)

    ks = [np.arange(np.ceil(k0), np.floor(k1)+1)
          for k0, k1 in zip(kmin, kmax)]

    # Calculate k sum
    ksum = [np.sum((-1)**k/(factorial(k)*factorial(j1[i]+j2[i]-j[i]-k) *
                            factorial(j1[i]-m1[i]-k)*factorial(j2[i]+m2[i]-k) *
                            factorial(j[i]-j2[i]+m1[i]+k) *
                            factorial(j[i]-j1[i]-m2[i]+k)))
            for i, k in enumerate(ks)]

    cg = np.where(m == m1+m2,
                  np.sqrt((2*j+1)*factorial(j+j1-j2)*factorial(j-j1+j2) *
                          factorial(j1+j2-j)/factorial(j1+j2+j+1)) *
                  np.sqrt(factorial(j+m)*factorial(j-m)*factorial(j1-m1) *
                          factorial(j1+m1)*factorial(j2-m2)*factorial(j2+m2)) *
                  ksum,
                  0)
    if n == 1:
        cg = cg[0]

    return cg


def wigner_3j(j1, m1, j2, m2, j3, m3):
    """ Wigner 3j symbols for given quantum numbers:

    /                 \
    | j1    j2     j3 |
    | m1    m2     m3 |
    \                 /

    The numbers passed can be arrays (just make sure they're the
    same size).
    """

    # Expressed as a function of Clebsch-Gordan cohefficients
    j1, m1, j2, m2, j3, m3 = np.array([j1, m1, j2, m2, j3, m3])
    return (clebsch_gordan(j3, -m3, j1, m1, j2, m2) *
            (-1)**((j1-j2-m3) % 2)/np.sqrt(2*j3+1))


def max_distance_in_cell(cell):
    """Maximum distance between two points achievable inside a single unit 
    cell.
    Relies on the fact that the isosurface for equal distance in cartesian
    space is an ellipsoid, meaning that the extreme has to be one of the 8
    corners of the cube of side 1 in fractional space.
    """

    corners = np.array(np.meshgrid(*[(-1, 1)]*3)).reshape((3, -1))
    # Upper distance bound in cell
    return np.amax(np.linalg.norm(np.dot(cell.T, corners), axis=0))


def periodic_bridson(cell, rmin, max_attempts=30,
                     prepoints=None, prepoints_cuts=None):
    """ Periodic version of the Bridson algorithm for generation of Poisson
    sphere distributions of points. This returns a generator.

    | Args:
    |   cell (np.ndarray): periodic cell in which to create the points.
    |   rmin (float): minimum distance between each generated point.
    |   max_attempts (int): maximum number of candidate neighbours generated
    |                       for each point.
    |   prepoints (np.ndarray or list): pre-existing points to avoid during
    |                                   generation. These must be in
    |                                   fractional coordinates.
    |   prepoints_cuts (np.ndarray or list): custom cutoffs for each prepoint.
    |                                        If not included defaults to rmin.

    | Returns:
    |   bridsonGen (generator): an iterator producing Poisson-sphere like
    |                           distributed points in cell until space runs
    |                           out or enough attempts fail.
    """

    # 1. Compute the necessary number of divisions for the cell's grid
    ubound = max_distance_in_cell(cell)
    N = int(np.ceil(ubound/rmin))

    # 2. Mask for finding existing points that are too close
    grid_cell = cell/N
    shape = minimum_supcell(rmin, grid_cell)
    # Is it valid?
    if (np.array(shape) > N).any():
        raise ValueError('Value of rmin is too big for this unit cell')
    checkMask = np.array(np.meshgrid(*[range(int((1-i)/2), int((i-1)/2+1))
                                       for i in shape],
                                     indexing='ij')).reshape((3, -1))
    checkMask = np.array([m for m in checkMask.T if not (m == 0).all()]).T
    # 2.5 Mask for searching neighbours
    shape = minimum_supcell(2*rmin, grid_cell)
    newMask = np.array(np.meshgrid(*[range(int((1-i)/2), int((i-1)/2+1))
                                     for i in shape],
                                   indexing='ij')).reshape((3, -1))
    newMask = np.array([m for m in newMask.T if not (m == 0).all()]).T

    # 3. Now generate the grid
    grid = np.zeros((N, N, N)).astype(int)
    grid_points = np.zeros((N, N, N, 3))
    # 3.5 if there are prepoints, fill in the grid cells that are too close to
    # begin with
    if prepoints is not None:
        if prepoints_cuts is None:
            prepoints_cuts = np.ones(prepoints.shape[0])*rmin
        else:
            prepoints_cuts = np.array(prepoints_cuts)
        # Create corner points for cells
        grid_origins = np.array(np.meshgrid(*[range(N)]*3,
                                            indexing='ij')).reshape((3, -1))
        cell_corners = np.array(np.meshgrid(*[[0, 1]]*3, indexing='ij')
                                ).reshape((3, -1))
        grid_corners = (grid_origins[:, :, None]+cell_corners[:, None, :])/N
        dfx = (prepoints[:, :, None, None] - grid_corners[None, :, :, :]
               + 0.5) % 1-0.5
        r = np.linalg.norm(np.tensordot(dfx, cell, axes=(1, 0)), axis=-1)
        # Which ones are actually fully taken?
        full = np.sum(np.prod(r < prepoints_cuts[:, None, None], axis=-1),
                      axis=0) > 0
        # And occupy the grid (2 means taken by an external point)
        grid[tuple(grid_origins[:, np.where(full)[0]])] = 2

    # 4. And the queue with a random, non-occupied grid point
    if prepoints is None:
        queue = [np.array([0, 0, 0])]
    else:
        free_ijk = np.array(np.where(grid == 0))
        queue = [free_ijk[:, np.random.randint(free_ijk.shape[1])]]

    # Start iterations
    while np.sum(grid != 0) < N**3 and len(queue) > 0:
        # While there is still free space and queued points...
        iter_ijk0 = queue.pop()
        iter_mask = (newMask+iter_ijk0[:, None]) % N
        attempts = max_attempts

        # Try to generate points around the dequeued cell
        while attempts > 0:
            attempts -= 1
            candidates = np.array(iter_mask[:,
                                            np.where(grid[tuple(iter_mask)] ==
                                                     0)][:, 0])
            good = False
            if candidates.shape[1] == 0:
                continue
            i = np.random.randint(candidates.shape[1])
            ijk = candidates[:, i]
            # Pick the point
            fp = (ijk+np.random.random(3))/N
            # Check it
            near_mask = (checkMask+ijk[:, None]) % N
            near_points = tuple(near_mask[:, np.where(grid[tuple(near_mask)]
                                                      == 1)])
            near_points = grid_points[near_points][0]
            # Distances?
            if near_points.shape[0] > 0:
                dfx = (near_points - fp[None, :]+0.5) % 1-0.5
                r = np.linalg.norm(np.dot(dfx, cell), axis=1)
                good = not (r < rmin).any()
            else:
                good = True
            # Additional check: verify against prepoints
            if good and prepoints is not None:
                dfx = (prepoints - fp[None, :]+0.5) % 1-0.5
                r = np.linalg.norm(np.dot(dfx, cell), axis=1)
                good = not (r < prepoints_cuts).any()
            if good:
                grid[tuple(ijk)] = 1
                grid_points[tuple(ijk)] = fp
                queue.insert(0, ijk)
                yield np.dot(fp, cell)

    # So once we're here we ran out of options...
    raise StopIteration('No more points can be generated')


######


def periodic_center(v_frac):
    # Apply an operation meant to find a center for a set of periodic points
    # and cancel out translational effects, allowing direct mapping
    """ Alright, how does this work?
    Basically, we're looking for the point, inside the unit cell, which
    minimizes the sum of the squared distance from all ions.
    Of course we need to consider the periodic boundaries. So the distance
    on a single axis isn't simply abs(x), but a triangular wave.
    Fun times! A triangular wave can be represented as a Fourier series. And
    we can truncate that series to the first term because the minimum
    basically stays the same and get:
    sum((x-x_i)**2) ~ sum((4/pi**2*sin(2*pi*(x-x_i-1/4))+0.5)**2)
    All the factors depend on the fact that we need to move the triangular
    wave to the interval [0,1] and center it so that it's 0 for x-x_i == 0.
    It gets better! We take the derivative of this thing and look for a spot
    where it becomes zero. The derivative is kind of a trigonometric
    monstrosity but we can solve the equation by setting t = e^(2*pi*1.0j*x)
    and then replacing cosines and sines with it. As a result, we get an
    equation of 4th degree in t. And then we solve that with numpy.roots, take
    the phase, turn that into a coordinate, find the one corresponding to the
    absolute minimum.
    All of which we can perform independently on each of the three axes
    because the function is just the sum of the three components:
    sum((r-r_i)**2) = sum((x-x_i)**2) + sum((y-y_i)**2) + sum((z-z_i)**2).
    And there you go! Problem solved.
    """

    # The distance function. Will be required to find the absolute minimum
    # later
    def distf(x, x_i):
        return np.sum((4/np.pi**2*np.sin(2*np.pi *
                                         (x[:, None]-x_i[None, :]-0.25)) +
                       0.5)**2, axis=-1)
    # These coefficients appear in the derivative
    sinS = np.sum(np.sin(2*np.pi*v_frac), axis=0)
    cosS = np.sum(np.cos(2*np.pi*v_frac), axis=0)
    sin2S = np.sum(np.sin(4*np.pi*v_frac), axis=0)
    cos2S = np.sum(np.cos(4*np.pi*v_frac), axis=0)

    # Here's the equation coefficients instead
    coeffs = np.array([32/np.pi**3*(sin2S+1.0j*cos2S),
                       8/np.pi*(sinS+1.0j*cosS),
                       0*sinS,
                       8/np.pi*(sinS-1.0j*cosS),
                       32/np.pi**3*(sin2S-1.0j*cos2S)])*0.5
    # Finding the roots, turning them to coordinates
    roots = [np.roots(c) for c in coeffs.T]
    roots = np.angle(roots)/(2*np.pi)
    # For each of them, find the minimum
    x_mins = []
    for a_i, axis in enumerate(roots):
        R2 = distf(axis, v_frac[:, a_i])
        R2min = np.amin(R2)
        x_mins.append(axis[np.where(np.isclose(R2, R2min))[0]])

    # How to decide if there are multiple minima?
    x_mins = np.array(np.meshgrid(*x_mins)).reshape((3, -1)).T
    if x_mins.shape[0] > 1:
        dpos = (x_mins[:, None]-v_frac[None, :]+0.5) % 1-0.5
        dposMag = np.sum(dpos**2, axis=-1)
        dpos_nn_i = np.argmin(dposMag, axis=-1)
        dpos_nn = dpos[range(8), dpos_nn_i].round(5)
        dpos_nn_M = dposMag[range(8), dpos_nn_i]
        x = x_mins[np.lexsort((-dpos_nn[:, 2],
                               -dpos_nn[:, 1],
                               -dpos_nn[:, 0],
                               dpos_nn_M))[0]]
    else:
        x = x_mins[0]

    return x
