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
Definition of the Collection class.

It handles multiple Atoms ASE objects and mirrors in this sense the structure
of the Atoms object itself.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import ase
import glob
import shutil
import inspect
import numpy as np
# 2-to-3 compatibility
try:
    import cPickle as pickle
except ImportError:
    import pickle
# More 2-to-3 compatibility, defining a wrapper function for inspect
# Internal imports
from ase import io as ase_io
from ase.build import niggli_reduce
from ase.calculators.singlepoint import SinglePointCalculator
from soprano import utils


class _AllCaller(object):

    """_AllCaller class.

    A meta-object that serves the purpose of calling a function on all members
    of a list in a natural way.
    """

    def __init__(self, all_list, all_class=None):
        """Initialize the AllCaller with an 'all' list"""
        if all_class is None:
            self._class = all_list[0].__class__
        else:
            self._class = all_class
        if not all([x.__class__ is self._class for x in all_list]):
            raise ValueError('Elements of list passed to an _AllCaller'
                             ' must be of the same type.')
        self._all = all_list
        # Now get a list of all members in common between these instances
        if len(self._all) > 0:
            self._instance_attrs = set.intersection(*[set(ins.__dict__.keys())
                                                      for ins in self._all])

    def __getattr__(self, name):
        """Here's the magic of the class - when a method isn't found belonging
        to it, go looking for it in its ._all list..."""

        if hasattr(self._class, name):
            attr = getattr(self._class, name)
            # Is it a function?
            if not hasattr(attr, '__call__'):
                return np.array([getattr(x, name) for x in self._all])

            def iterfunc(*args, **kwargs):
                return np.array([getattr(x, name)(*args, **kwargs)
                                 for x in self._all])
            return iterfunc
        elif name in self._instance_attrs:
            # It's an instance attribute
            return np.array([getattr(x, name) for x in self._all])
        else:
            raise AttributeError(('Not all \'{0}\' objects have attribute'
                                  ' \'{1}\'').format(self._class.__name__,
                                                     name))

    def map(self, f, *args, **kwargs):
        """Map a function to each element of the ._all list and return the
        results."""

        # First, check the signature
        if not hasattr(f, '__call__'):
            raise TypeError('Only functions can be mapped')

        nargs, nargs_def = utils.inspect_args(f)
        if nargs < 1 + len(args) + len(kwargs):
            # Function is invalid!
            raise ValueError('Invalid function passed to map')
        return [f(x, *args, **kwargs) for x in self._all]


class AtomsCollection(object):

    """AtomsCollection object.

    An AtomsCollection represents a group of ASE Atoms objects.
    It handles them together, can perform mass operations on them, and stores
    arrays of informations related to them.
    """

    def __init__(self, structures=[],
                 info={},
                 cell_reduce=False,
                 progress=False, suppress_ase_warnings=True):
        """
        Initialize the AtomsCollection

        | Args:
        |    structures (list[str] or list[ase.Atoms]): list of file names or
        |                                               Atoms that will form
        |                                               the collection
        |    info (dict): dictionary of general information to attach
        |                 to this collection
        |    cell_reduce (bool): if True, perform a Niggli cell reduction on
        |                        all loaded structures
        |    progress (bool): visualize a progress bar for the loading process
        |    suppress_ase_warnings (bool): suppress annoying ASE warnings when
        |                                  loading files (default is True)
        """

        # Start by parsing out the structures
        self.structures = []

        if isinstance(structures, ase.Atoms):
            # Well, it's just one...
            structures = [structures]
        elif inspect.isgenerator(structures):
            # Let's unravel it
            iter_structs = structures
            structures = []
            for s in iter_structs:
                structures.append(s)

        if progress:
            sys.stdout.write("Loading collection...\n")
        s_n = len(structures)
        for s_i, struct in enumerate(structures):
            if progress:
                # Progress bar
                sys.stdout.write("\rLoading: {0}".format(utils.progbar(s_i+1,
                                                                       s_n)))
            # Is it an Atoms object?
            if type(struct) is ase.Atoms:
                self.structures.append(ase.Atoms(struct))
                # Copy all arrays
                for k in struct.arrays.keys():
                    if not self.structures[-1].has(k):
                        self.structures[-1].new_array(k, struct.get_array(k))
                if struct.calc is not None:
                    # Prevents pointless attempts at re-calculating
                    self.structures[-1].calc._old_atoms = self.structures[-1]
            # Or is it a string?
            elif utils.is_string(struct):
                with utils.silence_stdio(suppress_ase_warnings,
                                         suppress_ase_warnings):
                    self.structures.append(ase_io.read(str(struct)))
                # If there's no name, give it the filename
                if 'name' not in self.structures[-1].info:
                    self.structures[-1].info['name'] = utils.seedname(struct)
            else:
                raise TypeError('Structures must be Atoms objects or valid '
                                'file names,'
                                ' not {0}'.format(type(struct).__name__))
            if cell_reduce:
                # Here we must keep the energy if it was present
                # We do this by hand because ASE has its good reasons
                # for severing the atoms-calculator connection when changing
                # the unit cell.
                try:
                    _E = self.structures[-1].calc.results['energy']
                except (KeyError, AttributeError):
                    _E = None
                niggli_reduce(self.structures[-1])
                if _E is not None:
                    _calc = SinglePointCalculator(self.structures[-1],
                                                  energy=_E)
                    self.structures[-1].set_calculator(_calc)

        if progress:
            sys.stdout.write('\nLoaded {0} structures\n'.format(s_n))

        self._all = _AllCaller(self.structures, ase.Atoms)

        self._arrays = {}

        # Now assign the info
        if type(info) is not dict:
            raise TypeError('Info must be dict,'
                            ' not {0}'.format(type(info).__name__))
        else:
            self.info = info.copy()

    def __add__(self, other):
        """Addition of two collections brings a merging"""

        if not isinstance(other, self.__class__):
            raise TypeError('\'AtomsCollection\' does not support operator +'
                            ' with object of type'
                            ' \'{0}\''.format(type(other).__name__))

        # Create a common collection, join arrays where present, copy all info
        all_struct = list(self.structures)
        all_struct += list(other.structures)

        all_info = dict(self.info)
        for k in other.info:
            if k not in all_info:
                all_info[k] = other.info[k]
            else:
                all_info.pop(k)

        sum_struct = AtomsCollection(all_struct, all_info)

        # Now arrays
        all_arrays = {}
        for aname in self._arrays:
            # Grab the shape
            shape = self._arrays[aname].shape[1:]
            # Check if present in the other as well
            all_arr = list(self.get_array(aname))
            if aname in other._arrays:
                all_arr += list(other.get_array(aname))
            else:
                all_arr += [np.zeros(shape)*np.nan]*other.length
            sum_struct.set_array(aname, all_arr, shape=shape)

        for aname in other._arrays:
            # Can not be present in both
            if aname in self._arrays:
                continue
            # Grab the shape
            shape = other._arrays[aname].shape[1:]
            all_arr = list(other.get_array(aname))
            all_arr = [np.zeros(shape)]*self.length + all_arr
            sum_struct.set_array(aname, all_arr, shape=shape)

        return sum_struct

    def __iadd__(self, other):
        self = self + other
        return self

    def __getitem__(self, indices):
        """Allow sophisticated slicing"""

        if type(indices) is int:
            # Special case, a single element!
            indices = indices%len(self)
            indices = slice(indices, indices+1)

        try:
            struct_slice = self.structures[indices]
        except TypeError:
            indices = np.array(indices)
            if indices.dtype == bool:
                indices = np.where(indices)[0]  # Support for bool arrays
            struct_slice = [self.structures[i] for i in indices]

        sliced = AtomsCollection(struct_slice, info=self.info)

        # Now to add the arrays
        for a in self._arrays:
            sliced.set_array(a, self._arrays[a][indices])

        return sliced

    def __iter__(self):
        return self.structures.__iter__()

    def __deepcopy__(self, memodict={}):
        """Protects against problems with infinite recursion in AllCaller"""

        dcopy = AtomsCollection(self.structures, self.info)
        for arr in self._arrays:
            dcopy.set_array(arr, self._arrays[arr])

        return dcopy

    def __len__(self):
        return self.length

    @property
    def length(self):
        return len(self.structures)

    @property
    def all(self):
        return self._all

    def set_array(self, name, a, dtype=None, shape=None, args={}):
        """Add or modify an array of data related to the Atoms objects
        in this collection.

        | Args:
        |   name (str): name of the array to operate on.
        |   a (np.ndarray or function<Atoms, \*\*kwargs>
        |                    => Any): the data to assign to the array (must
        |                             be same length as the collection) or
        |                             a function that takes an Atoms object
        |                             as the first argument and returns a
        |                             value. This will be mapped over the
        |                             structures to create the array.
        |   dtype (type): type to cast the values of the array to.
        |   shape (tuple [int]): shape of each entry of the array. Will be
        |                        checked if provided.
        |   args (dict): named arguments to pass to the function provided
        |                as a. Will be ignored if an array is passed instead.

        """

        # a can be an actual array or a function that operates on each
        # separate Atoms object and returns a value

        a = np.array(a, dtype)
        if a.shape == ():
            a = a.item()
            if hasattr(a, '__call__'):
                # It's a function
                a = np.array(self.all.map(a, **args), dtype)
            else:
                # Invalid
                raise TypeError('new_array requires to pass either an array'
                                ' or a function taking an Atoms object as its'
                                ' first argument and returning a value.')

        # Now check that the shape is valid
        if shape is not None:
            if shape == (1,):
                targ_shape = (self.length,)
            else:
                targ_shape = (self.length,) + shape
            if a.shape != targ_shape:
                raise ValueError('Array of invalid shape passed to new_array')
        else:
            if a.shape[0] != self.length:
                raise ValueError('Array passed to new_array should be'
                                 ' as long as the number of structures')

        # And finally, assign
        self._arrays[name] = a

    def get_array(self, name, copy=True):
        """Get a copy of an array of given name (or a reference if copy=False)

        | Args:
        |   name (str): name of the array to retrieve.
        |   copy (bool): if the array should be copied or a reference should
        |                be returned instead.

        | Returns:
        |   array (np.ndarray): the requested array

        """

        if name not in self._arrays:
            raise ValueError('Array \'{0}\' does not exist'.format(name))
        else:
            return np.array(self._arrays[name], copy=copy)

    def has(self, name):
        """Check if array of given name exists"""

        return name in self._arrays

    def set_calculators(self, calctype, labels=None, params={}):
        """Set an ASE calculator on each structure in the collection,
        and set said calculator's parameters.

        | Args:
        |   calctype (ASE Calculator type): the type of calculator
        |                                   to instantiate.
        |   labels (Optional[list[str]]): names to use for the calculators'
        |                                 files. If not present, random
        |                                 generated names are used.
        |   params (Optional[dict]): parameters of the calculator to set.

        """

        # First, a check
        from ase.calculators.general import Calculator as gCalculator
        from ase.calculators.calculator import Calculator as cCalculator
        from ase.calculators.calculator import FileIOCalculator as ioCalculator

        if (gCalculator not in calctype.__bases__) and \
           (cCalculator not in calctype.__bases__) and \
           (ioCalculator not in calctype.__bases__):
            raise TypeError('calctype must be a type of ASE Calculator')

        if labels is not None and len(labels) != self.length:
            raise ValueError('labels must be long as the collection itself')

        # Then set it up
        for i, s in enumerate(self.structures):
            if labels is None:
                # First: do we have a name?
                if 'name' in s.info:
                    label = s.info['name']
                else:
                    label = 'struct_{0}'.format(i)
            else:
                label = labels[i]
            calc = calctype(atoms=s,
                            label=str(label),
                            **params)
            # To make sure...
            s.set_calculator(calc)

    def run_calculators(self, properties=None, system_changes=None):
        """Run all previously set ASE calculators.

        | Args:
        |   properties (list[str]): list of properties to calculate (depends
        |                           on type of Calculator used)
        |   system_changes (list[str]): list of changes to the structure
        |                               since the last calculation. Can be
        |                               any combination of these five:
        |                               'positions', 'numbers', 'cell',
        |                               'pbc', 'initial_charges' and
        |                               'initial_magmoms'.

        """

        # First, check if we even have those
        if any([c is None for c in self.all.calc]):
            raise RuntimeError('Not all structures in collection'
                               ' have a calculator')
        kwargs = {}
        if properties is not None:
            kwargs['properties'] = properties
        if system_changes is not None:
            kwargs['system_changes'] = system_changes
        self.all.map(lambda s: s.calc.calculate(atoms=s, **kwargs))

    def chunkify(self, chunk_size=None, chunk_n=None):
        """Split this collection into multiple collections based on either
        size or number of chunks.

        | Args:
        |   chunk_size (Optional[int]): maximum size of a generated chunk
        |   chunk_n (Optional[int]): number of chunks to generate

        | Returns:
        |   chunks (list[AtomsCollection]): a list of the generated chunks

        """

        if [chunk_size, chunk_n].count(None) != 1:
            raise RuntimeError('Only one between chunk_size and chunk_n'
                               'must be passed')

        if chunk_size is not None and type(chunk_size) is not int:
            raise TypeError('chunk_size must be an int')
        if chunk_n is not None and type(chunk_n) is not int:
            raise TypeError('chunk_n must be an int')

        # Now determine the size
        if chunk_size is None:
            chunk_size = int(np.ceil(self.length*1.0/chunk_n))

        struct_chunks = [self.structures[i:i+chunk_size]
                         for i in range(0, self.length, chunk_size)]
        chunks = [AtomsCollection(s, self.info) for s in struct_chunks]

        # Now the arrays
        for aname in self._arrays:
            for i, c in enumerate(chunks):
                c.set_array(aname, self._arrays[aname][i*chunk_size:
                                                       i*(chunk_size+1)])

        return chunks

    def sorted_byarray(self, name, reverse=False):
        """Return a copy of this collection sorted by a given array.

        | Args:
        |   name (str): name of the array to use for the sorting
        |   reverse (Optional[bool]): reverse order of sorting (max to min)

        | Returns:
        |   sorted (AtomsCollection): a sorted copy of the collection

        """

        # First, check that we do have the array
        if not self.has(name):
            raise ValueError("Array \'{0}\' does not exist".format(name))

        arr_names = list(self._arrays.keys())
        data_block = zip(self.structures,
                         *[self._arrays[n] for n in arr_names])
        key_i = arr_names.index(name)
        data_block = list(zip(*sorted(data_block,
                                      key=lambda x: x[key_i+1],
                                      reverse=reverse)))

        sorted_copy = AtomsCollection(data_block[0], info=self.info)
        for ai, an in enumerate(arr_names):
            sorted_copy.set_array(an, data_block[ai+1])

        return sorted_copy

    def filter(self, filter_func):
        """Return a collection composed only of the elements for which a given
        filter function returns True.

        | Args:
        |   filter_func (function<Atoms>
        |                       => bool): filter function. Should take an
        |                                 Atoms object and return a boolean

        | Returns:
        |   filtered (AtomsCollection): the filtered version of the collection

        """

        filter_slice = []

        for i, s in enumerate(self.structures):
            if (filter_func(s)):
                filter_slice.append(i)

        return self[filter_slice]

    def classify(self, classes):
        """Return a dictionary of collections based on the names of assigned
        classes.

        | Args:
        |   classes (np.ndarray): array of the class to which each structure
        |                         belongs. For example [1, 2, 1] will put the
        |                         first and third structures in class 1 and
        |                         the other in class 2. The classes can be any
        |                         hashable types, like int or str.

        | Returns:
        |   classified (dict): a dictionary using class names as keys and 
        |                      sliced collections as values

        """

        classes = np.array(classes)
        classified = {k: self[np.where(classes == k)[0]]
                      for k in set(classes)}

        return classified

    def save(self, filename):
        """Simply save a pickled copy to a given file path"""

        # Pickling doesn't deal well with the _AllCaller, so we get rid of it
        selfcopy = self[:]
        selfcopy._all = None

        with open(filename, 'wb') as f:
            pickle.dump(selfcopy, f, protocol=2)

    @staticmethod
    def load(filename):
        """Load a pickled copy from a given file path"""

        with open(filename, 'rb') as f:
            coll = pickle.load(f)

        if not isinstance(coll, AtomsCollection):
            raise ValueError('File does not contain an AtomsCollection'
                             ' object')
        # Restore the _AllCaller
        coll._all = _AllCaller(coll.structures, ase.Atoms)
        return coll

    @staticmethod
    def check_tree(path):
        """Checks if a path is a valid 'tree' format for a collection. This is
        any folder that satisfies the following conditions:

        - contains a .collection file storing metadata
        - contains a series of folders matching the list stored in the
          .collection file, and nothing else

        This function will return 0 if both conditions are satisfied, 1 if 
        only the first is, 2 if no .collection file is found, and -1 if the
        folder itself doesn't exist.

        | Args: 
        |   path (str): path to check for whether it matches or not the 
        |               collection pattern

        | Returns:
        |   result (int): 0, 1 or 2 depending on the outcome of the checks
        """

        if not os.path.exists(path):
            return -1

        # Begin by checking whether there is a .collection file
        try:
            with open(os.path.join(path, '.collection'), 'rb') as f:
                coll = pickle.load(f)
        except (IOError, UnicodeDecodeError):
            return 2  # No or invalid .collection file found

        # Check if the directories match
        dirlist = coll['dirlist']
        dirs = glob.glob(os.path.join(path, '*'))
        all_dirs = all([os.path.isdir(d) for d in dirs])
        dirs = [os.path.relpath(d, path) for d in dirs]

        # Are they even all directories?
        if (not all_dirs or set(dirlist) != set(dirs)):
            return 1

        return 0

    def save_tree(self, path, save_format, name_root='structure',
                  opt_args={}, safety_check=3):
        """Save the collection's structures as a series of folders, named like
        the structures, inside a given parent folder (that will be created if
        not present). Arrays and info are stored in a pickled .collection file
        which works as metadata for the whole directory tree.
        The files can be saved in a format of choice, or a function can be
        passed that will save them in a custom way. Only one collection can be
        saved per folder.

        | Args:
        |   path (str): folder path in which the collection should be saved.
        |   save_format (str or function): format in which the structures
        |                                  should be saved.
        |                                  If a string, it will be used as a
        |                                  file extension. If a function, it
        |                                  must take as arguments the
        |                                  structure (an ase.Atoms object)
        |                                  the save path (a string), and any
        |                                  additional arguments passed as
        |                                  opt_args, and take care of saving
        |                                  the required files.
        |   name_root (str): name prefix to be used for structures when a name
        |                    is not available in their info dictionary
        |   opt_args (dict): dictionary of additional arguments to pass to
        |                    either ase.io.write (if save_format is a string)
        |                    or to the save_format function.
        |   safety_check (int): how much care should be taken not to overwrite
        |                       potentially important data in path. Can be a
        |                       number from 0 to 3.
        |                       Here's the meaning of the codes:
        |
        |                       3 (default): always ask before overwriting an
        |                         existing folder that passes the check_tree
        |                         control, raise an exception otherwise;
        |                       2: overwite any folder that passes fully the
        |                          check_tree control, raise an exception
        |                          otherwise;
        |                       1: overwrite any folder that passes fully the
        |                          check_tree control, ask for user input
        |                          otherwise;
        |                       0 (DANGER - use at your own risk!): no checks,
        |                         always overwrite path.

        """

        check = AtomsCollection.check_tree(path)

        def ow_ask(path):
            return utils.safe_input(('Folder {0} exists, '
                                     'overwrite (y/n)?').format(path)
                                    ).lower() == 'y'

        if check > -1:
            # The folder exists
            if check == 0:
                if safety_check >= 3:
                    # Ask for permission
                    perm = ow_ask(path)
                else:
                    perm = True
            else:
                if safety_check >= 2:
                    raise IOError(('Trying to overwrite folder {0} which did'
                                   ' not pass check_tree control (result {1})'
                                   ' with safety_check level '
                                   '{2}').format(path,
                                                 check,
                                                 safety_check))
                elif safety_check == 1:
                    perm = ow_ask(path)
                else:
                    perm = True

            if not perm:
                print('Can not overwrite folder {0}, skipping...'.format(path))

            shutil.rmtree(path)

        # Re-create folder
        os.mkdir(path)

        # Format type?
        is_ext = utils.is_string(save_format)
        is_func = hasattr(save_format, '__call__')
        if not (is_ext or is_func):
            raise ValueError('Invalid save_format passed to save_tree')

        dirlist = []
        for i, s in enumerate(self.structures):
            sname = s.info.get('name', '{0}_{1}'.format(name_root, i+1))
            fold = os.path.join(path, sname)
            try:
                os.mkdir(fold)
            except OSError:
                shutil.rmtree(fold)
                os.mkdir(fold)
            if is_ext:
                ase_io.write(os.path.join(fold, sname + '.' + save_format), s,
                             **opt_args)
            elif is_func:
                save_format(s, fold, **opt_args)

            dirlist.append(sname)

        with open(os.path.join(path, '.collection'), 'wb') as f:
            pickle.dump({'dirlist': dirlist,
                         'arrays': self._arrays,
                         'info': self.info}, f,
                        protocol=2)

    @staticmethod
    def load_tree(path, load_format, opt_args={}, safety_check=3):
        """Load a collection's structures from a series of folders, named like
        the structures, inside a given parent folder, as created by save_tree.
        The files can be loaded from a format of choice, or a
        function can be passed that will load them in a custom way.

        | Args:
        |   path (str): folder path in which the collection should be saved.
        |   load_format (str or function): format from which the structures
        |                                  should be loaded.
        |                                  If a string, it will be used as a
        |                                  file extension. If a function, it
        |                                  must take as arguments the load
        |                                  path (a string) and any additional
        |                                  arguments passed as opt_args, and
        |                                  return the loaded structure as an
        |                                  ase.Atoms object.
        |   opt_args(dict): dictionary of additional arguments to pass to
        |                   either ase.io.read (if load_format is a string)
        |                   or to the load_format function.
        |   safety_check (int): how much care should be taken to verify the
        |                       folder that is being loaded. Can be a number
        |                       from 0 to 3.
        |                       Here's the meaning of the codes:
        |
        |                       3 (default): only load a folder if it passes
        |                         fully the check_tree control;
        |                       2: load any folder that has a valid
        |                          .collection file, but only the listed
        |                          subfolders;
        |                       1: load any folder that has a valid
        |                          .collection file, all subfolders. Array
        |                          data will be discarded;
        |                       0: no checks, try to load from all subfolders.

        | Returns:
        |   coll (AtomsCollection): loaded collection

        """

        check = AtomsCollection.check_tree(path)

        if check == -1:
            raise IOError('Folder {0} does not exist'.format(path))

        dirlist = []
        if check < 2:
            with open(os.path.join(path, '.collection'), 'rb') as f:
                coll = pickle.load(f)
            if check == 1 and safety_check == 3:
                raise IOError(('Folder {0} is not a valid collection '
                               'tree').format(path))
            if safety_check >= 2:
                dirlist = coll['dirlist']
            else:
                dirlist = [os.path.relpath(d, path)
                           for d in glob.glob(os.path.join(path, '*')) if
                           os.path.isdir(d)]
        else:
            if safety_check > 0:
                raise IOError(('Folder {0} is not a valid collection '
                               'tree').format(path))
            dirlist = [os.path.relpath(d, path)
                       for d in glob.glob(os.path.join(path, '*')) if
                       os.path.isdir(d)]

        # Format type?
        is_ext = utils.is_string(load_format)
        is_func = hasattr(load_format, '__call__')
        if not (is_ext or is_func):
            raise ValueError('Invalid load_format passed to load_tree')

        structures = []
        for d in dirlist:
            if is_ext:
                s = ase_io.read(os.path.join(path, d, d + '.' + load_format),
                                **opt_args)
            elif is_func:
                s = load_format(os.path.join(path, d), **opt_args)

            structures.append(s)

        if check < 2:
            info = coll['info']
        else:
            info = {}

        loaded_coll = AtomsCollection(structures, info=info)

        if safety_check >= 2:
            arrays = coll['arrays']
            for k, a in arrays.items():
                loaded_coll.set_array(k, a)

        return loaded_coll


if __name__ == '__main__':

    # Just load whatever was passed in the command line!

    if len(sys.argv) > 1:
        coll = AtomsCollection(sys.argv[1:], progress=True, cell_reduce=True)
