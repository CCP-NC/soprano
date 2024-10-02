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
Methods used to spawn or kill multiple daemons on a cluster machine.
"""


import shutil
import sys

from soprano.utils import import_module

# Spawn a Submitter instance from a given module
subm_module, subm_name = sys.argv[1:3]
loaded_module = import_module(subm_module)
subm = getattr(loaded_module, subm_name)
try:
    subm.start()
except Exception as e:
    # If ANYTHING goes wrong, at least clean up!
    for j in subm._jobs:
        subm.queue.kill(j)
        try:
            shutil.rmtree(subm._jobs[j]["folder"])
        except OSError:
            pass  # Whatever, it was deleted already I guess
    subm.log(
        "Submitter crashed following error:" f"\n{type(e).__name__}:\n\t{e}"
    )
