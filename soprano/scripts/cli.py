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

'''Click command line interface for Soprano.'''


__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "June 21, 2023"


import click
from soprano.scripts import  nmr, nmr_plot, dipolar
import logging
import click_log
# logging
logging.captureWarnings(True)
logger = logging.getLogger(__name__)
click_log.basic_config(logger)

epilog = f"""
    Author: {__author__} ({__email__})\n
    Last updated: {__date__}"""
help_text = """
A CLI tool to streamline common soprano tasks. It has various 
subcommands, each of which has its own set of options and help.
"""

@click.group(
    name="Soprano Command Line Interface",
    help=help_text, epilog=epilog,
    invoke_without_command=True)
    

@click_log.simple_verbosity_option(logger)

def soprano():
    pass

soprano.add_command(nmr.nmr)
soprano.add_command(nmr_plot.plotnmr)
soprano.add_command(dipolar.dipolar)

if __name__ == '__main__':
    soprano()