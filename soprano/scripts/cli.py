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
__date__ = "July 04, 2022"


import click
import os
from soprano.scripts import  nmr, nmr_plot
from configparser import ConfigParser

epilog = f"""
    Author: {__author__} ({__email__})\n
    Last updated: {__date__}"""
help_text = """
A CLI tool to streamline common soprano tasks. It has various 
subcommands, each of which has its own set of options and help.
"""
# join home and config file
home = os.path.expanduser('~')
# get default soprano config file:
DEFAULT_CFG = os.environ.get('SOPRANO_CONFIG', f'{home}/.soprano/config.ini')

#callback to load config file
def configure(ctx, param, filename):
    cfg = ConfigParser()
    cfg.read(filename)
    ctx.default_map = {}
    for sect in cfg.sections():
        command_path = sect.split('.')
        if command_path[0] != 'soprano':
            continue
        defaults = ctx.default_map
        for cmdname in command_path[1:]:
            defaults = defaults.setdefault(cmdname, {})
        defaults.update(cfg[sect])

@click.group(
    name="Soprano Command Line Interface",
    help=help_text, epilog=epilog,
    invoke_without_command=True)

@click.option(
    '-c', '--config',
    type         = click.Path(dir_okay=False),
    default      = DEFAULT_CFG,
    callback     = configure,
    is_eager     = True,
    expose_value = False,
    show_default = True,
    help         = 'Read option defaults from the specified INI file'
                    'If not set, first checks environment variable: '
                    '``SOPRANO_CONFIG`` and then ``~/.soprano/config.ini``',
)
def soprano():
    pass

soprano.add_command(nmr.nmr)

if __name__ == '__main__':
    soprano()