#!/usr/bin/env python
"""
Utility functions for tests in the Soprano project
"""

import functools
import unittest
import warnings
import re
from typing import Callable

import ase


def get_ase_version():
    """Get the major and minor version numbers of ASE.
    
    Returns:
        tuple: A tuple containing (major, minor) version numbers.
    """
    version_match = re.match(r'(\d+)\.(\d+)', ase.__version__)
    if version_match:
        return tuple(map(int, version_match.groups()))
    return (0, 0)  # Default if parsing fails


def is_problematic_ase_version() -> bool:
    """Check if the installed ASE version is in the problematic range (3.23-3.25).
    
    Returns:
        bool: True if the ASE version is in the problematic range, False otherwise.
    """
    major, minor = get_ase_version()
    print(f"ASE version: {major}.{minor}")
    return major == 3 and 23 <= minor <= 25


def skip_if_problematic_ase(func: Callable) -> Callable:
    """Decorator to skip tests with problematic ASE versions.
    
    This decorator will skip the test if the ASE version is in the 
    problematic range (3.23-3.25).
    
    Args:
        func: The test function to decorate
    
    Returns:
        Callable: The wrapped function that will be skipped if needed
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_problematic_ase_version():
            warnings.warn(
                f"Test {func.__name__} skipped due to ASE version {ase.__version__} compatibility issue. "
                f"Known issue with ASE versions between 3.23 and 3.25."
            )
            raise unittest.SkipTest(f"Test skipped due to ASE version {ase.__version__} compatibility issue.")
        return func(*args, **kwargs)
    return wrapper
