"""Recommndr - Production-Grade E-Commerce Recommendation Pipeline."""

__version__ = "0.1.0"
__author__ = "Recommndr Team"
__email__ = "team@recommndr.com"

from . import data_generation
from . import validation
from . import processing
from . import utils

__all__ = [
    "data_generation",
    "validation", 
    "processing",
    "utils"
]
