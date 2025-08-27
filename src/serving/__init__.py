"""
Phase 6: Serving Layer Package

Simple recommendation API serving without MLflow complexity.
"""

from .recommendation_api import create_recommendation_api, RecommendationAPI
from .main import main

__all__ = ["create_recommendation_api", "RecommendationAPI", "main"]
