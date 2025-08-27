"""
Phase 6: Simple Recommendation API (Minimal Version)

A very lightweight FastAPI that provides the structure without loading heavy models.
This allows us to move to frontend development quickly.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    generation_time: float
    phase: str
    note: str

class SimilarItemsResponse(BaseModel):
    item_id: int
    similar_items: List[Dict[str, Any]]
    query_time: float
    note: str

class UserProfileResponse(BaseModel):
    user_id: int
    profile: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    note: str

class SimpleRecommendationAPI:
    """Simple recommendation API without heavy model loading."""
    
    def __init__(self):
        self.app = FastAPI(
            title="Recommndr Simple API",
            description="Lightweight recommendation API for frontend development",
            version="1.0.0"
        )
        
        # Setup CORS for frontend
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Recommndr Simple API v1.0.0",
                "status": "running",
                "note": "This is a lightweight API for frontend development",
                "endpoints": [
                    "/recommend/{user_id}",
                    "/similar_items/{item_id}",
                    "/user_profile/{user_id}",
                    "/health"
                ]
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "note": "API is running but models are not loaded",
                "ready_for_frontend": True
            }
        
        @self.app.get("/recommend/{user_id}", response_model=RecommendationResponse)
        async def get_recommendations(user_id: int, top_k: int = 10):
            """Get mock recommendations for frontend development."""
            try:
                start_time = time.time()
                
                # Mock recommendations for frontend development
                mock_recommendations = [
                    {
                        "item_id": 1,
                        "name": "Professional Electronics Device",
                        "category": "Electronics",
                        "brand": "Premium",
                        "price": 299.99,
                        "rating": 4.5,
                        "ranking_score": 0.95,
                        "als_score": 0.85,
                        "lgb_score": 0.10
                    },
                    {
                        "item_id": 2,
                        "name": "Classic Clothing Item",
                        "category": "Clothing",
                        "brand": "Generic",
                        "price": 89.99,
                        "rating": 4.2,
                        "ranking_score": 0.88,
                        "als_score": 0.78,
                        "lgb_score": 0.10
                    },
                    {
                        "item_id": 3,
                        "name": "Modern Home & Garden Tool",
                        "category": "Home & Garden",
                        "brand": "Premium",
                        "price": 149.99,
                        "rating": 4.7,
                        "ranking_score": 0.82,
                        "als_score": 0.72,
                        "lgb_score": 0.10
                    }
                ]
                
                # Return only requested number
                recommendations = mock_recommendations[:top_k]
                generation_time = time.time() - start_time
                
                return RecommendationResponse(
                    user_id=user_id,
                    recommendations=recommendations,
                    generation_time=generation_time,
                    phase="Phase 4 (ALS + LightGBM) - Mock Data",
                    note="Using mock data for frontend development. Real models will be integrated later."
                )
                
            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/similar_items/{item_id}", response_model=SimilarItemsResponse)
        async def get_similar_items(item_id: int, top_k: int = 10):
            """Get mock similar items for frontend development."""
            try:
                start_time = time.time()
                
                # Mock similar items
                mock_similar_items = [
                    {
                        "item_id": 101,
                        "name": "Similar Electronics Device",
                        "category": "Electronics",
                        "price": 279.99,
                        "similarity_score": 0.89
                    },
                    {
                        "item_id": 102,
                        "name": "Related Electronics Accessory",
                        "category": "Electronics",
                        "price": 89.99,
                        "similarity_score": 0.76
                    },
                    {
                        "item_id": 103,
                        "name": "Electronics Component",
                        "category": "Electronics",
                        "price": 199.99,
                        "similarity_score": 0.72
                    }
                ]
                
                similar_items = mock_similar_items[:top_k]
                query_time = time.time() - start_time
                
                return SimilarItemsResponse(
                    item_id=item_id,
                    similar_items=similar_items,
                    query_time=query_time,
                    note="Using mock data for frontend development. Real similarity engine will be integrated later."
                )
                
            except Exception as e:
                logger.error(f"Error finding similar items: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/user_profile/{user_id}", response_model=UserProfileResponse)
        async def get_user_profile(user_id: int):
            """Get mock user profile for frontend development."""
            try:
                # Mock user profile
                mock_profile = {
                    "user_id": user_id,
                    "age": 25,
                    "gender": "other",
                    "location": "New York",
                    "income_level": "medium",
                    "device": "mobile",
                    "preferences": ["Electronics", "Clothing", "Books"]
                }
                
                # Mock interaction history
                mock_history = [
                    {
                        "item_id": 1,
                        "action": "click",
                        "timestamp": "2025-08-27T17:00:00Z",
                        "category": "Electronics"
                    },
                    {
                        "item_id": 2,
                        "action": "view",
                        "timestamp": "2025-08-27T16:45:00Z",
                        "category": "Clothing"
                    },
                    {
                        "item_id": 3,
                        "action": "add_to_cart",
                        "timestamp": "2025-08-27T16:30:00Z",
                        "category": "Home & Garden"
                    }
                ]
                
                return UserProfileResponse(
                    user_id=user_id,
                    profile=mock_profile,
                    interaction_history=mock_history,
                    note="Using mock data for frontend development. Real user data will be integrated later."
                )
                
            except Exception as e:
                logger.error(f"Error getting user profile: {e}")
                raise HTTPException(status_code=500, detail=str(e))

def create_simple_api() -> FastAPI:
    """Create and return the simple recommendation API."""
    api = SimpleRecommendationAPI()
    return api.app
