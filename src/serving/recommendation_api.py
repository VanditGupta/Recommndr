"""
Phase 6: Simple Recommendation API (No MLflow)

A lightweight FastAPI serving layer that loads models directly from local files.
Provides clean endpoints for the frontend to consume.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pickle
import time
import logging
from pathlib import Path

# Import our recommendation components
from src.retrieval.main import CandidateGenerationPipeline
from src.ranking.lightgbm_ranker import LightGBMRanker
from src.ranking.feature_engineering import RankingFeatureEngineer
from src.similarity.item_similarity import ItemSimilarityEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Any]]
    generation_time: float
    phase: str

class SimilarItemsRequest(BaseModel):
    item_id: int
    top_k: int = 10

class SimilarItemsResponse(BaseModel):
    item_id: int
    similar_items: List[Dict[str, Any]]
    query_time: float

class UserProfileResponse(BaseModel):
    user_id: int
    profile: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]

class RecommendationAPI:
    """Simple recommendation API without MLflow complexity."""
    
    def __init__(self):
        self.app = FastAPI(
            title="Recommndr API",
            description="Simple recommendation serving API",
            version="1.0.0"
        )
        
        # Initialize components
        self.candidate_generator = None
        self.ranker = None
        self.feature_engineer = None
        self.similarity_engine = None
        
        # Setup CORS for frontend
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins for development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        self._load_models()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Recommndr API v1.0.0",
                "status": "running",
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
                "models_loaded": {
                    "candidate_generator": self.candidate_generator is not None,
                    "ranker": self.ranker is not None,
                    "feature_engineer": self.feature_engineer is not None,
                    "similarity_engine": self.similarity_engine is not None
                }
            }
        
        @self.app.post("/recommend", response_model=RecommendationResponse)
        async def get_recommendations(request: RecommendationRequest):
            """Get personalized recommendations for a user."""
            try:
                # Ensure models are loaded
                self._ensure_models_loaded()
                
                start_time = time.time()
                
                # Generate candidates using Phase 3
                candidates = self.candidate_generator.generate_candidates(
                    request.user_id, 
                    top_k=50
                )
                
                if not candidates:
                    raise HTTPException(status_code=404, detail="No candidates found for user")
                
                # Rank candidates using Phase 4
                ranked_recommendations = self.ranker.rank_candidates(
                    request.user_id,
                    candidates,
                    top_k=request.top_k
                )
                
                generation_time = time.time() - start_time
                
                return RecommendationResponse(
                    user_id=request.user_id,
                    recommendations=ranked_recommendations,
                    generation_time=generation_time,
                    phase="Phase 4 (ALS + LightGBM)"
                )
                
            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/recommend/{user_id}", response_model=RecommendationResponse)
        async def get_recommendations_get(user_id: int, top_k: int = 10):
            """Get recommendations via GET request."""
            request = RecommendationRequest(user_id=user_id, top_k=top_k)
            return await get_recommendations(request)
        
        @self.app.get("/similar_items/{item_id}", response_model=SimilarItemsResponse)
        async def get_similar_items(item_id: int, top_k: int = 10):
            """Get similar items for a given item."""
            try:
                # Ensure models are loaded
                self._ensure_models_loaded()
                
                start_time = time.time()
                
                similar_items = self.similarity_engine.get_similar_items(
                    item_id, 
                    top_k=top_k,
                    similarity_type="hybrid"
                )
                
                query_time = time.time() - start_time
                
                return SimilarItemsResponse(
                    item_id=item_id,
                    similar_items=similar_items,
                    query_time=query_time
                )
                
            except Exception as e:
                logger.error(f"Error finding similar items: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/user_profile/{user_id}", response_model=UserProfileResponse)
        async def get_user_profile(user_id: int):
            """Get user profile and interaction history."""
            try:
                # Ensure models are loaded
                self._ensure_models_loaded()
                
                # Get user profile
                user_profile = self.feature_engineer.get_user_profile(user_id)
                
                # Get interaction history
                interaction_history = self.feature_engineer.get_user_interactions(user_id)
                
                return UserProfileResponse(
                    user_id=user_id,
                    profile=user_profile,
                    interaction_history=interaction_history
                )
                
            except Exception as e:
                logger.error(f"Error getting user profile: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _load_models(self):
        """Load models lazily - only initialize the pipeline objects."""
        logger.info("🔄 Initializing serving layer components...")
        
        try:
            # Just initialize the pipeline objects, don't load heavy models yet
            logger.info("   📥 Initializing Phase 3 (Candidate Generation)...")
            self.candidate_generator = CandidateGenerationPipeline()
            self.candidate_generator.load_data()
            logger.info("   ✅ Candidate generation pipeline initialized")
            
            # Initialize other components
            logger.info("   📥 Initializing Phase 4 (Ranking)...")
            self.ranker = LightGBMRanker()
            logger.info("   ✅ Ranking pipeline initialized")
            
            logger.info("   📥 Initializing Phase 5 (Feature Engineering)...")
            self.feature_engineer = RankingFeatureEngineer()
            logger.info("   ✅ Feature engineering pipeline initialized")
            
            logger.info("   📥 Initializing Phase 5 (Similarity)...")
            self.similarity_engine = ItemSimilarityEngine()
            logger.info("   ✅ Similarity pipeline initialized")
            
            logger.info("✅ All components initialized successfully!")
            logger.info("⚠️ Models will be loaded on first request (lazy loading)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            logger.warning("⚠️ Some components failed to initialize")
    
    def _ensure_models_loaded(self):
        """Ensure models are loaded before serving requests."""
        try:
            # Load ALS model if not loaded
            if not hasattr(self.candidate_generator, 'als_model') or self.candidate_generator.als_model is None:
                als_model_path = Path("models/phase3/als_model.pkl")
                if als_model_path.exists():
                    with open(als_model_path, 'rb') as f:
                        self.candidate_generator.als_model = pickle.load(f)
                    logger.info("✅ ALS model loaded (lazy)")
            
            # Load Faiss index if not loaded
            if not hasattr(self.candidate_generator, 'faiss_search') or self.candidate_generator.faiss_search is None:
                faiss_index_path = Path("models/phase3/faiss_index.pkl")
                if faiss_index_path.exists():
                    with open(faiss_index_path, 'rb') as f:
                        self.candidate_generator.faiss_search = pickle.load(f)
                    logger.info("✅ Faiss index loaded (lazy)")
            
            # Load LightGBM model if not loaded
            if not hasattr(self.ranker, 'model') or self.ranker.model is None:
                model_path = Path("models/phase4/lightgbm_ranker.pkl")
                if model_path.exists():
                    self.ranker.load_model(str(model_path))
                    logger.info("✅ LightGBM model loaded (lazy)")
            
            # Load feature engineering data if not loaded
            if not hasattr(self.feature_engineer, '_base_features_loaded') or not self.feature_engineer._base_features_loaded:
                self.feature_engineer.load_base_features()
                logger.info("✅ Feature engineering data loaded (lazy)")
            
            # Load similarity data if not loaded
            if not hasattr(self.similarity_engine, '_similarity_data_loaded') or not self.similarity_engine._similarity_data_loaded:
                self.similarity_engine.load_similarity_data()
                logger.info("✅ Similarity data loaded (lazy)")
                
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            raise

def create_recommendation_api() -> FastAPI:
    """Create and return the recommendation API."""
    api = RecommendationAPI()
    return api.app
