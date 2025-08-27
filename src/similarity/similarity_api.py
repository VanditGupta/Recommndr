"""
Similarity API for Phase 5

FastAPI endpoints for serving item-item similarity results:
- /similar_items/{item_id} - Get similar items for a given item
- /copurchase_analysis/{item_id} - Get co-purchase analysis  
- /similarity_stats - Get similarity engine statistics
- /market_basket_recommendations - Get recommendations based on current basket
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
import time
import logging

from .item_similarity import ItemSimilarityEngine
from .copurchase_similarity import CoPurchaseSimilarity
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SimilarItem(BaseModel):
    """Response model for similar item."""
    item_id: int
    similarity_score: float
    name: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    description: Optional[str] = None


class SimilarItemsResponse(BaseModel):
    """Response model for similar items endpoint."""
    query_item_id: int
    similar_items: List[SimilarItem]
    similarity_type: str
    response_time_ms: float
    total_found: int


class CoPurchaseAnalysis(BaseModel):
    """Response model for co-purchase analysis."""
    item_id: int
    item_frequency: int
    item_support: float
    top_cooccurring_items: List[Dict]
    analysis_type: str = "market_basket"


class MarketBasketRequest(BaseModel):
    """Request model for market basket recommendations."""
    current_basket: List[int]
    similarity_type: str = "jaccard"
    top_k: int = 10


class MarketBasketResponse(BaseModel):
    """Response model for market basket recommendations."""
    current_basket: List[int]
    recommendations: List[Dict]
    similarity_type: str
    response_time_ms: float


class SimilarityStats(BaseModel):
    """Response model for similarity statistics."""
    engine_status: str
    data_loaded: bool
    similarity_matrices_computed: List[str]
    index_built: bool
    stats: Dict


class SimilarityAPI:
    """FastAPI application for similarity serving."""
    
    def __init__(self):
        """Initialize similarity API."""
        self.app = FastAPI(
            title="Recommndr Similarity API",
            description="Phase 5: Item-Item Similarity Service",
            version="1.0.0"
        )
        
        # Initialize engines
        self.similarity_engine = ItemSimilarityEngine()
        self.copurchase_engine = CoPurchaseSimilarity()
        
        # Setup routes
        self._setup_routes()
        
        # Initialize flag
        self._initialized = False
        
        logger.info("🌐 Similarity API initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Load similarity data on startup."""
            try:
                logger.info("🚀 Starting Similarity API...")
                
                # Try to load pre-computed similarity data
                try:
                    self.similarity_engine.load_data()
                    self.similarity_engine.load_similarity_data()
                    logger.info("✅ Pre-computed similarity data loaded")
                    self._initialized = True
                except Exception as e:
                    logger.warning(f"Pre-computed data not found: {e}")
                    logger.info("🔄 Computing similarity data on startup...")
                    
                    # Compute similarity data if not available
                    self.similarity_engine.run_similarity_pipeline()
                    self._initialized = True
                    
            except Exception as e:
                logger.error(f"Failed to initialize similarity API: {e}")
                self._initialized = False
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "service": "Recommndr Similarity API",
                "version": "1.0.0",
                "phase": 5,
                "description": "Item-item similarity service using ALS embeddings and co-purchase patterns",
                "initialized": self._initialized
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy" if self._initialized else "initializing",
                "timestamp": time.time(),
                "initialized": self._initialized
            }
        
        @self.app.get("/similar_items/{item_id}", response_model=SimilarItemsResponse)
        async def get_similar_items(
            item_id: int,
            top_k: int = Query(10, ge=1, le=50, description="Number of similar items to return"),
            similarity_type: str = Query("hybrid", regex="^(als|copurchase|hybrid)$", description="Type of similarity to use"),
            include_metadata: bool = Query(True, description="Include product metadata in response")
        ):
            """Get similar items for a given item ID."""
            if not self._initialized:
                raise HTTPException(status_code=503, detail="Similarity engine not initialized")
            
            start_time = time.time()
            
            try:
                # Ensure the correct similarity type is used in the index
                if (similarity_type == "als" and self.similarity_engine.als_similarity_matrix is None) or \
                   (similarity_type == "copurchase" and self.similarity_engine.copurchase_similarity_matrix is None) or \
                   (similarity_type == "hybrid" and self.similarity_engine.hybrid_similarity_matrix is None):
                    
                    # Rebuild index with requested similarity type
                    logger.info(f"Rebuilding similarity index with {similarity_type} similarity...")
                    self.similarity_engine.build_similarity_index(similarity_type, top_k=50)
                
                # Get similar items
                similar_items = self.similarity_engine.get_similar_items(
                    item_id=item_id,
                    top_k=top_k,
                    include_metadata=include_metadata
                )
                
                if not similar_items:
                    raise HTTPException(status_code=404, detail=f"No similar items found for item {item_id}")
                
                response_time = (time.time() - start_time) * 1000
                
                # Convert to response model
                similar_items_response = [
                    SimilarItem(**item) for item in similar_items
                ]
                
                return SimilarItemsResponse(
                    query_item_id=item_id,
                    similar_items=similar_items_response,
                    similarity_type=similarity_type,
                    response_time_ms=response_time,
                    total_found=len(similar_items)
                )
                
            except Exception as e:
                logger.error(f"Error getting similar items for {item_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/copurchase_analysis/{item_id}", response_model=CoPurchaseAnalysis)
        async def get_copurchase_analysis(
            item_id: int,
            top_k: int = Query(5, ge=1, le=20, description="Number of co-occurring items to analyze")
        ):
            """Get co-purchase analysis for a given item."""
            if not self._initialized:
                raise HTTPException(status_code=503, detail="Similarity engine not initialized")
            
            try:
                # Initialize co-purchase engine if needed
                if self.copurchase_engine.interactions_df is None:
                    self.copurchase_engine.load_interactions(self.similarity_engine.interactions_df)
                    self.copurchase_engine.extract_user_baskets()
                    self.copurchase_engine.compute_item_frequencies()
                    self.copurchase_engine.compute_cooccurrence_matrix()
                
                # Get market basket analysis
                analysis = self.copurchase_engine.analyze_market_basket(item_id, top_k)
                
                if 'error' in analysis:
                    raise HTTPException(status_code=404, detail=analysis['error'])
                
                return CoPurchaseAnalysis(**analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing co-purchase for {item_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/market_basket_recommendations", response_model=MarketBasketResponse)
        async def get_market_basket_recommendations(request: MarketBasketRequest):
            """Get recommendations based on current market basket."""
            if not self._initialized:
                raise HTTPException(status_code=503, detail="Similarity engine not initialized")
            
            start_time = time.time()
            
            try:
                # Initialize co-purchase engine if needed
                if self.copurchase_engine.interactions_df is None:
                    self.copurchase_engine.load_interactions(self.similarity_engine.interactions_df)
                    self.copurchase_engine.extract_user_baskets()
                    self.copurchase_engine.compute_item_frequencies()
                    self.copurchase_engine.compute_cooccurrence_matrix()
                
                # Get recommendations
                user_basket = set(request.current_basket)
                recommendations = self.copurchase_engine.get_recommendations_via_copurchase(
                    user_basket=user_basket,
                    top_k=request.top_k,
                    similarity_type=request.similarity_type
                )
                
                response_time = (time.time() - start_time) * 1000
                
                return MarketBasketResponse(
                    current_basket=request.current_basket,
                    recommendations=recommendations,
                    similarity_type=request.similarity_type,
                    response_time_ms=response_time
                )
                
            except Exception as e:
                logger.error(f"Error getting market basket recommendations: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/similarity_stats", response_model=SimilarityStats)
        async def get_similarity_stats():
            """Get similarity engine statistics."""
            try:
                stats = self.similarity_engine.get_similarity_stats()
                
                # Determine computed matrices
                computed_matrices = []
                if self.similarity_engine.als_similarity_matrix is not None:
                    computed_matrices.append("als")
                if self.similarity_engine.copurchase_similarity_matrix is not None:
                    computed_matrices.append("copurchase")
                if self.similarity_engine.hybrid_similarity_matrix is not None:
                    computed_matrices.append("hybrid")
                
                return SimilarityStats(
                    engine_status="initialized" if self._initialized else "not_initialized",
                    data_loaded=self.similarity_engine.item_embeddings is not None,
                    similarity_matrices_computed=computed_matrices,
                    index_built=self.similarity_engine.similarity_index is not None,
                    stats=stats
                )
                
            except Exception as e:
                logger.error(f"Error getting similarity stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/rebuild_similarity_index")
        async def rebuild_similarity_index(
            similarity_type: str = Query("hybrid", regex="^(als|copurchase|hybrid)$"),
            top_k: int = Query(50, ge=10, le=100)
        ):
            """Rebuild similarity index with specified parameters."""
            if not self._initialized:
                raise HTTPException(status_code=503, detail="Similarity engine not initialized")
            
            try:
                start_time = time.time()
                
                # Rebuild index
                self.similarity_engine.build_similarity_index(similarity_type, top_k)
                
                rebuild_time = (time.time() - start_time) * 1000
                
                return {
                    "status": "success",
                    "similarity_type": similarity_type,
                    "top_k": top_k,
                    "rebuild_time_ms": rebuild_time,
                    "items_indexed": len(self.similarity_engine.similarity_index)
                }
                
            except Exception as e:
                logger.error(f"Error rebuilding similarity index: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self.app


def create_similarity_api() -> FastAPI:
    """Create and return the similarity API application."""
    api = SimilarityAPI()
    return api.get_app()
