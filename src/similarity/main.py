"""
Phase 5: Similarity Layer Main Pipeline

Main entry point for the similarity computation pipeline:
1. Computes ALS embedding-based similarity
2. Computes co-purchase pattern similarity  
3. Creates hybrid similarity combining both approaches
4. Builds fast similarity index for serving
5. Saves results for API serving

Usage:
    python -m src.similarity.main --compute-all
    python -m src.similarity.main --serve-api
    python -m src.similarity.main --test-similarity --item-id 123
"""

import argparse
import time
import asyncio
import uvicorn
from pathlib import Path
from typing import List, Dict, Optional

from .item_similarity import ItemSimilarityEngine
from .copurchase_similarity import CoPurchaseSimilarity
from .similarity_api import create_similarity_api
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SimilarityPipeline:
    """Main pipeline for Phase 5 similarity computation and serving."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        """Initialize similarity pipeline."""
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Initialize engines
        self.similarity_engine = ItemSimilarityEngine(data_dir, models_dir)
        self.copurchase_engine = CoPurchaseSimilarity()
        
        logger.info("🚀 Phase 5 Similarity Pipeline initialized")
    
    def compute_all_similarities(self, als_weight: float = 0.7, 
                               copurchase_weight: float = 0.3,
                               save_results: bool = True) -> Dict:
        """Compute all similarity types and build index."""
        logger.info("🎯 Computing all similarity types...")
        
        start_time = time.time()
        
        # Run complete similarity pipeline
        results = self.similarity_engine.run_similarity_pipeline(
            als_weight=als_weight,
            copurchase_weight=copurchase_weight,
            similarity_type="hybrid",
            top_k=50,
            save_results=save_results
        )
        
        total_time = time.time() - start_time
        
        summary = {
            'status': 'success',
            'total_time_seconds': total_time,
            'stats': self.similarity_engine.get_similarity_stats(),
            'models_saved': save_results
        }
        
        logger.info(f"✅ All similarities computed in {total_time:.2f}s")
        
        return summary
    
    def test_similarity_search(self, item_id: int, top_k: int = 10,
                             similarity_types: List[str] = None) -> Dict:
        """Test similarity search for a specific item."""
        if similarity_types is None:
            similarity_types = ["als", "copurchase", "hybrid"]
        
        logger.info(f"🔍 Testing similarity search for item {item_id}")
        
        # Load data if not already loaded
        if self.similarity_engine.item_embeddings is None:
            self.similarity_engine.load_data()
        
        # Load similarity data if available
        try:
            self.similarity_engine.load_similarity_data()
        except Exception as e:
            logger.warning(f"Pre-computed similarity not found: {e}")
            logger.info("Computing similarities on-demand...")
            self.compute_all_similarities()
        
        results = {'item_id': item_id, 'similarity_results': {}}
        
        # Test each similarity type
        for sim_type in similarity_types:
            try:
                logger.info(f"   Testing {sim_type} similarity...")
                
                # Rebuild index for this similarity type
                self.similarity_engine.build_similarity_index(sim_type, top_k=50)
                
                # Get similar items
                similar_items = self.similarity_engine.get_similar_items(
                    item_id=item_id,
                    top_k=top_k,
                    include_metadata=True
                )
                
                results['similarity_results'][sim_type] = {
                    'found_items': len(similar_items),
                    'similar_items': similar_items
                }
                
                logger.info(f"      Found {len(similar_items)} similar items")
                
            except Exception as e:
                logger.error(f"Error testing {sim_type} similarity: {e}")
                results['similarity_results'][sim_type] = {'error': str(e)}
        
        return results
    
    def test_copurchase_analysis(self, item_id: int, top_k: int = 5) -> Dict:
        """Test co-purchase analysis for a specific item."""
        logger.info(f"🛒 Testing co-purchase analysis for item {item_id}")
        
        # Load interaction data
        if self.similarity_engine.interactions_df is None:
            self.similarity_engine.load_data()
        
        # Initialize co-purchase engine
        self.copurchase_engine.load_interactions(self.similarity_engine.interactions_df)
        self.copurchase_engine.extract_user_baskets()
        self.copurchase_engine.compute_item_frequencies()
        self.copurchase_engine.compute_cooccurrence_matrix()
        
        # Perform analysis
        analysis = self.copurchase_engine.analyze_market_basket(item_id, top_k)
        
        logger.info(f"✅ Co-purchase analysis completed")
        
        return analysis
    
    def benchmark_similarity_performance(self, num_queries: int = 100) -> Dict:
        """Benchmark similarity search performance."""
        logger.info(f"⚡ Benchmarking similarity performance with {num_queries} queries...")
        
        # Ensure similarity engine is ready
        if self.similarity_engine.similarity_index is None:
            logger.info("Building similarity index for benchmarking...")
            self.similarity_engine.load_data()
            self.similarity_engine.load_similarity_data()
            self.similarity_engine.build_similarity_index("hybrid", top_k=50)
        
        # Get random items for testing
        import random
        test_items = random.sample(list(self.similarity_engine.item_mapping.keys()), 
                                 min(num_queries, len(self.similarity_engine.item_mapping)))
        
        # Benchmark similarity search
        start_time = time.time()
        
        total_results = 0
        for item_id in test_items:
            similar_items = self.similarity_engine.get_similar_items(
                item_id=item_id,
                top_k=10,
                include_metadata=False
            )
            total_results += len(similar_items)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_query_time = (total_time / num_queries) * 1000  # ms
        queries_per_second = num_queries / total_time
        avg_results_per_query = total_results / num_queries
        
        benchmark_results = {
            'num_queries': num_queries,
            'total_time_seconds': total_time,
            'avg_query_time_ms': avg_query_time,
            'queries_per_second': queries_per_second,
            'avg_results_per_query': avg_results_per_query,
            'total_results': total_results
        }
        
        logger.info("⚡ Benchmark Results:")
        logger.info(f"   Average query time: {avg_query_time:.2f}ms")
        logger.info(f"   Queries per second: {queries_per_second:.1f}")
        logger.info(f"   Average results per query: {avg_results_per_query:.1f}")
        
        return benchmark_results
    
    def serve_api(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Serve the similarity API."""
        logger.info(f"🌐 Starting Similarity API server on {host}:{port}")
        
        # Create FastAPI app
        app = create_similarity_api()
        
        # Run server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    
    def run_interactive_demo(self):
        """Run an interactive demo of the similarity system."""
        logger.info("🎭 Starting interactive similarity demo...")
        
        # Load data
        self.similarity_engine.load_data()
        
        try:
            self.similarity_engine.load_similarity_data()
            logger.info("✅ Loaded pre-computed similarity data")
        except:
            logger.info("Computing similarity data...")
            self.compute_all_similarities()
        
        print("\n🎯 Similarity Demo - Enter item IDs to find similar items")
        print("Available commands:")
        print("  - Enter an item ID (e.g., 123) to find similar items")
        print("  - 'stats' to show similarity statistics")
        print("  - 'random' to test with a random item")
        print("  - 'benchmark' to run performance benchmark")
        print("  - 'quit' to exit")
        
        while True:
            try:
                user_input = input("\n> ").strip().lower()
                
                if user_input == 'quit':
                    break
                elif user_input == 'stats':
                    stats = self.similarity_engine.get_similarity_stats()
                    print("\n📊 Similarity Engine Statistics:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                elif user_input == 'random':
                    import random
                    random_item = random.choice(list(self.similarity_engine.item_mapping.keys()))
                    results = self.test_similarity_search(random_item, top_k=5)
                    self._display_similarity_results(results)
                elif user_input == 'benchmark':
                    benchmark_results = self.benchmark_similarity_performance(100)
                    print("\n⚡ Benchmark Results:")
                    for key, value in benchmark_results.items():
                        print(f"   {key}: {value}")
                else:
                    try:
                        item_id = int(user_input)
                        results = self.test_similarity_search(item_id, top_k=5)
                        self._display_similarity_results(results)
                    except ValueError:
                        print("❌ Invalid input. Please enter an item ID or command.")
                        
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Demo ended!")
    
    def _display_similarity_results(self, results: Dict):
        """Display similarity search results in a formatted way."""
        item_id = results['item_id']
        similarity_results = results['similarity_results']
        
        print(f"\n🔍 Similar items for Item {item_id}:")
        print("=" * 80)
        
        for sim_type, data in similarity_results.items():
            if 'error' in data:
                print(f"\n❌ {sim_type.upper()} similarity: {data['error']}")
                continue
            
            similar_items = data['similar_items']
            print(f"\n📊 {sim_type.upper()} Similarity ({len(similar_items)} items):")
            print("-" * 60)
            
            for i, item in enumerate(similar_items, 1):
                name = item.get('name', f"Item {item['item_id']}")[:40]
                score = item['similarity_score']
                category = item.get('category', 'Unknown')
                price = item.get('price', 0)
                
                print(f"{i:2d}. {name:<40} | Score: {score:.3f}")
                print(f"    Category: {category:<20} | Price: ${price:.2f}")


def main():
    """Main entry point for Phase 5 similarity pipeline."""
    parser = argparse.ArgumentParser(description="Phase 5: Similarity Layer Pipeline")
    
    # Main actions
    parser.add_argument('--compute-all', action='store_true', help='Compute all similarity types')
    parser.add_argument('--serve-api', action='store_true', help='Serve similarity API')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    # Testing options
    parser.add_argument('--test-similarity', action='store_true', help='Test similarity search')
    parser.add_argument('--test-copurchase', action='store_true', help='Test co-purchase analysis')
    parser.add_argument('--item-id', type=int, help='Item ID for testing')
    
    # Configuration options
    parser.add_argument('--als-weight', type=float, default=0.7, help='Weight for ALS similarity')
    parser.add_argument('--copurchase-weight', type=float, default=0.3, help='Weight for co-purchase similarity')
    parser.add_argument('--top-k', type=int, default=10, help='Number of similar items to return')
    parser.add_argument('--port', type=int, default=8000, help='API server port')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='API server host')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SimilarityPipeline()
    
    if args.compute_all:
        logger.info("🎯 Computing all similarities...")
        results = pipeline.compute_all_similarities(
            als_weight=args.als_weight,
            copurchase_weight=args.copurchase_weight
        )
        print(f"\n✅ Similarity computation completed!")
        print(f"   Total time: {results['total_time_seconds']:.2f}s")
        print(f"   Results saved: {results['models_saved']}")
    
    if args.test_similarity:
        if not args.item_id:
            logger.error("--item-id required for similarity testing")
            return
        
        results = pipeline.test_similarity_search(args.item_id, args.top_k)
        pipeline._display_similarity_results(results)
    
    if args.test_copurchase:
        if not args.item_id:
            logger.error("--item-id required for co-purchase testing")
            return
        
        analysis = pipeline.test_copurchase_analysis(args.item_id, args.top_k)
        print(f"\n🛒 Co-purchase Analysis for Item {args.item_id}:")
        print(f"   Frequency: {analysis['item_frequency']}")
        print(f"   Support: {analysis['item_support']:.4f}")
        print(f"   Top co-occurring items: {len(analysis['top_cooccurring_items'])}")
    
    if args.benchmark:
        logger.info("⚡ Running performance benchmark...")
        results = pipeline.benchmark_similarity_performance()
        print(f"\n⚡ Benchmark completed!")
        print(f"   Average query time: {results['avg_query_time_ms']:.2f}ms")
        print(f"   Throughput: {results['queries_per_second']:.1f} queries/second")
    
    if args.demo:
        pipeline.run_interactive_demo()
    
    if args.serve_api:
        pipeline.serve_api(host=args.host, port=args.port)
    
    # Default action if no specific command given
    if not any([args.compute_all, args.serve_api, args.demo, args.benchmark, 
                args.test_similarity, args.test_copurchase]):
        logger.info("🎯 No specific action provided. Running compute-all...")
        results = pipeline.compute_all_similarities()
        print(f"\n✅ Phase 5 completed! Similarity data computed and saved.")
        print(f"   Run with --serve-api to start the API server")
        print(f"   Run with --demo for interactive testing")


if __name__ == "__main__":
    main()
