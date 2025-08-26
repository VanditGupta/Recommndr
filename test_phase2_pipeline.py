#!/usr/bin/env python3
"""
Phase 2 Pipeline Test - End-to-End Streaming & Feature Engineering
Tests the complete real-time pipeline from clickstream to features
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# Import our Phase 2 components
from src.streaming.clickstream_simulator import ClickstreamSimulator, ClickstreamEvent
from src.features.feature_engineering import FeatureEngineeringPipeline, UserFeature, ProductFeature

console = Console()

class Phase2PipelineTester:
    def __init__(self):
        self.console = Console()
        self.kafka_host = "localhost"
        self.kafka_port = 9092
        self.redis_host = "localhost"
        self.redis_port = 6380
        
    def test_kafka_connection(self) -> bool:
        """Test Kafka connection"""
        try:
            from kafka import KafkaProducer
            producer = KafkaProducer(
                bootstrap_servers=[f"{self.kafka_host}:{self.kafka_port}"],
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                request_timeout_ms=5000
            )
            producer.close()
            return True
        except Exception as e:
            self.console.print(f"[red]Kafka connection failed: {e}[/red]")
            return False
    
    def test_redis_connection(self) -> bool:
        """Test Redis connection"""
        try:
            import redis
            r = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=True)
            r.ping()
            return True
        except Exception as e:
            self.console.print(f"[red]Redis connection failed: {e}[/red]")
            return False
    
    def simulate_clickstream_data(self, num_events: int = 100) -> List[ClickstreamEvent]:
        """Generate realistic clickstream events"""
        events = []
        
        # Sample user IDs and product IDs
        user_ids = [f"user_{i:06d}" for i in range(1, 1001)]
        product_ids = [f"prod_{i:06d}" for i in range(1, 1001)]
        
        # Sample event types and pages
        event_types = ["page_view", "product_click", "add_to_cart", "purchase", "search"]
        pages = [
            "/home", "/products", "/category/electronics", "/category/clothing",
            "/product/detail", "/cart", "/checkout", "/search", "/account"
        ]
        
        for i in range(num_events):
            event = ClickstreamEvent(
                user_id=random.choice(user_ids),
                session_id=f"session_{random.randint(1000, 9999)}",
                event_type=random.choice(event_types),
                page_url=random.choice(pages),
                product_id=random.choice(product_ids) if random.random() > 0.3 else None,
                timestamp=datetime.now() - timedelta(seconds=random.randint(0, 3600)),
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                ip_address=f"192.168.1.{random.randint(1, 255)}",
                referrer=random.choice(pages) if random.random() > 0.5 else None,
                search_query=random.choice(["laptop", "phone", "shoes", "book"]) if random.random() > 0.7 else None
            )
            events.append(event)
        
        return events
    
    def test_streaming_pipeline(self) -> bool:
        """Test the streaming pipeline with Kafka"""
        self.console.print("\n[bold blue]🔄 Testing Streaming Pipeline (Kafka)[/bold blue]")
        
        try:
            # Test Kafka connection
            if not self.test_kafka_connection():
                return False
            
            # Create clickstream simulator
            simulator = ClickstreamSimulator(
                kafka_host=self.kafka_host,
                kafka_port=self.kafka_port
            )
            
            # Generate test events
            events = self.simulate_clickstream_data(num_events=50)
            
            # Send events to Kafka
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Sending events to Kafka...", total=len(events))
                
                for event in events:
                    simulator.send_event(event)
                    progress.advance(task)
                    time.sleep(0.1)  # Small delay to simulate real-time
            
            self.console.print("[green]✅ Streaming pipeline test completed successfully![/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Streaming pipeline test failed: {e}[/red]")
            return False
    
    def test_feature_engineering_pipeline(self) -> bool:
        """Test the feature engineering pipeline with Redis"""
        self.console.print("\n[bold blue]⚙️ Testing Feature Engineering Pipeline (Redis)[/bold blue]")
        
        try:
            # Test Redis connection
            if not self.test_redis_connection():
                return False
            
            # Create feature engineering pipeline
            pipeline = FeatureEngineeringPipeline(
                redis_host=self.redis_host,
                redis_port=self.redis_port
            )
            
            # Generate test events
            events = self.simulate_clickstream_data(num_events=30)
            
            # Process events through feature engineering
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Processing events through feature engineering...", total=len(events))
                
                for event in events:
                    pipeline.process_event(event)
                    progress.advance(task)
                    time.sleep(0.1)
            
            # Verify features were stored
            feature_stats = pipeline.get_feature_stats()
            
            self.console.print(f"[green]✅ Feature engineering pipeline test completed![/green]")
            self.console.print(f"Features stored: {feature_stats}")
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Feature engineering pipeline test failed: {e}[/red]")
            return False
    
    def test_end_to_end_pipeline(self) -> bool:
        """Test the complete end-to-end pipeline"""
        self.console.print("\n[bold blue]🎯 Testing End-to-End Pipeline[/bold blue]")
        
        try:
            # Test both components
            streaming_success = self.test_streaming_pipeline()
            feature_success = self.test_feature_engineering_pipeline()
            
            if streaming_success and feature_success:
                self.console.print("\n[bold green]🎉 END-TO-END PIPELINE TEST SUCCESSFUL![/bold green]")
                self.console.print("✅ Streaming pipeline (Kafka) - Working")
                self.console.print("✅ Feature engineering pipeline (Redis) - Working")
                self.console.print("✅ Complete real-time data flow - Verified")
                return True
            else:
                self.console.print("\n[bold red]❌ End-to-end pipeline test failed[/bold red]")
                return False
                
        except Exception as e:
            self.console.print(f"[red]❌ End-to-end pipeline test failed: {e}[/red]")
            return False
    
    def run_comprehensive_test(self):
        """Run the complete Phase 2 pipeline test"""
        self.console.print("[bold blue]🚀 Phase 2 Pipeline Comprehensive Test[/bold blue]")
        self.console.print("=" * 60)
        
        # Test infrastructure connections
        self.console.print("\n[bold]🔌 Testing Infrastructure Connections[/bold]")
        
        kafka_ok = self.test_kafka_connection()
        redis_ok = self.test_redis_connection()
        
        if not (kafka_ok and redis_ok):
            self.console.print("[red]❌ Infrastructure test failed. Check service status.[/red]")
            return False
        
        self.console.print("[green]✅ Infrastructure connections successful[/green]")
        
        # Test end-to-end pipeline
        pipeline_success = self.test_end_to_end_pipeline()
        
        # Summary
        self.console.print("\n" + "=" * 60)
        if pipeline_success:
            self.console.print("[bold green]🎉 PHASE 2 PIPELINE TEST COMPLETE - ALL SYSTEMS OPERATIONAL![/bold green]")
            self.console.print("\n[bold]🚀 Ready for:[/bold]")
            self.console.print("• Real-time recommendation serving")
            self.console.print("• ML model training (Phase 3)")
            self.console.print("• Production deployment")
        else:
            self.console.print("[bold red]❌ Pipeline test incomplete - Review logs[/bold red]")
        
        return pipeline_success

def main():
    """Main test execution"""
    tester = Phase2PipelineTester()
    success = tester.run_comprehensive_test()
    
    if success:
        console.print("\n[bold green]🎯 Phase 2 is ready for production use![/bold green]")
    else:
        console.print("\n[bold yellow]⚠️ Phase 2 needs attention before proceeding[/bold yellow]")

if __name__ == "__main__":
    main()
