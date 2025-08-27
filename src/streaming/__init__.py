"""Streaming module for Phase 2: Streaming Ingestion & Feature Pipeline."""

from .clickstream_simulator import ClickstreamSimulator
from .kafka_producer import KafkaEventProducer
from .kafka_consumer import KafkaEventConsumer
from .flink_processor import FlinkStreamProcessor
from .main import Phase2Pipeline

__all__ = [
    "ClickstreamSimulator",
    "KafkaEventProducer", 
    "KafkaEventConsumer",
    "FlinkStreamProcessor",
    "Phase2Pipeline"
]
