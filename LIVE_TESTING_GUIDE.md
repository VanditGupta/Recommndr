# 🚀 Live Clickstream Recommendation Testing Guide

This guide explains how to test your recommendation system with live clickstream data flowing through Kafka and Flink.

## 🏗️ Architecture Overview

```
User Events → Kafka → Flink Processing → Feature Store → Updated Recommendations
```

## 🧪 Testing Scripts

### 1. `test_kafka_recommendations.py` - Basic Integration Test
**Purpose**: Test Kafka event sending and basic integration
```bash
./venv/bin/python test_kafka_recommendations.py
```

**What it does**:
- ✅ Sends test events to Kafka
- ✅ Shows current recommendations for users
- ✅ Demonstrates event structure
- ✅ Tests streaming pipeline

### 2. `simulate_live_events.py` - Realistic Event Simulation
**Purpose**: Generate realistic user sessions and clickstream events
```bash
./venv/bin/python simulate_live_events.py
```

**What it does**:
- ✅ Generates realistic user sessions
- ✅ Sends events with proper timing
- ✅ Simulates different user behaviors
- ✅ Creates diverse event types (view, click, add_to_cart, purchase)

### 3. `test_live_recommendations.py` - End-to-End Live Testing
**Purpose**: Full end-to-end testing with real-time recommendation updates
```bash
./venv/bin/python test_live_recommendations.py
```

**What it does**:
- ✅ Tests complete live pipeline
- ✅ Shows before/after recommendations
- ✅ Monitors processing in real-time
- ✅ Analyzes recommendation changes

## 🔄 Testing Workflow

### Step 1: Verify Infrastructure
```bash
# Ensure all Docker services are running
docker-compose ps

# Check Kafka topics
docker exec recommndr-kafka kafka-topics --bootstrap-server localhost:9092 --list

# Check Redis feature store
docker exec recommndr-redis redis-cli keys "*"
```

### Step 2: Run Basic Integration Test
```bash
./venv/bin/python test_kafka_recommendations.py
```

### Step 3: Simulate Realistic Traffic
```bash
./venv/bin/python simulate_live_events.py
```

### Step 4: Monitor Streaming Pipeline
```bash
# Run Phase 2 streaming pipeline
./venv/bin/python -m src.streaming.main --num-events 100

# In another terminal, check processing
docker logs recommndr-flink-jobmanager --tail 20
docker logs recommndr-flink-taskmanager --tail 20
```

### Step 5: Test Recommendation Updates
```bash
# Get recommendations for a user before events
./venv/bin/python get_user_recommendations.py 1118

# Send live events (in another terminal)
./venv/bin/python simulate_live_events.py

# Get recommendations again to see changes
./venv/bin/python get_user_recommendations.py 1118
```

## 📊 What to Expect

### Event Flow
1. **Events Generated**: Realistic user interactions (views, clicks, purchases)
2. **Kafka Ingestion**: Events sent to `clickstream-events` topic
3. **Flink Processing**: Real-time feature extraction and aggregation
4. **Feature Store**: Updated user/product features in Redis
5. **Recommendations**: ML model uses updated features for personalization

### Real-time Updates
- **Immediate**: Event ingestion through Kafka
- **~5-10 seconds**: Flink processing and feature updates
- **~10-15 seconds**: Feature store updates
- **Next request**: Updated recommendations served

## 🧪 Advanced Testing Scenarios

### Test Scenario 1: User Behavior Change
```bash
# 1. Get initial recommendations for user
./venv/bin/python get_user_recommendations.py 1118

# 2. Simulate user exploring new category
./venv/bin/python -c "
from simulate_live_events import LiveEventSimulator
import asyncio

async def test():
    sim = LiveEventSimulator()
    sim.initialize()
    # Send events in Electronics category
    await sim.send_user_session(1118)
    await sim.cleanup()

asyncio.run(test())
"

# 3. Check updated recommendations
./venv/bin/python get_user_recommendations.py 1118
```

### Test Scenario 2: High-Value User Journey
```bash
# Simulate a purchasing user journey
./venv/bin/python -c "
import asyncio
from src.streaming.kafka_producer import KafkaProducer
from datetime import datetime, timezone

async def purchase_journey():
    producer = KafkaProducer()
    user_id = 1118
    product_id = 123
    
    events = [
        {'event_type': 'view', 'dwell_time': 45},
        {'event_type': 'click', 'dwell_time': None},
        {'event_type': 'add_to_cart', 'quantity': 1},
        {'event_type': 'purchase', 'quantity': 1}
    ]
    
    for event_data in events:
        event = {
            'user_id': user_id,
            'product_id': product_id,
            'event_type': event_data['event_type'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': f'test_purchase_{user_id}',
            **{k: v for k, v in event_data.items() if k != 'event_type'}
        }
        await producer.send_event(event)
        print(f'Sent: {event_data[\"event_type\"]}')
    
    await producer.close()

asyncio.run(purchase_journey())
"
```

### Test Scenario 3: A/B Testing Simulation
```bash
# Test two different user groups
./venv/bin/python -c "
import asyncio
from simulate_live_events import LiveEventSimulator

async def ab_test():
    sim = LiveEventSimulator()
    sim.initialize()
    
    # Group A: Electronics focused
    print('Group A: Electronics users')
    for user_id in [1, 2, 3]:
        await sim.send_user_session(user_id)
    
    # Group B: Books focused  
    print('Group B: Books users')
    for user_id in [4, 5, 6]:
        await sim.send_user_session(user_id)
    
    await sim.cleanup()

asyncio.run(ab_test())
"
```

## 📈 Monitoring and Analytics

### Kafka Monitoring
```bash
# Check topic message counts
docker exec recommndr-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic clickstream-events \
  --from-beginning --max-messages 10

# Monitor live events
docker exec recommndr-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic clickstream-events
```

### Flink Monitoring
```bash
# Access Flink Web UI
open http://localhost:8081

# Check job status
curl -s http://localhost:8081/jobs | jq '.'
```

### Redis Feature Store
```bash
# Check feature keys
docker exec recommndr-redis redis-cli keys "user:*" | head -10
docker exec recommndr-redis redis-cli keys "product:*" | head -10

# Get specific user features
docker exec recommndr-redis redis-cli get "user:1118"
```

### MLflow Experiment Tracking
```bash
# Access MLflow UI
open http://localhost:5001

# Log recommendation performance
./venv/bin/python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
mlflow.log_metric('recommendations_updated', 1)
print('Metric logged to MLflow')
"
```

## 🎯 Success Metrics

### Technical Metrics
- **Event Throughput**: Events/second processed
- **Latency**: Time from event to recommendation update
- **Processing Success Rate**: % of events successfully processed
- **Feature Freshness**: Time since last feature update

### Business Metrics
- **Recommendation Diversity**: Category spread in recommendations
- **Relevance Score Changes**: ML confidence improvements
- **User Engagement**: Simulated CTR and conversion rates
- **Real-time Responsiveness**: Adaptation to user behavior

## 🔧 Troubleshooting

### Common Issues

1. **Kafka Connection Failed**
   ```bash
   # Restart Kafka
   docker-compose restart kafka zookeeper
   ```

2. **Flink Job Not Processing**
   ```bash
   # Check Flink logs
   docker logs recommndr-flink-jobmanager
   docker logs recommndr-flink-taskmanager
   ```

3. **Redis Connection Issues**
   ```bash
   # Test Redis connectivity
   docker exec recommndr-redis redis-cli ping
   ```

4. **No Recommendation Changes**
   - Wait longer (15-30 seconds) for processing
   - Check if events are reaching Kafka
   - Verify Flink is processing events
   - Ensure feature store is updating

### Debug Commands
```bash
# Full system health check
./venv/bin/python -c "
import asyncio
from src.streaming.main import Phase2Pipeline

async def health_check():
    pipeline = Phase2Pipeline()
    await pipeline.health_check()

asyncio.run(health_check())
"
```

## 🚀 Next Steps

After successful testing:

1. **Scale Testing**: Increase event volume and user count
2. **Performance Optimization**: Monitor and optimize processing times
3. **Real User Integration**: Connect to actual frontend application
4. **A/B Testing**: Implement recommendation algorithm variations
5. **Production Deployment**: Move to cloud infrastructure

---

**🎉 Your live recommendation system is ready for real-world testing!**
