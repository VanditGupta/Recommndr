#!/usr/bin/env python3
"""
Test script for Phase 6: Simple Recommendation API

Tests all endpoints to ensure they're working for frontend development.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8001"

def test_health():
    """Test health endpoint."""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data['status']}")
            print(f"   📝 Note: {data['note']}")
            print(f"   🎯 Ready for Frontend: {data['ready_for_frontend']}")
            return True
        else:
            print(f"   ❌ Failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_recommendations():
    """Test recommendation endpoint."""
    print("\n🎯 Testing Recommendation Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/recommend/2?top_k=3")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ User ID: {data['user_id']}")
            print(f"   📊 Recommendations: {len(data['recommendations'])}")
            print(f"   ⚡ Generation Time: {data['generation_time']:.6f}s")
            print(f"   🔄 Phase: {data['phase']}")
            print(f"   📝 Note: {data['note']}")
            
            # Show first recommendation
            if data['recommendations']:
                first = data['recommendations'][0]
                print(f"   🏆 Top Recommendation: {first['name']} (${first['price']})")
            
            return True
        else:
            print(f"   ❌ Failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_user_profile():
    """Test user profile endpoint."""
    print("\n👤 Testing User Profile Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/user_profile/2")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ User ID: {data['user_id']}")
            print(f"   📊 Profile: {data['profile']['age']}y {data['profile']['gender']} from {data['profile']['location']}")
            print(f"   📱 Device: {data['profile']['device']}")
            print(f"   💰 Income: {data['profile']['income_level']}")
            print(f"   🏷️  Preferences: {', '.join(data['profile']['preferences'])}")
            print(f"   📈 Interactions: {len(data['interaction_history'])}")
            print(f"   📝 Note: {data['note']}")
            return True
        else:
            print(f"   ❌ Failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_similar_items():
    """Test similar items endpoint."""
    print("\n🔄 Testing Similar Items Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/similar_items/1?top_k=3")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Item ID: {data['item_id']}")
            print(f"   📊 Similar Items: {len(data['similar_items'])}")
            print(f"   ⚡ Query Time: {data['query_time']:.6f}s")
            print(f"   📝 Note: {data['note']}")
            
            # Show first similar item
            if data['similar_items']:
                first = data['similar_items'][0]
                print(f"   🔗 Most Similar: {first['name']} (Score: {first['similarity_score']:.2f})")
            
            return True
        else:
            print(f"   ❌ Failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_api_documentation():
    """Test API documentation endpoint."""
    print("\n📚 Testing API Documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("   ✅ Swagger UI available at /docs")
            print(f"   🌐 Open in browser: {BASE_URL}/docs")
            return True
        else:
            print(f"   ❌ Documentation not available: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 PHASE 6 API TESTING")
    print("=" * 50)
    print(f"Testing API at: {BASE_URL}")
    print()
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Run tests
    tests = [
        test_health,
        test_recommendations,
        test_user_profile,
        test_similar_items,
        test_api_documentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    # Summary
    print("=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("   🎯 Phase 6 API is ready for frontend development!")
        print(f"   🌐 API Documentation: {BASE_URL}/docs")
        print(f"   🧪 Health Check: {BASE_URL}/health")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the API server.")
    
    print("\n🚀 Ready to move to Phase 7 (Frontend)!")

if __name__ == "__main__":
    main()
