#!/usr/bin/env python3
"""
MemoryAI Enterprise - Global Router with Fallback Cascade
Routes requests between local and global LLMs with intelligent fallback strategies
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import sqlite3
from collections import defaultdict, deque
import numpy as np

# Import routing components
sys.path.append('/mnt/okcomputer/memoryai-enterprise/src')
from llm.router_zk.zk_router import get_zk_router, RoutingInput, RoutingOutput, RoutingDecision
from suggest.local_suggest import get_suggestion_engine, SuggestionContext

class RoutingStrategy(Enum):
    """Routing strategies for different scenarios"""
    OPTIMAL_QUALITY = "optimal_quality"
    OPTIMAL_SPEED = "optimal_speed"
    BALANCED = "balanced"
    COST_EFFICIENT = "cost_efficient"
    PRIVACY_FIRST = "privacy_first"

@dataclass
class LLMEndpoint:
    """LLM endpoint configuration"""
    name: str
    url: str
    model_name: str
    max_tokens: int
    timeout: float
    cost_per_token: float
    privacy_level: int  # 0=public cloud, 1=private cloud, 2=local
    capabilities: List[str]  # Features supported
    current_load: float = 0.0
    avg_latency: float = 0.0
    success_rate: float = 1.0

@dataclass
class RoutingMetrics:
    """Metrics for routing decisions"""
    decision_time: float
    chosen_endpoint: str
    fallback_used: bool
    final_latency: float
    token_usage: int
    cost: float
    user_satisfaction: Optional[float] = None

class GlobalRouter:
    """Global router with intelligent fallback cascade"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.zk_router = get_zk_router()
        self.suggestion_engine = get_suggestion_engine(user_id)
        
        # LLM endpoints
        self.endpoints = self._setup_endpoints()
        
        # Routing configuration
        self.fallback_cascade = True
        self.timeout_threshold = 1.2  # 1.2 seconds
        self.circuit_breaker_threshold = 0.5  # 50% failure rate
        
        # Performance tracking
        self.metrics_history = deque(maxlen=1000)
        self.endpoint_stats = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'total_latency': 0.0,
            'last_failure': None
        })
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _setup_endpoints(self) -> Dict[str, LLMEndpoint]:
        """Setup available LLM endpoints"""
        
        return {
            'local_llama': LLMEndpoint(
                name='local_llama',
                url='http://localhost:8080/generate',
                model_name='llama-4-2b-moe-q4_0',
                max_tokens=4096,
                timeout=2.0,
                cost_per_token=0.0,
                privacy_level=2,
                capabilities=['text_generation', 'code', 'reasoning', 'safety_filtered']
            ),
            'global_gpt4': LLMEndpoint(
                name='global_gpt4',
                url='https://api.openai.com/v1/chat/completions',
                model_name='gpt-4',
                max_tokens=8192,
                timeout=3.0,
                cost_per_token=0.00006,
                privacy_level=0,
                capabilities=['text_generation', 'code', 'reasoning', 'multimodal', 'web_browsing']
            ),
            'global_claude': LLMEndpoint(
                name='global_claude',
                url='https://api.anthropic.com/v1/messages',
                model_name='claude-3-sonnet',
                max_tokens=4096,
                timeout=3.0,
                cost_per_token=0.00003,
                privacy_level=0,
                capabilities=['text_generation', 'code', 'reasoning', 'analysis']
            ),
            'cached_responses': LLMEndpoint(
                name='cached_responses',
                url='internal://cache',
                model_name='cached',
                max_tokens=2048,
                timeout=0.1,
                cost_per_token=0.0,
                privacy_level=2,
                capabilities=['cached_answers', 'quick_responses']
            )
        }
    
    def _init_database(self):
        """Initialize database for routing metrics"""
        db_path = f"/tmp/routing_{self.user_id}.db"
        self.conn = sqlite3.connect(db_path)
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS routing_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                decision_time REAL NOT NULL,
                chosen_endpoint TEXT NOT NULL,
                fallback_used BOOLEAN NOT NULL,
                final_latency REAL NOT NULL,
                token_usage INTEGER,
                cost REAL,
                user_satisfaction REAL,
                routing_strategy TEXT,
                zk_proof TEXT
            )
        ''')
        
        self.conn.commit()
    
    async def route_query(
        self, 
        query: str, 
        context: Dict[str, Any],
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        require_zk_proof: bool = True
    ) -> Dict[str, Any]:
        """Route query with intelligent fallback cascade"""
        
        start_time = time.time()
        
        # Create routing input
        routing_input = RoutingInput(
            query=query,
            user_id=self.user_id,
            context=context,
            timestamp=time.time(),
            latency_threshold=self.timeout_threshold,
            safety_score=context.get('safety_score', 1.0),
            complexity_score=context.get('complexity_score', 0.5),
            cache_hit_probability=context.get('cache_hit_probability', 0.0)
        )
        
        # Get routing decision with ZK proof
        routing_output = self.zk_router.route_query(routing_input)
        
        decision_time = time.time() - start_time
        
        # Select endpoint based on routing decision and strategy
        selected_endpoint = await self._select_endpoint(
            routing_output.decision, 
            strategy, 
            context
        )
        
        # Execute with fallback cascade
        result = await self._execute_with_fallback(
            query, 
            selected_endpoint, 
            routing_output, 
            strategy
        )
        
        # Record metrics
        final_latency = time.time() - start_time
        metrics = RoutingMetrics(
            decision_time=decision_time,
            chosen_endpoint=result['endpoint_used'],
            fallback_used=result['fallback_used'],
            final_latency=final_latency,
            token_usage=result.get('token_usage', 0),
            cost=result.get('cost', 0.0)
        )
        
        self._record_metrics(metrics, routing_input, routing_output, strategy)
        
        return {
            'response': result['response'],
            'metadata': {
                'routing_decision': routing_output.decision.value,
                'endpoint_used': result['endpoint_used'],
                'fallback_used': result['fallback_used'],
                'latency': final_latency,
                'cost': result.get('cost', 0.0),
                'zk_proof': routing_output.zk_proof if require_zk_proof else None,
                'suggestions': await self._get_contextual_suggestions(query, context)
            }
        }
    
    async def _select_endpoint(
        self, 
        decision: RoutingDecision, 
        strategy: RoutingStrategy,
        context: Dict[str, Any]
    ) -> str:
        """Select appropriate endpoint based on decision and strategy"""
        
        # Strategy-based endpoint selection
        if strategy == RoutingStrategy.PRIVACY_FIRST:
            # Prefer local endpoints
            if decision == RoutingDecision.LOCAL_LLM:
                return 'local_llama'
            elif decision == RoutingDecision.CACHED_ANSWER:
                return 'cached_responses'
            else:
                return 'local_llama'  # Fallback to local
        
        elif strategy == RoutingStrategy.OPTIMAL_SPEED:
            # Fastest available endpoint
            if decision == RoutingDecision.CACHED_ANSWER:
                return 'cached_responses'
            elif self._is_endpoint_healthy('local_llama'):
                return 'local_llama'
            else:
                return 'global_gpt4'  # Usually fast global option
        
        elif strategy == RoutingStrategy.COST_EFFICIENT:
            # Cheapest appropriate endpoint
            if decision == RoutingDecision.CACHED_ANSWER:
                return 'cached_responses'
            elif decision == RoutingDecision.LOCAL_LLM:
                return 'local_llama'
            else:
                return 'global_claude'  # Cheaper than GPT-4
        
        elif strategy == RoutingStrategy.OPTIMAL_QUALITY:
            # Best quality for complex queries
            if decision == RoutingDecision.GLOBAL_LLM:
                return 'global_gpt4'
            elif self._is_endpoint_healthy('global_claude'):
                return 'global_claude'
            else:
                return 'local_llama'
        
        else:  # BALANCED
            # Balanced approach based on decision
            decision_mapping = {
                RoutingDecision.LOCAL_LLM: 'local_llama',
                RoutingDecision.GLOBAL_LLM: 'global_gpt4',
                RoutingDecision.CACHED_ANSWER: 'cached_responses',
                RoutingDecision.FALLBACK: 'local_llama',
                RoutingDecision.SAFETY_REFUSAL: 'local_llama'
            }
            
            return decision_mapping.get(decision, 'local_llama')
    
    async def _execute_with_fallback(
        self, 
        query: str, 
        primary_endpoint: str, 
        routing_output: RoutingOutput,
        strategy: RoutingStrategy
    ) -> Dict[str, Any]:
        """Execute query with intelligent fallback cascade"""
        
        endpoints_to_try = [primary_endpoint]
        
        # Add fallback endpoints based on strategy
        if self.fallback_cascade:
            fallback_endpoints = self._get_fallback_endpoints(primary_endpoint, strategy)
            endpoints_to_try.extend(fallback_endpoints)
        
        last_error = None
        
        for i, endpoint_name in enumerate(endpoints_to_try):
            endpoint = self.endpoints[endpoint_name]
            
            # Check if endpoint is healthy
            if not self._is_endpoint_healthy(endpoint_name):
                self.logger.warning(f"Endpoint {endpoint_name} is unhealthy, skipping")
                continue
            
            try:
                # Try to execute on this endpoint
                result = await self._execute_on_endpoint(query, endpoint, routing_output)
                
                # Success - update stats
                self._update_endpoint_stats(endpoint_name, success=True, latency=result['latency'])
                
                return {
                    'response': result['response'],
                    'endpoint_used': endpoint_name,
                    'fallback_used': i > 0,
                    'token_usage': result.get('token_usage', 0),
                    'cost': result.get('cost', 0.0)
                }
                
            except Exception as e:
                self.logger.error(f"Execution failed on {endpoint_name}: {e}")
                self._update_endpoint_stats(endpoint_name, success=False)
                last_error = e
                
                # If this was the primary endpoint and we have fallbacks, continue
                if i == 0 and len(endpoints_to_try) > 1:
                    continue
                else:
                    # No more fallbacks or this wasn't the primary
                    break
        
        # All endpoints failed - return fallback response
        if last_error:
            self.logger.error(f"All endpoints failed. Last error: {last_error}")
        
        return {
            'response': self._generate_fallback_response(query, routing_output),
            'endpoint_used': 'fallback',
            'fallback_used': True,
            'token_usage': 0,
            'cost': 0.0
        }
    
    async def _execute_on_endpoint(
        self, 
        query: str, 
        endpoint: LLMEndpoint, 
        routing_output: RoutingOutput
    ) -> Dict[str, Any]:
        """Execute query on specific endpoint"""
        
        start_time = time.time()
        
        if endpoint.name == 'cached_responses':
            # Check cache
            cached_response = await self._check_cache(query)
            if cached_response:
                return {
                    'response': cached_response,
                    'latency': time.time() - start_time,
                    'token_usage': len(cached_response.split()),
                    'cost': 0.0
                }
            else:
                raise Exception("No cached response available")
        
        elif endpoint.name == 'local_llama':
            # Use local LLM
            return await self._call_local_llm(query, endpoint)
        
        else:  # Global endpoints
            return await self._call_global_llm(query, endpoint, routing_output)
    
    async def _call_local_llm(self, query: str, endpoint: LLMEndpoint) -> Dict[str, Any]:
        """Call local LLM endpoint"""
        
        # This would integrate with the actual local LLM service
        # For demo, simulate response
        await asyncio.sleep(0.5)  # Simulate processing time
        
        response = f"Local LLM response to: {query[:50]}..."
        
        return {
            'response': response,
            'latency': 0.5,
            'token_usage': len(response.split()),
            'cost': 0.0
        }
    
    async def _call_global_llm(
        self, 
        query: str, 
        endpoint: LLMEndpoint, 
        routing_output: RoutingOutput
    ) -> Dict[str, Any]:
        """Call global LLM endpoint"""
        
        # This would make actual API calls to global LLM services
        # For demo, simulate response
        await asyncio.sleep(np.random.uniform(0.8, 2.0))  # Variable latency
        
        response = f"Global {endpoint.model_name} response to: {query[:50]}..."
        token_usage = len(response.split())
        cost = token_usage * endpoint.cost_per_token
        
        return {
            'response': response,
            'latency': np.random.uniform(0.8, 2.0),
            'token_usage': token_usage,
            'cost': cost
        }
    
    async def _check_cache(self, query: str) -> Optional[str]:
        """Check if response is cached"""
        
        # Simple cache check based on query hash
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # In production, this would check a distributed cache
        # For demo, simulate cache hit 20% of the time
        if np.random.random() < 0.2:
            return f"Cached response for query hash: {query_hash[:8]}"
        
        return None
    
    def _generate_fallback_response(self, query: str, routing_output: RoutingOutput) -> str:
        """Generate fallback response when all endpoints fail"""
        
        return (
            f"I apologize, but I'm currently experiencing technical difficulties. "
            f"Your query about '{query[:30]}...' couldn't be processed at the moment. "
            f"Please try again in a few moments, or I can help you with basic questions "
            f"using my local knowledge base."
        )
    
    def _get_fallback_endpoints(self, primary: str, strategy: RoutingStrategy) -> List[str]:
        """Get fallback endpoints in priority order"""
        
        fallback_priority = {
            RoutingStrategy.BALANCED: ['cached_responses', 'local_llama', 'global_claude'],
            RoutingStrategy.OPTIMAL_QUALITY: ['global_claude', 'local_llama', 'cached_responses'],
            RoutingStrategy.OPTIMAL_SPEED: ['cached_responses', 'local_llama', 'global_gpt4'],
            RoutingStrategy.COST_EFFICIENT: ['cached_responses', 'local_llama', 'global_claude'],
            RoutingStrategy.PRIVACY_FIRST: ['cached_responses', 'local_llama']
        }
        
        fallbacks = fallback_priority.get(strategy, [])
        return [ep for ep in fallbacks if ep != primary]
    
    def _is_endpoint_healthy(self, endpoint_name: str) -> bool:
        """Check if endpoint is healthy"""
        
        stats = self.endpoint_stats[endpoint_name]
        
        # Check circuit breaker
        if stats['total_requests'] > 10:
            failure_rate = 1 - (stats['successful_requests'] / stats['total_requests'])
            if failure_rate > self.circuit_breaker_threshold:
                return False
        
        # Check if recently failed
        if stats['last_failure']:
            time_since_failure = time.time() - stats['last_failure']
            if time_since_failure < 60:  # Cooldown period
                return False
        
        return True
    
    def _update_endpoint_stats(self, endpoint_name: str, success: bool, latency: float = 0.0):
        """Update endpoint statistics"""
        
        stats = self.endpoint_stats[endpoint_name]
        stats['total_requests'] += 1
        
        if success:
            stats['successful_requests'] += 1
            stats['total_latency'] += latency
        else:
            stats['last_failure'] = time.time()
    
    async def _get_contextual_suggestions(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Get contextual suggestions for user"""
        
        # Create suggestion context
        suggestion_context = SuggestionContext(
            user_id=self.user_id,
            current_activity=query,
            recent_queries=[query],
            location_context=context.get('location', {}),
            time_context={'timestamp': datetime.now().isoformat()},
            user_mood=context.get('mood', 'neutral'),
            focus_level=context.get('focus_level', 0.5),
            available_time=30,
            device_type=context.get('device_type', 'desktop')
        )
        
        # Get suggestions
        suggestions = self.suggestion_engine.generate_suggestions(suggestion_context, max_suggestions=2)
        
        return [s.content for s in suggestions]
    
    def _record_metrics(
        self, 
        metrics: RoutingMetrics, 
        routing_input: RoutingInput, 
        routing_output: RoutingOutput,
        strategy: RoutingStrategy
    ):
        """Record routing metrics"""
        
        # Store in database
        self.conn.execute('''
            INSERT INTO routing_metrics 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            hashlib.md5(routing_input.query.encode()).hexdigest(),
            metrics.decision_time,
            metrics.chosen_endpoint,
            metrics.fallback_used,
            metrics.final_latency,
            metrics.token_usage,
            metrics.cost,
            metrics.user_satisfaction,
            strategy.value,
            routing_output.zk_proof
        ))
        self.conn.commit()
        
        # Add to history
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'strategy': strategy
        })
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        
        if not self.metrics_history:
            return {'total_requests': 0}
        
        total_requests = len(self.metrics_history)
        fallback_usage = sum(1 for m in self.metrics_history if m['metrics'].fallback_used)
        avg_latency = np.mean([m['metrics'].final_latency for m in self.metrics_history])
        total_cost = sum([m['metrics'].cost for m in self.metrics_history])
        
        endpoint_usage = defaultdict(int)
        for m in self.metrics_history:
            endpoint_usage[m['metrics'].chosen_endpoint] += 1
        
        return {
            'total_requests': total_requests,
            'fallback_usage_rate': fallback_usage / total_requests,
            'average_latency': avg_latency,
            'total_cost': total_cost,
            'endpoint_usage': dict(endpoint_usage),
            'success_rate': sum(m['metrics'].successful_requests for m in self.endpoint_stats.values()) / 
                           sum(m['metrics'].total_requests for m in self.endpoint_stats.values())
        }

# Global router instances
_global_routers = {}

def get_global_router(user_id: str) -> GlobalRouter:
    """Get global router for user"""
    if user_id not in _global_routers:
        _global_routers[user_id] = GlobalRouter(user_id)
    return _global_routers[user_id]

async def main():
    """Demo global router functionality"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Setup demo user
    user_id = "demo_user_123"
    router = get_global_router(user_id)
    
    # Test queries
    test_queries = [
        ("What is machine learning?", {"complexity_score": 0.3, "safety_score": 1.0}),
        ("Explain quantum computing applications in cryptography", {"complexity_score": 0.9, "safety_score": 1.0}),
        ("How do I optimize Python code for performance?", {"complexity_score": 0.6, "safety_score": 1.0}),
        ("Write a function to calculate fibonacci numbers", {"complexity_score": 0.4, "safety_score": 1.0}),
        ("What are the ethical implications of AI?", {"complexity_score": 0.7, "safety_score": 0.8})
    ]
    
    print("🌐 Global Router Demo")
    print("=" * 60)
    print("Testing different routing strategies and fallback cascade...\n")
    
    strategies = [
        RoutingStrategy.BALANCED,
        RoutingStrategy.OPTIMAL_QUALITY,
        RoutingStrategy.OPTIMAL_SPEED,
        RoutingStrategy.PRIVACY_FIRST
    ]
    
    for strategy in strategies:
        print(f"📊 Testing Strategy: {strategy.value.upper()}")
        print("-" * 40)
        
        for i, (query, context) in enumerate(test_queries[:2], 1):
            print(f"\nQuery {i}: {query}")
            
            try:
                result = await router.route_query(query, context, strategy=strategy)
                
                print(f"  ✅ Response: {result['response'][:80]}...")
                print(f"  🎯 Endpoint: {result['metadata']['endpoint_used']}")
                print(f"  ⏱️  Latency: {result['metadata']['latency']:.3f}s")
                print(f"  💰 Cost: ${result['metadata']['cost']:.6f}")
                print(f"  🔄 Fallback used: {result['metadata']['fallback_used']}")
                
                if result['metadata']['suggestions']:
                    print(f"  💡 Suggestions: {len(result['metadata']['suggestions'])} available")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print(f"\n" + "="*60 + "\n")
    
    # Show routing statistics
    stats = router.get_routing_stats()
    print("📈 Routing Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n✅ Global Router demonstration complete!")

if __name__ == "__main__":
    asyncio.run(main())