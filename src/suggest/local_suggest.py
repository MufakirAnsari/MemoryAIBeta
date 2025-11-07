#!/usr/bin/env python3
"""
MemoryAI Enterprise - Local Suggestion Engine
Generates contextual suggestions based on user patterns and memory graph
"""

import os
import sys
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import sqlite3
from collections import defaultdict, deque
import re

# Import memory and routing components
sys.path.append('/mnt/okcomputer/memoryai-enterprise/src')
from memory.wpmg_builder import get_wpmg_builder
from memory.rag_fusion import get_rag_fusion
from llm.router_zk.zk_router import get_zk_router, RoutingDecision

@dataclass
class SuggestionContext:
    """Context for generating suggestions"""
    user_id: str
    current_activity: str
    recent_queries: List[str]
    location_context: Dict[str, Any]
    time_context: Dict[str, Any]
    user_mood: str = "neutral"
    focus_level: float = 0.5  # 0-1 scale
    available_time: int = 30  # minutes
    device_type: str = "desktop"

@dataclass
class Suggestion:
    """Generated suggestion"""
    suggestion_id: str
    content: str
    suggestion_type: str  # 'proactive', 'reactive', 'contextual', 'memory-based'
    confidence: float
    reasoning: str
    action_url: Optional[str] = None
    priority: int = 3  # 1=high, 2=medium, 3=low
    expiration_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class LocalSuggestionEngine:
    """Local suggestion engine using personal memory graph"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.wpmg = get_wpmg_builder(user_id)
        self.rag_fusion = get_rag_fusion(user_id)
        self.router = get_zk_router()
        
        # Suggestion patterns and templates
        self.suggestion_patterns = self._load_suggestion_patterns()
        self.context_weights = self._get_context_weights()
        
        # Learning components
        self.suggestion_history = deque(maxlen=1000)
        self.success_patterns = defaultdict(float)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize database for suggestion tracking
        self._init_database()
    
    def _init_database(self):
        """Initialize database for suggestion tracking"""
        db_path = f"/tmp/suggestions_{self.user_id}.db"
        self.conn = sqlite3.connect(db_path)
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS suggestions (
                suggestion_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                suggestion_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                accepted BOOLEAN DEFAULT NULL,
                user_feedback INTEGER DEFAULT NULL,
                context TEXT DEFAULT '{}'
            )
        ''')
        
        self.conn.commit()
    
    def _load_suggestion_patterns(self) -> Dict[str, List[str]]:
        """Load suggestion patterns for different contexts"""
        
        return {
            'productivity': [
                "You mentioned working on {topic}. Would you like me to find related resources?",
                "Based on your recent activity, you might be interested in {related_topic}",
                "I noticed you haven't followed up on {previous_task}. Should we revisit it?",
                "Your productivity patterns suggest you're most effective at {time_of_day} for {task_type}"
            ],
            'learning': [
                "You've been studying {subject}. Here's a related concept you might find interesting.",
                "Your learning progress suggests you're ready for {next_topic}",
                "Based on your interests, I recommend exploring {related_field}",
                "You seem to learn best through {learning_style}. Try this approach for {current_topic}"
            ],
            'memory': [
                "This reminds me of {previous_memory}. Would you like to see the connection?",
                "You had a similar experience with {related_memory}. Should I show you the pattern?",
                "Your memories about {topic} might be relevant here.",
                "I've noticed a pattern in your memories about {theme}. Would you like to explore it?"
            ],
            'contextual': [
                "Given your current location and time, you might want to {suggested_action}",
                "Your calendar shows {upcoming_event}. Here's something relevant.",
                "Based on your device and location, I suggest {device_specific_action}",
                "Your recent patterns indicate you usually {usual_activity} at this time"
            ],
            'proactive': [
                "Good morning! Based on your routine, you might want to start with {morning_task}",
                "I noticed it's been a while since you last {activity}. Should we revisit it?",
                "Your goals suggest you should focus on {goal_related_task} today",
                "I've prepared {resource_type} based on your recent interests"
            ]
        }
    
    def _get_context_weights(self) -> Dict[str, float]:
        """Get weights for different context factors"""
        
        return {
            'recent_activity': 0.25,
            'memory_relevance': 0.30,
            'temporal_patterns': 0.20,
            'user_preferences': 0.15,
            'external_context': 0.10
        }
    
    def generate_suggestions(self, context: SuggestionContext, max_suggestions: int = 5) -> List[Suggestion]:
        """Generate contextual suggestions"""
        
        suggestions = []
        
        # 1. Memory-based suggestions
        memory_suggestions = self._generate_memory_suggestions(context)
        suggestions.extend(memory_suggestions)
        
        # 2. Pattern-based suggestions
        pattern_suggestions = self._generate_pattern_suggestions(context)
        suggestions.extend(pattern_suggestions)
        
        # 3. Contextual suggestions
        contextual_suggestions = self._generate_contextual_suggestions(context)
        suggestions.extend(contextual_suggestions)
        
        # 4. Proactive suggestions
        proactive_suggestions = self._generate_proactive_suggestions(context)
        suggestions.extend(proactive_suggestions)
        
        # Rank and filter suggestions
        ranked_suggestions = self._rank_suggestions(suggestions, context)
        
        # Limit to max_suggestions
        final_suggestions = ranked_suggestions[:max_suggestions]
        
        # Track suggestions
        for suggestion in final_suggestions:
            self._track_suggestion(suggestion, context)
        
        return final_suggestions
    
    def _generate_memory_suggestions(self, context: SuggestionContext) -> List[Suggestion]:
        """Generate suggestions based on personal memories"""
        
        suggestions = []
        
        # Query relevant memories
        query = " ".join(context.recent_queries[-3:]) if context.recent_queries else context.current_activity
        
        if query:
            # Get relevant memories
            memories = self.wpmg.query_memories(query, limit=10)
            
            for memory in memories[:3]:  # Top 3 memories
                if memory['similarity_score'] > 0.3:
                    # Generate suggestion based on memory
                    template = np.random.choice(self.suggestion_patterns['memory'])
                    
                    content = template.format(
                        previous_memory=memory['content'][:50] + "...",
                        topic=self._extract_topic(memory['content']),
                        related_memory=self._find_related_memory(memory['content']),
                        theme=self._extract_theme(memory['content'])
                    )
                    
                    suggestion = Suggestion(
                        suggestion_id=self._generate_suggestion_id(),
                        content=content,
                        suggestion_type="memory-based",
                        confidence=memory['similarity_score'] * memory['importance'],
                        reasoning=f"Based on similar memory with score {memory['similarity_score']:.2f}",
                        priority=2,
                        metadata={'memory_id': memory['node_id'], 'similarity': memory['similarity_score']}
                    )
                    
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_pattern_suggestions(self, context: SuggestionContext) -> List[Suggestion]:
        """Generate suggestions based on user patterns"""
        
        suggestions = []
        
        # Analyze recent queries for patterns
        if len(context.recent_queries) >= 3:
            patterns = self._analyze_query_patterns(context.recent_queries)
            
            for pattern in patterns[:2]:
                template = np.random.choice(self.suggestion_patterns['productivity'])
                
                content = template.format(
                    topic=pattern['topic'],
                    related_topic=self._find_related_topic(pattern['topic']),
                    previous_task=pattern.get('previous_task', 'a previous task'),
                    time_of_day=self._get_time_of_day(context),
                    task_type=pattern.get('task_type', 'this type of work')
                )
                
                suggestion = Suggestion(
                    suggestion_id=self._generate_suggestion_id(),
                    content=content,
                    suggestion_type="pattern-based",
                    confidence=pattern['confidence'],
                    reasoning=f"Detected pattern in recent queries",
                    priority=2,
                    metadata={'pattern_type': pattern['type'], 'topic': pattern['topic']}
                )
                
                suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_contextual_suggestions(self, context: SuggestionContext) -> List[Suggestion]:
        """Generate suggestions based on current context"""
        
        suggestions = []
        
        # Location and time-based suggestions
        if context.location_context or context.time_context:
            template = np.random.choice(self.suggestion_patterns['contextual'])
            
            suggested_action = self._suggest_location_action(context)
            upcoming_event = self._check_calendar(context)
            device_action = self._suggest_device_action(context)
            usual_activity = self._get_usual_activity(context)
            
            content = template.format(
                suggested_action=suggested_action,
                upcoming_event=upcoming_event,
                device_specific_action=device_action,
                usual_activity=usual_activity
            )
            
            suggestion = Suggestion(
                suggestion_id=self._generate_suggestion_id(),
                content=content,
                suggestion_type="contextual",
                confidence=0.7,  # Base confidence for contextual suggestions
                reasoning="Based on current context (location, time, device)",
                priority=3,
                metadata={
                    'location_context': context.location_context,
                    'time_context': context.time_context,
                    'device_type': context.device_type
                }
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_proactive_suggestions(self, context: SuggestionContext) -> List[Suggestion]:
        """Generate proactive suggestions based on user goals and patterns"""
        
        suggestions = []
        
        # Time-based proactive suggestions
        current_hour = datetime.now().hour
        
        if 6 <= current_hour <= 9:  # Morning
            morning_task = self._get_morning_task(context)
            template = "Good morning! Based on your routine, you might want to start with {morning_task}"
            
            content = template.format(morning_task=morning_task)
            
        elif 20 <= current_hour <= 23:  # Evening
            # Check for incomplete tasks
            incomplete_tasks = self._get_incomplete_tasks(context)
            if incomplete_tasks:
                template = "I noticed you haven't completed {task}. Should we work on it?"
                content = template.format(task=incomplete_tasks[0])
            else:
                template = "Your goals suggest you should review {goal_related_task} today"
                content = template.format(goal_related_task=self._get_goal_related_task(context))
        
        else:
            # General proactive suggestions
            if np.random.random() < 0.3:  # 30% chance of proactive suggestion
                goal_task = self._get_goal_related_task(context)
                template = "I've prepared {resource_type} based on your recent interests"
                content = template.format(resource_type=self._suggest_resource_type(context, goal_task))
        
        if 'content' in locals():
            suggestion = Suggestion(
                suggestion_id=self._generate_suggestion_id(),
                content=content,
                suggestion_type="proactive",
                confidence=0.6,
                reasoning="Proactive suggestion based on time and user patterns",
                priority=1,  # High priority for proactive suggestions
                metadata={'time_of_day': current_hour, 'context': context.current_activity}
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _rank_suggestions(self, suggestions: List[Suggestion], context: SuggestionContext) -> List[Suggestion]:
        """Rank suggestions based on multiple factors"""
        
        for suggestion in suggestions:
            # Base score from confidence
            score = suggestion.confidence
            
            # Adjust based on context
            score *= self._get_context_relevance_score(suggestion, context)
            
            # Adjust based on user state
            score *= self._get_user_state_multiplier(context)
            
            # Adjust based on historical success
            score *= self._get_historical_success_multiplier(suggestion.suggestion_type)
            
            # Priority adjustment
            priority_multiplier = {1: 1.2, 2: 1.0, 3: 0.8}
            score *= priority_multiplier.get(suggestion.priority, 1.0)
            
            suggestion.metadata['final_score'] = score
        
        # Sort by final score
        suggestions.sort(key=lambda x: x.metadata.get('final_score', 0), reverse=True)
        
        return suggestions
    
    def _get_context_relevance_score(self, suggestion: Suggestion, context: SuggestionContext) -> float:
        """Calculate relevance score based on context"""
        
        score = 1.0
        
        # Recent activity relevance
        if suggestion.suggestion_type in ['pattern-based', 'memory-based']:
            if context.recent_queries:
                recent_topics = [self._extract_topic(q) for q in context.recent_queries[-3:]]
                suggestion_topics = [self._extract_topic(suggestion.content)]
                
                # Simple topic overlap check
                overlap = len(set(recent_topics) & set(suggestion_topics))
                score *= (1 + overlap * 0.2)
        
        # Time relevance
        if suggestion.suggestion_type == 'contextual':
            if context.available_time < 15:  # Less than 15 minutes
                score *= 1.1 if 'quick' in suggestion.content.lower() else 0.9
        
        # Device relevance
        if context.device_type == 'mobile':
            score *= 1.1 if 'mobile' in suggestion.content.lower() or 'quick' in suggestion.content.lower() else 0.9
        
        return min(score, 2.0)  # Cap at 2.0
    
    def _get_user_state_multiplier(self, context: SuggestionContext) -> float:
        """Get multiplier based on user state"""
        
        multiplier = 1.0
        
        # Focus level adjustment
        if context.focus_level > 0.7:  # High focus
            multiplier *= 1.1
        elif context.focus_level < 0.3:  # Low focus
            multiplier *= 0.8
        
        # Mood adjustment
        mood_multipliers = {
            'happy': 1.1,
            'stressed': 0.7,
            'focused': 1.2,
            'tired': 0.6,
            'neutral': 1.0
        }
        
        multiplier *= mood_multipliers.get(context.user_mood, 1.0)
        
        return multiplier
    
    def _get_historical_success_multiplier(self, suggestion_type: str) -> float:
        """Get multiplier based on historical success rates"""
        
        # Get success rate for this suggestion type
        cursor = self.conn.execute('''
            SELECT 
                CAST(SUM(CASE WHEN accepted = 1 THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) as success_rate
            FROM suggestions 
            WHERE suggestion_type = ? AND accepted IS NOT NULL
        ''', (suggestion_type,))
        
        row = cursor.fetchone()
        success_rate = row[0] if row and row[0] is not None else 0.5
        
        # Map success rate to multiplier (0.5 to 1.5)
        return 0.5 + success_rate
    
    def _track_suggestion(self, suggestion: Suggestion, context: SuggestionContext):
        """Track suggestion for learning"""
        
        # Store in database
        self.conn.execute('''
            INSERT INTO suggestions 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            suggestion.suggestion_id,
            suggestion.content,
            suggestion.suggestion_type,
            suggestion.confidence,
            datetime.now().isoformat(),
            None,  # accepted (to be updated later)
            None,  # user_feedback (to be updated later)
            json.dumps({
                'context': context.to_dict() if hasattr(context, 'to_dict') else str(context),
                'final_score': suggestion.metadata.get('final_score', 0)
            })
        ))
        self.conn.commit()
        
        # Add to history
        self.suggestion_history.append({
            'suggestion_id': suggestion.suggestion_id,
            'timestamp': datetime.now(),
            'context': context,
            'suggestion': suggestion
        })
    
    def provide_feedback(self, suggestion_id: str, accepted: bool, user_feedback: Optional[int] = None):
        """Provide feedback on suggestion acceptance"""
        
        self.conn.execute('''
            UPDATE suggestions 
            SET accepted = ?, user_feedback = ?
            WHERE suggestion_id = ?
        ''', (accepted, user_feedback, suggestion_id))
        self.conn.commit()
        
        # Update success patterns
        # Implementation would update machine learning models
        self.logger.info(f"Feedback recorded for suggestion {suggestion_id}: accepted={accepted}")
    
    # Helper methods
    def _generate_suggestion_id(self) -> str:
        """Generate unique suggestion ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = np.random.randint(1000, 9999)
        return f"sugg_{timestamp}_{random_suffix}"
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text (simplified)"""
        # In practice, would use NLP techniques
        words = text.split()[:3]
        return " ".join(words)
    
    def _find_related_memory(self, content: str) -> str:
        """Find related memory (simplified)"""
        return "a similar experience"
    
    def _extract_theme(self, text: str) -> str:
        """Extract theme from text (simplified)"""
        return "general"
    
    def _analyze_query_patterns(self, queries: List[str]) -> List[Dict]:
        """Analyze patterns in recent queries"""
        patterns = []
        
        # Simple pattern detection
        topics = [self._extract_topic(q) for q in queries]
        
        if len(set(topics)) < len(topics):  # Some repetition
            patterns.append({
                'type': 'repeated_topic',
                'topic': topics[0],
                'confidence': 0.7
            })
        
        return patterns
    
    def _find_related_topic(self, topic: str) -> str:
        """Find related topic (simplified)"""
        related_topics = {
            'python': 'programming',
            'machine learning': 'AI',
            'hiking': 'outdoor activities'
        }
        return related_topics.get(topic, 'related topics')
    
    def _get_time_of_day(self, context: SuggestionContext) -> str:
        """Get time of day description"""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        else:
            return "evening"
    
    def _suggest_location_action(self, context: SuggestionContext) -> str:
        """Suggest action based on location"""
        return "continue your current work"
    
    def _check_calendar(self, context: SuggestionContext) -> str:
        """Check calendar for upcoming events"""
        return "your next meeting"
    
    def _suggest_device_action(self, context: SuggestionContext) -> str:
        """Suggest action based on device"""
        return "use voice commands"
    
    def _get_usual_activity(self, context: SuggestionContext) -> str:
        """Get usual activity for this time"""
        return "work on personal projects"
    
    def _get_morning_task(self, context: SuggestionContext) -> str:
        """Get suggested morning task"""
        return "review your goals for the day"
    
    def _get_incomplete_tasks(self, context: SuggestionContext) -> List[str]:
        """Get incomplete tasks"""
        return ["your reading list", "your exercise routine"]
    
    def _get_goal_related_task(self, context: SuggestionContext) -> str:
        """Get task related to user goals"""
        return "practice coding"
    
    def _suggest_resource_type(self, context: SuggestionContext, goal_task: str) -> str:
        """Suggest resource type"""
        return "some tutorials"

# Global suggestion engines
_global_suggestion_engines = {}

def get_suggestion_engine(user_id: str) -> LocalSuggestionEngine:
    """Get suggestion engine for user"""
    if user_id not in _global_suggestion_engines:
        _global_suggestion_engines[user_id] = LocalSuggestionEngine(user_id)
    return _global_suggestion_engines[user_id]

def main():
    """Demo local suggestion engine"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Setup demo user
    user_id = "demo_user_123"
    
    # Initialize suggestion engine
    engine = get_suggestion_engine(user_id)
    
    # Create sample context
    context = SuggestionContext(
        user_id=user_id,
        current_activity="Working on Python programming project",
        recent_queries=[
            "How to optimize Python code?",
            "Best practices for Python functions",
            "Python debugging techniques"
        ],
        location_context={"type": "home", "timezone": "PST"},
        time_context={"hour": 14, "day": "weekday"},
        user_mood="focused",
        focus_level=0.8,
        available_time=45,
        device_type="desktop"
    )
    
    print("🤖 Local Suggestion Engine Demo")
    print("=" * 50)
    print(f"User: {user_id}")
    print(f"Current Activity: {context.current_activity}")
    print(f"Recent Queries: {', '.join(context.recent_queries)}")
    print(f"Context: {context.user_mood} mood, {context.focus_level} focus level")
    print()
    
    # Generate suggestions
    suggestions = engine.generate_suggestions(context, max_suggestions=5)
    
    print(f"🔍 Generated {len(suggestions)} suggestions:\n")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion.content}")
        print(f"   Type: {suggestion.suggestion_type}")
        print(f"   Confidence: {suggestion.confidence:.3f}")
        print(f"   Priority: {suggestion.priority}")
        print(f"   Reasoning: {suggestion.reasoning}")
        print()
    
    # Simulate user feedback
    if suggestions:
        print("📊 Recording user feedback...")
        engine.provide_feedback(suggestions[0].suggestion_id, accepted=True, user_feedback=5)
        print(f"✅ Feedback recorded for suggestion {suggestions[0].suggestion_id}")
    
    print(f"\n✅ Local Suggestion Engine demonstration complete!")

if __name__ == "__main__":
    main()