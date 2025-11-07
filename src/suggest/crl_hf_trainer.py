#!/usr/bin/env python3
"""
MemoryAI Enterprise - Continual Reinforcement Learning with Human Feedback (CRL-HF)
Implements policy gradient training on accept/reject feedback stream with DP-SGD
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
import sqlite3
from abc import ABC, abstractmethod

# Differential Privacy components
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

@dataclass
class FeedbackEvent:
    """Human feedback event"""
    timestamp: datetime
    user_id: str
    query: str
    suggestion_id: str
    suggestion_content: str
    context: Dict[str, Any]
    action_taken: str  # 'accepted', 'rejected', 'ignored'
    feedback_score: Optional[float] = None  # 1-5 scale
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyState:
    """Current state of the suggestion policy"""
    user_context_vector: np.ndarray
    temporal_features: np.ndarray
    content_features: np.ndarray
    historical_performance: np.ndarray

class PolicyNetwork(nn.Module):
    """Neural network for suggestion policy"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        logits = self.network(x)
        return self.softmax(logits)
    
    def get_action_probabilities(self, state: PolicyState) -> torch.Tensor:
        """Get action probabilities for given state"""
        
        # Combine all features
        state_vector = np.concatenate([
            state.user_context_vector,
            state.temporal_features,
            state.content_features,
            state.historical_performance
        ])
        
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        with torch.no_grad():
            return self.forward(state_tensor).squeeze(0)

class CRLHFTrainer:
    """Continual Reinforcement Learning with Human Feedback Trainer"""
    
    def __init__(
        self,
        user_id: str,
        policy_config: Dict[str, Any],
        dp_epsilon: float = 0.5,
        dp_delta: float = 1e-5,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        max_gradient_norm: float = 1.0
    ):
        self.user_id = user_id
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        
        # Initialize policy network
        self.policy_network = PolicyNetwork(**policy_config)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine()
        
        # Make model private
        self.policy_network, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.policy_network,
            optimizer=self.optimizer,
            data_loader=None,  # Will be set during training
            noise_multiplier=self._compute_noise_multiplier(),
            max_grad_norm=max_gradient_norm,
        )
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        self.feedback_buffer = deque(maxlen=5000)
        
        # State tracking
        self.state_history = defaultdict(deque)
        self.user_patterns = defaultdict(lambda: defaultdict(float))
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Training metrics
        self.training_metrics = {
            'total_updates': 0,
            'total_feedback_events': 0,
            'average_reward': 0.0,
            'policy_entropy': 0.0
        }
    
    def _init_database(self):
        """Initialize database for training data"""
        db_path = f"/tmp/crlhf_{self.user_id}.db"
        self.conn = sqlite3.connect(db_path)
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                state_vector TEXT NOT NULL,
                action_taken INTEGER NOT NULL,
                reward REAL NOT NULL,
                feedback_event_id TEXT NOT NULL
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS policy_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                loss REAL NOT NULL,
                entropy REAL NOT NULL,
                noise_multiplier REAL NOT NULL,
                batch_size INTEGER NOT NULL
            )
        ''')
        
        self.conn.commit()
    
    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier for DP-SGD"""
        # Using the relationship: epsilon = sqrt(2 * log(1.25/delta)) * noise_multiplier
        # Rearranged: noise_multiplier = epsilon / sqrt(2 * log(1.25/delta))
        
        import math
        return self.dp_epsilon / math.sqrt(2 * math.log(1.25 / self.dp_delta))
    
    def process_feedback(self, feedback_event: FeedbackEvent):
        """Process human feedback event"""
        
        # Convert feedback to reward
        reward = self._compute_reward(feedback_event)
        
        # Extract state from feedback context
        state = self._extract_state_from_feedback(feedback_event)
        
        # Get action index
        action_idx = self._get_action_index(feedback_event.suggestion_content)
        
        # Store experience
        experience = {
            'state': state,
            'action': action_idx,
            'reward': reward,
            'timestamp': feedback_event.timestamp,
            'feedback_id': feedback_event.suggestion_id
        }
        
        self.experience_buffer.append(experience)
        self.feedback_buffer.append(feedback_event)
        
        # Update user patterns
        self._update_user_patterns(feedback_event)
        
        # Trigger training if buffer is large enough
        if len(self.experience_buffer) >= self.batch_size:
            self._train_on_batch()
        
        self.training_metrics['total_feedback_events'] += 1
        
        self.logger.info(f"Processed feedback event: {feedback_event.action_taken} with reward {reward:.3f}")
    
    def _compute_reward(self, feedback_event: FeedbackEvent) -> float:
        """Convert feedback to reward signal"""
        
        base_rewards = {
            'accepted': 1.0,
            'rejected': -1.0,
            'ignored': -0.1
        }
        
        reward = base_rewards.get(feedback_event.action_taken, 0.0)
        
        # Adjust based on feedback score
        if feedback_event.feedback_score:
            # Scale feedback score (1-5) to (-1, 1)
            score_adjustment = (feedback_event.feedback_score - 3) / 2.0
            reward += score_adjustment * 0.5
        
        # Temporal decay for old feedback
        time_decay = self._compute_time_decay(feedback_event.timestamp)
        reward *= time_decay
        
        return np.clip(reward, -2.0, 2.0)
    
    def _compute_time_decay(self, timestamp: datetime) -> float:
        """Compute time decay factor for feedback"""
        
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        # Exponential decay with 24-hour half-life
        return np.exp(-age_hours * np.log(2) / 24)
    
    def _extract_state_from_feedback(self, feedback_event: FeedbackEvent) -> PolicyState:
        """Extract policy state from feedback context"""
        
        # User context vector
        user_context = self._extract_user_context(feedback_event)
        
        # Temporal features
        temporal_features = self._extract_temporal_features(feedback_event)
        
        # Content features
        content_features = self._extract_content_features(feedback_event)
        
        # Historical performance
        historical_features = self._extract_historical_features(feedback_event)
        
        return PolicyState(
            user_context_vector=user_context,
            temporal_features=temporal_features,
            content_features=content_features,
            historical_performance=historical_features
        )
    
    def _extract_user_context(self, feedback_event: FeedbackEvent) -> np.ndarray:
        """Extract user context features"""
        
        context = feedback_event.context
        
        # Feature extraction (simplified)
        features = [
            context.get('focus_level', 0.5),
            context.get('available_time', 30) / 60.0,  # Normalize to hours
            1.0 if context.get('device_type') == 'mobile' else 0.0,
            context.get('location_score', 0.5),
            self.user_patterns[feedback_event.user_id]['accept_rate'],
            self.user_patterns[feedback_event.user_id]['avg_feedback_score']
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_temporal_features(self, feedback_event: FeedbackEvent) -> np.ndarray:
        """Extract temporal features"""
        
        timestamp = feedback_event.timestamp
        
        features = [
            timestamp.hour / 24.0,  # Hour of day (normalized)
            timestamp.weekday() / 7.0,  # Day of week (normalized)
            1.0 if timestamp.weekday() < 5 else 0.0,  # Is weekday
            np.sin(2 * np.pi * timestamp.hour / 24),  # Cyclical hour
            np.cos(2 * np.pi * timestamp.hour / 24),  # Cyclical hour
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_content_features(self, feedback_event: FeedbackEvent) -> np.ndarray:
        """Extract content-based features"""
        
        content = feedback_event.suggestion_content.lower()
        
        # Simple content features
        features = [
            len(content) / 200.0,  # Normalized length
            content.count('?') / 5.0,  # Number of questions
            1.0 if 'help' in content else 0.0,
            1.0 if 'suggest' in content else 0.0,
            1.0 if 'remember' in content else 0.0,
            content.count(' ') / 50.0,  # Word count proxy
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_historical_features(self, feedback_event: FeedbackEvent) -> np.ndarray:
        """Extract historical performance features"""
        
        user_id = feedback_event.user_id
        suggestion_type = feedback_event.metadata.get('suggestion_type', 'unknown')
        
        # Historical success rates
        if user_id in self.user_patterns:
            user_stats = self.user_patterns[user_id]
            accept_rate = user_stats.get('accept_rate', 0.5)
            avg_feedback = user_stats.get('avg_feedback_score', 3.0) / 5.0
            recent_success = user_stats.get('recent_success_rate', 0.5)
        else:
            accept_rate = avg_feedback = recent_success = 0.5
        
        # Type-specific features
        type_features = {
            'memory-based': [1.0, 0.0, 0.0],
            'pattern-based': [0.0, 1.0, 0.0],
            'contextual': [0.0, 0.0, 1.0],
            'proactive': [0.5, 0.5, 0.0]
        }
        
        type_feature = type_features.get(suggestion_type, [0.0, 0.0, 0.0])
        
        features = [
            accept_rate,
            avg_feedback,
            recent_success,
            *type_feature
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _get_action_index(self, suggestion_content: str) -> int:
        """Map suggestion content to action index"""
        
        # Simple action mapping based on content type
        if 'memory' in suggestion_content.lower():
            return 0  # Memory-based action
        elif 'pattern' in suggestion_content.lower():
            return 1  # Pattern-based action
        elif 'context' in suggestion_content.lower():
            return 2  # Contextual action
        else:
            return 3  # Proactive action
    
    def _update_user_patterns(self, feedback_event: FeedbackEvent):
        """Update user-specific patterns based on feedback"""
        
        user_id = feedback_event.user_id
        
        # Update accept rate
        if feedback_event.action_taken == 'accepted':
            self.user_patterns[user_id]['accept_count'] = self.user_patterns[user_id].get('accept_count', 0) + 1
        
        self.user_patterns[user_id]['total_count'] = self.user_patterns[user_id].get('total_count', 0) + 1
        
        # Calculate accept rate
        if self.user_patterns[user_id]['total_count'] > 0:
            self.user_patterns[user_id]['accept_rate'] = (
                self.user_patterns[user_id].get('accept_count', 0) / 
                self.user_patterns[user_id]['total_count']
            )
        
        # Update average feedback score
        if feedback_event.feedback_score:
            current_avg = self.user_patterns[user_id].get('avg_feedback_score', 3.0)
            count = self.user_patterns[user_id].get('feedback_count', 0)
            
            new_avg = (current_avg * count + feedback_event.feedback_score) / (count + 1)
            
            self.user_patterns[user_id]['avg_feedback_score'] = new_avg
            self.user_patterns[user_id]['feedback_count'] = count + 1
    
    def _train_on_batch(self):
        """Train policy network on batch of experiences"""
        
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch_indices = np.random.choice(
            len(self.experience_buffer), 
            size=self.batch_size, 
            replace=False
        )
        
        batch = [self.experience_buffer[i] for i in batch_indices]
        
        # Prepare batch data
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        
        # Forward pass
        action_probs = self.policy_network(states)
        
        # Compute policy gradient loss
        selected_probs = action_probs.gather(1, actions.unsqueeze(1))
        loss = -torch.mean(torch.log(selected_probs + 1e-8) * rewards)
        
        # Add entropy bonus for exploration
        entropy = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1))
        loss = loss - 0.01 * entropy  # Entropy regularization
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_gradient_norm)
        
        # Update parameters
        self.optimizer.step()
        
        # Record training metrics
        self._record_training_update(loss.item(), entropy.item())
        
        self.training_metrics['total_updates'] += 1
        self.training_metrics['average_reward'] = 0.9 * self.training_metrics['average_reward'] + 0.1 * rewards.mean().item()
        self.training_metrics['policy_entropy'] = entropy.item()
        
        self.logger.info(f"Policy update complete. Loss: {loss.item():.4f}, Entropy: {entropy.item():.4f}")
    
    def _record_training_update(self, loss: float, entropy: float):
        """Record training update metrics"""
        
        self.conn.execute('''
            INSERT INTO policy_updates 
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            loss,
            entropy,
            self._compute_noise_multiplier(),
            self.batch_size,
            self.dp_epsilon
        ))
        self.conn.commit()
    
    def get_policy_recommendation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get policy-based suggestion recommendation"""
        
        # Extract current state
        state = self._extract_state_from_context(context)
        
        # Get action probabilities
        with torch.no_grad():
            state_tensor = torch.FloatTensor(self._state_to_vector(state)).unsqueeze(0)
            action_probs = self.policy_network(state_tensor).squeeze(0)
        
        # Get recommended action
        action_idx = torch.argmax(action_probs).item()
        confidence = action_probs[action_idx].item()
        
        # Map action to suggestion type
        action_types = ['memory-based', 'pattern-based', 'contextual', 'proactive']
        recommended_type = action_types[action_idx] if action_idx < len(action_types) else 'proactive'
        
        return {
            'recommended_suggestion_type': recommended_type,
            'confidence': confidence,
            'action_probabilities': action_probs.numpy().tolist(),
            'state_vector': self._state_to_vector(state).tolist()
        }
    
    def _extract_state_from_context(self, context: Dict[str, Any]) -> PolicyState:
        """Extract policy state from context"""
        
        # Create mock feedback event for state extraction
        mock_feedback = FeedbackEvent(
            timestamp=datetime.now(),
            user_id=self.user_id,
            query=context.get('current_query', ''),
            suggestion_id='mock',
            suggestion_content=context.get('last_suggestion', ''),
            context=context,
            action_taken='ignored',
            metadata={'suggestion_type': context.get('suggestion_type', 'proactive')}
        )
        
        return self._extract_state_from_feedback(mock_feedback)
    
    def _state_to_vector(self, state: PolicyState) -> np.ndarray:
        """Convert state to feature vector"""
        
        return np.concatenate([
            state.user_context_vector,
            state.temporal_features,
            state.content_features,
            state.historical_performance
        ])
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        
        return {
            'total_updates': self.training_metrics['total_updates'],
            'total_feedback_events': self.training_metrics['total_feedback_events'],
            'average_reward': self.training_metrics['average_reward'],
            'policy_entropy': self.training_metrics['policy_entropy'],
            'experience_buffer_size': len(self.experience_buffer),
            'privacy_epsilon': self.dp_epsilon,
            'privacy_delta': self.dp_delta
        }

# Global CRL-HF trainers
_global_crlhf_trainers = {}

def get_crlhf_trainer(user_id: str) -> CRLHFTrainer:
    """Get CRL-HF trainer for user"""
    if user_id not in _global_crlhf_trainers:
        policy_config = {
            'input_dim': 23,  # Total feature dimensions
            'hidden_dims': [64, 32],
            'output_dim': 4   # 4 suggestion types
        }
        _global_crlhf_trainers[user_id] = CRLHFTrainer(user_id, policy_config)
    return _global_crlhf_trainers[user_id]

def main():
    """Demo CRL-HF trainer"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Setup demo user
    user_id = "demo_user_123"
    trainer = get_crlhf_trainer(user_id)
    
    # Simulate feedback events
    sample_feedback_events = [
        FeedbackEvent(
            timestamp=datetime.now() - timedelta(hours=1),
            user_id=user_id,
            query="How to learn Python?",
            suggestion_id="sugg_001",
            suggestion_content="You mentioned learning Python. Here's a tutorial on functions.",
            context={'focus_level': 0.8, 'available_time': 60, 'device_type': 'desktop'},
            action_taken='accepted',
            feedback_score=4,
            metadata={'suggestion_type': 'memory-based'}
        ),
        FeedbackEvent(
            timestamp=datetime.now() - timedelta(minutes=30),
            user_id=user_id,
            query="Python debugging tips",
            suggestion_id="sugg_002",
            suggestion_content="Based on your recent activity, try using pdb for debugging.",
            context={'focus_level': 0.9, 'available_time': 30, 'device_type': 'desktop'},
            action_taken='accepted',
            feedback_score=5,
            metadata={'suggestion_type': 'pattern-based'}
        ),
        FeedbackEvent(
            timestamp=datetime.now() - timedelta(minutes=15),
            user_id=user_id,
            query="Best Python IDE?",
            suggestion_id="sugg_003",
            suggestion_content="Given your current location, you might want to try VS Code.",
            context={'focus_level': 0.6, 'available_time': 15, 'device_type': 'mobile'},
            action_taken='rejected',
            feedback_score=2,
            metadata={'suggestion_type': 'contextual'}
        )
    ]
    
    print("🧠 CRL-HF Trainer Demo")
    print("=" * 50)
    print("Processing sample feedback events...\n")
    
    for i, feedback_event in enumerate(sample_feedback_events, 1):
        print(f"Feedback Event {i}:")
        print(f"  Query: {feedback_event.query}")
        print(f"  Suggestion: {feedback_event.suggestion_content[:50]}...")
        print(f"  Action: {feedback_event.action_taken}")
        print(f"  Score: {feedback_event.feedback_score}")
        
        # Process feedback
        trainer.process_feedback(feedback_event)
        
        print(f"  ✅ Processed with reward: {trainer._compute_reward(feedback_event):.3f}")
        print()
    
    # Get policy recommendation
    context = {
        'current_query': 'How to optimize Python code?',
        'last_suggestion': 'Try using list comprehensions for better performance.',
        'focus_level': 0.8,
        'available_time': 45,
        'device_type': 'desktop',
        'suggestion_type': 'pattern-based'
    }
    
    recommendation = trainer.get_policy_recommendation(context)
    
    print("🎯 Policy Recommendation:")
    print(f"  Recommended Type: {recommendation['recommended_suggestion_type']}")
    print(f"  Confidence: {recommendation['confidence']:.3f}")
    print(f"  Action Probabilities: {[f'{p:.3f}' for p in recommendation['action_probabilities']]}")
    
    # Show training metrics
    metrics = trainer.get_training_metrics()
    print(f"\n📊 Training Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ CRL-HF Trainer demonstration complete!")

if __name__ == "__main__":
    main()