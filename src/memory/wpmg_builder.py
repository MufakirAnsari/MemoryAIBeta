#!/usr/bin/env python3
"""
MemoryAI Enterprise - Weighted Personal Memory Graph (WPMG) Builder
Constructs personalized knowledge graphs with temporal and semantic weighting
"""

import os
import sys
import json
import sqlite3
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict

class EdgeType(Enum):
    """Types of edges in memory graph"""
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    ASSOCIATIVE = "associative"
    PERSONAL = "personal"
    CONTEXTUAL = "contextual"

@dataclass
class MemoryNode:
    """Node in personal memory graph"""
    node_id: str
    content: str
    content_type: str  # 'fact', 'preference', 'experience', 'goal', 'context'
    timestamp: datetime
    source: str
    confidence: float = 1.0
    importance: float = 0.5
    privacy_level: int = 2  # 0=public, 1=friends, 2=private
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'content': self.content,
            'content_type': self.content_type,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'confidence': self.confidence,
            'importance': self.importance,
            'privacy_level': self.privacy_level,
            'metadata': self.metadata
        }

@dataclass
class MemoryEdge:
    """Edge in personal memory graph"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'edge_type': self.edge_type.value,
            'weight': self.weight,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'metadata': self.metadata
        }

class WPMGBuilder:
    """Weighted Personal Memory Graph Builder"""
    
    def __init__(self, user_id: str, db_path: str = None):
        self.user_id = user_id
        self.db_path = db_path or f"/tmp/wpmg_{user_id}.db"
        self.graph = nx.MultiDiGraph()
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # TF-IDF for semantic similarity
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Temporal decay parameters
        self.half_life_days = 30  # Memory importance halves every 30 days
        self.recency_weight = 0.3
        self.frequency_weight = 0.4
        self.semantic_weight = 0.3
        
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS memory_nodes (
                node_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                importance REAL DEFAULT 0.5,
                privacy_level INTEGER DEFAULT 2,
                metadata TEXT DEFAULT '{}'
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS memory_edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                timestamp TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                metadata TEXT DEFAULT '{}',
                PRIMARY KEY (source_id, target_id, edge_type)
            )
        ''')
        
        self.conn.commit()
    
    def add_memory(
        self, 
        content: str, 
        content_type: str, 
        source: str,
        confidence: float = 1.0,
        importance: float = 0.5,
        privacy_level: int = 2,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a new memory node to the graph"""
        
        # Generate unique node ID
        timestamp = datetime.now()
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        node_id = f"{content_type}_{timestamp.strftime('%Y%m%d%H%M%S')}_{content_hash}"
        
        # Create memory node
        node = MemoryNode(
            node_id=node_id,
            content=content,
            content_type=content_type,
            timestamp=timestamp,
            source=source,
            confidence=confidence,
            importance=importance,
            privacy_level=privacy_level,
            metadata=metadata or {}
        )
        
        # Add to graph
        self.graph.add_node(node_id, **node.to_dict())
        
        # Store in database
        self.conn.execute('''
            INSERT INTO memory_nodes 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node_id, content, content_type, timestamp.isoformat(),
            source, confidence, importance, privacy_level,
            json.dumps(metadata or {})
        ))
        self.conn.commit()
        
        # Create semantic edges to similar memories
        self._create_semantic_edges(node)
        
        # Create temporal edges to recent memories
        self._create_temporal_edges(node)
        
        self.logger.info(f"✅ Added memory node: {node_id}")
        return node_id
    
    def _create_semantic_edges(self, node: MemoryNode, top_k: int = 5):
        """Create semantic edges to similar memories"""
        
        # Get candidate nodes for similarity comparison
        candidates = self._get_similarity_candidates(node)
        
        if not candidates:
            return
        
        # Prepare texts for vectorization
        texts = [node.content] + [c.content for c in candidates]
        
        # Compute TF-IDF vectors
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Create edges to top-k most similar nodes
            top_indices = np.argsort(similarities)[-top_k:]
            
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    self._add_edge(
                        source_id=node.node_id,
                        target_id=candidates[idx].node_id,
                        edge_type=EdgeType.SEMANTIC,
                        weight=similarities[idx],
                        confidence=node.confidence * candidates[idx].confidence
                    )
                    
        except Exception as e:
            self.logger.warning(f"Semantic edge creation failed: {e}")
    
    def _create_temporal_edges(self, node: MemoryNode, time_window_hours: int = 24):
        """Create temporal edges to recent memories"""
        
        time_threshold = node.timestamp - timedelta(hours=time_window_hours)
        
        # Query recent nodes
        cursor = self.conn.execute('''
            SELECT node_id FROM memory_nodes 
            WHERE timestamp > ? AND node_id != ?
            ORDER BY timestamp DESC LIMIT 10
        ''', (time_threshold.isoformat(), node.node_id))
        
        recent_node_ids = [row[0] for row in cursor.fetchall()]
        
        # Create temporal edges
        for recent_id in recent_node_ids:
            # Temporal decay weight
            time_diff = (node.timestamp - self.graph.nodes[recent_id]['timestamp']).total_seconds()
            weight = np.exp(-time_diff / (self.half_life_days * 24 * 3600))
            
            self._add_edge(
                source_id=recent_id,
                target_id=node.node_id,
                edge_type=EdgeType.TEMPORAL,
                weight=weight,
                confidence=1.0
            )
    
    def _get_similarity_candidates(self, node: MemoryNode, max_candidates: int = 20) -> List[MemoryNode]:
        """Get candidate nodes for semantic similarity"""
        
        # Query nodes of same type or related types
        related_types = {
            'fact': ['fact', 'experience'],
            'preference': ['preference', 'context'],
            'experience': ['experience', 'fact'],
            'goal': ['goal', 'preference'],
            'context': ['context', 'preference']
        }
        
        candidate_types = related_types.get(node.content_type, [node.content_type])
        
        cursor = self.conn.execute('''
            SELECT * FROM memory_nodes 
            WHERE content_type IN ({}) AND node_id != ?
            ORDER BY timestamp DESC LIMIT ?
        '''.format(','.join(['?' for _ in candidate_types])), 
        candidate_types + [node.node_id, max_candidates])
        
        candidates = []
        for row in cursor.fetchall():
            candidates.append(MemoryNode(
                node_id=row['node_id'],
                content=row['content'],
                content_type=row['content_type'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                source=row['source'],
                confidence=row['confidence'],
                importance=row['importance'],
                privacy_level=row['privacy_level'],
                metadata=json.loads(row['metadata'])
            ))
        
        return candidates
    
    def _add_edge(
        self, 
        source_id: str, 
        target_id: str, 
        edge_type: EdgeType, 
        weight: float,
        confidence: float = 1.0,
        metadata: Dict[str, Any] = None
    ):
        """Add an edge to the graph"""
        
        timestamp = datetime.now()
        
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            timestamp=timestamp,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Add to graph
        self.graph.add_edge(source_id, target_id, **edge.to_dict())
        
        # Store in database
        self.conn.execute('''
            INSERT OR REPLACE INTO memory_edges 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            source_id, target_id, edge_type.value, weight,
            timestamp.isoformat(), confidence, json.dumps(metadata or {})
        ))
        self.conn.commit()
    
    def query_memories(
        self, 
        query: str, 
        content_type: str = None,
        limit: int = 10,
        include_private: bool = False
    ) -> List[Dict[str, Any]]:
        """Query memories similar to query string"""
        
        # Build SQL query
        sql = '''
            SELECT * FROM memory_nodes 
            WHERE 1=1
        '''
        params = []
        
        if content_type:
            sql += ' AND content_type = ?'
            params.append(content_type)
        
        if not include_private:
            sql += ' AND privacy_level <= 1'
        
        sql += ' ORDER BY importance DESC, timestamp DESC LIMIT ?'
        params.append(limit * 2)  # Get more candidates for semantic filtering
        
        cursor = self.conn.execute(sql, params)
        candidates = [dict(row) for row in cursor.fetchall()]
        
        if not candidates:
            return []
        
        # Semantic filtering
        try:
            texts = [query] + [c['content'] for c in candidates]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Filter by similarity and return top results
            results = []
            for i, candidate in enumerate(candidates):
                if similarities[i] > 0.1:  # Minimum similarity threshold
                    candidate['similarity_score'] = similarities[i]
                    candidate['timestamp'] = datetime.fromisoformat(candidate['timestamp'])
                    results.append(candidate)
            
            # Sort by similarity and return top limit
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            self.logger.warning(f"Semantic search failed: {e}")
            # Fallback to importance-based ranking
            return candidates[:limit]
    
    def get_memory_context(
        self, 
        node_id: str, 
        hop_distance: int = 2,
        edge_types: List[EdgeType] = None
    ) -> Dict[str, Any]:
        """Get contextual memories around a specific node"""
        
        if node_id not in self.graph:
            return {'nodes': [], 'edges': []}
        
        # Get subgraph within hop distance
        if hop_distance > 0:
            subgraph_nodes = nx.single_source_shortest_path_length(
                self.graph, node_id, hop_distance
            )
            subgraph = self.graph.subgraph(subgraph_nodes.keys())
        else:
            subgraph = self.graph.subgraph([node_id])
        
        # Filter edges by type if specified
        if edge_types:
            edge_filter = lambda u, v, d: EdgeType(d['edge_type']) in edge_types
            subgraph = nx.edge_subgraph(subgraph, [
                (u, v) for u, v, d in subgraph.edges(data=True) 
                if edge_filter(u, v, d)
            ])
        
        # Convert to serializable format
        nodes = []
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            nodes.append({
                'node_id': node_id,
                'content': node_data['content'],
                'content_type': node_data['content_type'],
                'importance': node_data['importance'],
                'confidence': node_data['confidence']
            })
        
        edges = []
        for u, v, d in subgraph.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'weight': d['weight'],
                'type': d['edge_type']
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'centrality': self._compute_centrality(subgraph, node_id)
        }
    
    def _compute_centrality(self, graph: nx.Graph, focus_node: str) -> Dict[str, float]:
        """Compute centrality metrics for context"""
        
        try:
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph, k=min(100, len(graph)))
            
            return {
                'degree': degree_centrality.get(focus_node, 0),
                'betweenness': betweenness_centrality.get(focus_node, 0),
                'importance': self.graph.nodes[focus_node].get('importance', 0)
            }
        except:
            return {'degree': 0, 'betweenness': 0, 'importance': 0}
    
    def update_memory_importance(self, node_id: str, interaction_type: str):
        """Update memory importance based on user interaction"""
        
        if node_id not in self.graph:
            return
        
        # Importance update factors
        factors = {
            'view': 1.1,
            'click': 1.3,
            'save': 1.5,
            'share': 2.0,
            'edit': 1.8
        }
        
        factor = factors.get(interaction_type, 1.0)
        current_importance = self.graph.nodes[node_id]['importance']
        new_importance = min(1.0, current_importance * factor)
        
        # Update graph and database
        self.graph.nodes[node_id]['importance'] = new_importance
        
        self.conn.execute('''
            UPDATE memory_nodes 
            SET importance = ?, last_accessed = ?
            WHERE node_id = ?
        ''', (new_importance, datetime.now().isoformat(), node_id))
        self.conn.commit()
    
    def export_graph(self, format: str = 'json') -> str:
        """Export memory graph for analysis or backup"""
        
        if format == 'json':
            # NetworkX JSON export
            graph_data = json_graph.node_link_data(self.graph)
            
            # Convert datetime objects to strings
            for node in graph_data['nodes']:
                if 'timestamp' in node:
                    node['timestamp'] = str(node['timestamp'])
            
            for link in graph_data['links']:
                if 'timestamp' in link:
                    link['timestamp'] = str(link['timestamp'])
            
            return json.dumps(graph_data, indent=2, default=str)
        
        elif format == 'gexf':
            # GEXF format for Gephi visualization
            return '
'.join(nx.generate_gexf(self.graph))
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory graph"""
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'content_type_distribution': self._get_content_type_stats(),
            'average_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.nodes else 0,
            'clustering_coefficient': nx.average_clustering(self.graph.to_undirected()) if self.graph.nodes else 0,
            'memory_age_days': self._get_memory_age_stats()
        }
    
    def _get_content_type_stats(self) -> Dict[str, int]:
        """Get distribution of content types"""
        stats = defaultdict(int)
        for node_id in self.graph.nodes():
            content_type = self.graph.nodes[node_id]['content_type']
            stats[content_type] += 1
        return dict(stats)
    
    def _get_memory_age_stats(self) -> Dict[str, float]:
        """Get statistics about memory age"""
        if not self.graph.nodes:
            return {'oldest': 0, 'newest': 0, 'average': 0}
        
        now = datetime.now()
        ages = []
        
        for node_id in self.graph.nodes():
            timestamp = self.graph.nodes[node_id]['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            age_days = (now - timestamp).total_seconds() / 86400
            ages.append(age_days)
        
        return {
            'oldest': max(ages),
            'newest': min(ages),
            'average': np.mean(ages)
        }

# Global WPMG builder instance
_global_wpmg_builders = {}

def get_wpmg_builder(user_id: str) -> WPMGBuilder:
    """Get WPMG builder for specific user"""
    if user_id not in _global_wpmg_builders:
        _global_wpmg_builders[user_id] = WPMGBuilder(user_id)
    return _global_wpmg_builders[user_id]

def main():
    """Demo WPMG builder functionality"""
    
    logging.basicConfig(level=logging.INFO)
    
    # Create demo user
    user_id = "demo_user_123"
    builder = get_wpmg_builder(user_id)
    
    # Add sample memories
    sample_memories = [
        ("I love hiking in the mountains", "preference", 0.8),
        ("My birthday is March 15th", "fact", 1.0),
        ("Went hiking at Yosemite last weekend", "experience", 0.9),
        ("Goal: Run a marathon this year", "goal", 0.7),
        ("Currently working from home", "context", 0.6),
        ("Favorite food is Italian cuisine", "preference", 0.7),
        ("Visited Japan in 2019", "experience", 0.8),
        ("Learning Python programming", "goal", 0.9),
    ]
    
    print("🏗️  Building Weighted Personal Memory Graph...")
    print("Adding sample memories:\n")
    
    for content, content_type, importance in sample_memories:
        node_id = builder.add_memory(
            content=content,
            content_type=content_type,
            source="user_input",
            importance=importance,
            privacy_level=2
        )
        print(f"  ✅ Added: {content[:50]}...")
    
    print(f"\n📊 Memory Graph Statistics:")
    stats = builder.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Query memories
    print(f"\n🔍 Querying memories about 'hiking':")
    results = builder.query_memories("hiking", limit=5)
    for result in results:
        print(f"  • {result['content']} (similarity: {result['similarity_score']:.3f})")
    
    # Get memory context
    print(f"\n🕸️  Memory context around first hiking memory:")
    context = builder.get_memory_context(list(builder.graph.nodes())[0], hop_distance=2)
    print(f"  Context contains {len(context['nodes'])} nodes and {len(context['edges'])} edges")
    
    # Export graph
    print(f"\n📄 Exporting memory graph...")
    graph_json = builder.export_graph('json')
    with open('/mnt/okcomputer/memoryai-enterprise/src/memory/wpmg_demo.json', 'w') as f:
        f.write(graph_json)
    print(f"  ✅ Graph exported to wpmg_demo.json")
    
    print(f"\n✅ WPMG Builder demonstration complete!")

if __name__ == "__main__":
    main()