abstract class OrbRepository {
  Future<List<Map<String, dynamic>>> getSuggestions(String userId);
  Future<Map<String, dynamic>> getMemoryGraph(String userId);
  Future<void> submitFeedback(String suggestionId, bool accepted);
}

class OrbRepositoryImpl implements OrbRepository {
  @override
  Future<List<Map<String, dynamic>>> getSuggestions(String userId) async {
    // Mock data for demo
    await Future.delayed(const Duration(milliseconds: 500));
    
    return [
      {
        'id': '1',
        'content': 'You mentioned learning Python. Here\'s a tutorial on functions.',
        'type': 'memory-based',
        'confidence': 0.85,
        'priority': 'high',
      },
      {
        'id': '2',
        'content': 'Based on your recent activity, try using pdb for debugging.',
        'type': 'pattern-based',
        'confidence': 0.78,
        'priority': 'medium',
      },
      {
        'id': '3',
        'content': 'Your productivity peaks at 2 PM. Schedule deep work then.',
        'type': 'contextual',
        'confidence': 0.72,
        'priority': 'medium',
      },
    ];
  }
  
  @override
  Future<Map<String, dynamic>> getMemoryGraph(String userId) async {
    // Mock memory graph data
    await Future.delayed(const Duration(milliseconds: 300));
    
    return {
      'nodes': [
        {'id': '1', 'content': 'Python Programming', 'x': 0.2, 'y': 0.3},
        {'id': '2', 'content': 'Machine Learning', 'x': 0.6, 'y': 0.2},
        {'id': '3', 'content': 'Web Development', 'x': 0.3, 'y': 0.7},
      ],
      'connections': [
        {'from': '1', 'to': '2'},
        {'from': '1', 'to': '3'},
      ],
    };
  }
  
  @override
  Future<void> submitFeedback(String suggestionId, bool accepted) async {
    // Submit feedback to backend
    await Future.delayed(const Duration(milliseconds: 200));
    
    // In production, this would send data to the API
    print('Feedback submitted: suggestionId=$suggestionId, accepted=$accepted');
  }
}