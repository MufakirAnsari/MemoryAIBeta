import 'package:flutter/material.dart';
import 'package:glassmorphism/glassmorphism.dart';
import 'package:memoryai_ui/src/core/theme/app_theme.dart';

class SuggestionPanel extends StatefulWidget {
  final bool isVisible;
  final VoidCallback onClose;
  
  const SuggestionPanel({
    super.key,
    required this.isVisible,
    required this.onClose,
  });

  @override
  State<SuggestionPanel> createState() => _SuggestionPanelState();
}

class _SuggestionPanelState extends State<SuggestionPanel> {
  final List<SuggestionData> _suggestions = [
    SuggestionData(
      id: '1',
      content: 'You mentioned learning Python. Here\'s a tutorial on functions.',
      type: 'memory-based',
      confidence: 0.85,
      priority: 'high',
    ),
    SuggestionData(
      id: '2',
      content: 'Based on your recent activity, try using pdb for debugging.',
      type: 'pattern-based',
      confidence: 0.78,
      priority: 'medium',
    ),
    SuggestionData(
      id: '3',
      content: 'Your productivity peaks at 2 PM. Schedule deep work then.',
      type: 'contextual',
      confidence: 0.72,
      priority: 'medium',
    ),
    SuggestionData(
      id: '4',
      content: 'Good morning! Ready to review your goals for today?',
      type: 'proactive',
      confidence: 0.65,
      priority: 'low',
    ),
  ];
  
  @override
  Widget build(BuildContext context) {
    return AnimatedSlide(
      offset: widget.isVisible ? Offset.zero : const Offset(0, 1),
      duration: const Duration(milliseconds: 400),
      curve: Curves.easeOutCubic,
      child: Glassmorphism(
        blur: 25,
        opacity: 0.15,
        radius: 20,
        child: Container(
          padding: const EdgeInsets.all(16),
          constraints: const BoxConstraints(
            maxHeight: 300,
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildHeader(),
              const SizedBox(height: 16),
              Expanded(
                child: ListView.builder(
                  shrinkWrap: true,
                  itemCount: _suggestions.length,
                  itemBuilder: (context, index) {
                    return _buildSuggestionCard(_suggestions[index]);
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildHeader() {
    return Row(
      children: [
        const Icon(
          Icons.lightbulb_outline,
          color: AppTheme.teenOrange,
          size: 24,
        ),
        const SizedBox(width: 8),
        const Text(
          'Smart Suggestions',
          style: TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
        const Spacer(),
        IconButton(
          icon: const Icon(Icons.close, color: Colors.white, size: 20),
          onPressed: widget.onClose,
          tooltip: 'Close',
        ),
      ],
    );
  }
  
  Widget _buildSuggestionCard(SuggestionData suggestion) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      child: Glassmorphism(
        blur: 15,
        opacity: 0.1,
        radius: 12,
        child: InkWell(
          onTap: () => _handleSuggestionTap(suggestion),
          borderRadius: BorderRadius.circular(12),
          child: Container(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    _buildTypeBadge(suggestion.type),
                    const SizedBox(width: 8),
                    _buildPriorityIndicator(suggestion.priority),
                    const Spacer(),
                    _buildConfidenceIndicator(suggestion.confidence),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  suggestion.content,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.w400,
                  ),
                ),
                const SizedBox(height: 8),
                Row(
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    _buildActionButton(
                      icon: Icons.thumb_up,
                      label: 'Use',
                      color: AppTheme.teenGreen,
                      onTap: () => _handleSuggestionAccept(suggestion),
                    ),
                    const SizedBox(width: 8),
                    _buildActionButton(
                      icon: Icons.thumb_down,
                      label: 'Skip',
                      color: AppTheme.teenOrange,
                      onTap: () => _handleSuggestionReject(suggestion),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
  
  Widget _buildTypeBadge(String type) {
    Color color;
    IconData icon;
    
    switch (type) {
      case 'memory-based':
        color = AppTheme.teenPurple;
        icon = Icons.memory;
        break;
      case 'pattern-based':
        color = AppTheme.teenTeal;
        icon = Icons.pattern;
        break;
      case 'contextual':
        color = AppTheme.teenPink;
        icon = Icons.context;
        break;
      case 'proactive':
        color = AppTheme.teenOrange;
        icon = Icons.lightbulb;
        break;
      default:
        color = AppTheme.teenGreen;
        icon = Icons.help;
    }
    
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.3),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: color.withOpacity(0.5),
          width: 1,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: color),
          const SizedBox(width: 4),
          Text(
            type.replaceAll('-', ' '),
            style: TextStyle(
              color: color,
              fontSize: 10,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildPriorityIndicator(String priority) {
    Color color;
    switch (priority) {
      case 'high':
        color = Colors.red;
        break;
      case 'medium':
        color = Colors.orange;
        break;
      case 'low':
        color = Colors.green;
        break;
      default:
        color = Colors.grey;
    }
    
    return Container(
      width: 8,
      height: 8,
      decoration: BoxDecoration(
        color: color,
        shape: BoxShape.circle,
      ),
    );
  }
  
  Widget _buildConfidenceIndicator(double confidence) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.2),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        '${(confidence * 100).toStringAsFixed(0)}%',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 10,
          fontWeight: FontWeight.w500,
        ),
      ),
    );
  }
  
  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        decoration: BoxDecoration(
          color: color.withOpacity(0.2),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(
            color: color.withOpacity(0.5),
            width: 1,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: color),
            const SizedBox(width: 4),
            Text(
              label,
              style: TextStyle(
                color: color,
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  void _handleSuggestionTap(SuggestionData suggestion) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('💡 ${suggestion.content}'),
        backgroundColor: AppTheme.primaryColor,
      ),
    );
  }
  
  void _handleSuggestionAccept(SuggestionData suggestion) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('✅ Accepted: ${suggestion.content}'),
        backgroundColor: AppTheme.teenGreen,
      ),
    );
    
    // Remove from list
    setState(() {
      _suggestions.remove(suggestion);
    });
  }
  
  void _handleSuggestionReject(SuggestionData suggestion) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('❌ Skipped: ${suggestion.content}'),
        backgroundColor: AppTheme.teenOrange,
      ),
    );
    
    // Remove from list
    setState(() {
      _suggestions.remove(suggestion);
    });
  }
}

@immutable
class SuggestionData {
  final String id;
  final String content;
  final String type;
  final double confidence;
  final String priority;
  
  const SuggestionData({
    required this.id,
    required this.content,
    required this.type,
    required this.confidence,
    required this.priority,
  });
}