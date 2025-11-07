import 'package:flutter/material.dart';
import 'package:glassmorphism/glassmorphism.dart';
import 'package:memoryai_ui/src/core/theme/app_theme.dart';
import 'package:memoryai_ui/src/features/orb/presentation/widgets/orb_widget.dart';
import 'package:memoryai_ui/src/features/orb/presentation/widgets/mode_switcher.dart';
import 'package:memoryai_ui/src/features/orb/presentation/widgets/suggestion_panel.dart';
import 'package:memoryai_ui/src/features/orb/presentation/widgets/memory_graph.dart';

class OrbPage extends StatefulWidget {
  const OrbPage({super.key});

  @override
  State<OrbPage> createState() => _OrbPageState();
}

class _OrbPageState extends State<OrbPage> with SingleTickerProviderStateMixin {
  bool _isHaloMode = true;
  bool _showSuggestions = true;
  bool _showMemoryGraph = false;
  
  late AnimationController _animationController;
  late Animation<double> _orbAnimation;
  
  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(seconds: 4),
      vsync: this,
    )..repeat(reverse: true);
    
    _orbAnimation = Tween<double>(
      begin: 0.8,
      end: 1.2,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    ));
  }
  
  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }
  
  void _toggleMode() {
    setState(() {
      _isHaloMode = !_isHaloMode;
    });
  }
  
  void _toggleSuggestions() {
    setState(() {
      _showSuggestions = !_showSuggestions;
    });
  }
  
  void _toggleMemoryGraph() {
    setState(() {
      _showMemoryGraph = !_showMemoryGraph;
    });
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        title: const Text('MemoryAI'),
        actions: [
          IconButton(
            icon: const Icon(Icons.bubble_chart),
            onPressed: _toggleMemoryGraph,
            tooltip: 'Memory Graph',
          ),
          IconButton(
            icon: const Icon(Icons.lightbulb_outline),
            onPressed: _toggleSuggestions,
            tooltip: 'Suggestions',
          ),
        ],
      ),
      body: Stack(
        children: [
          // Background with gradient
          Container(
            decoration: BoxDecoration(
              gradient: _isHaloMode ? AppTheme.haloGradient : AppTheme.focusGradient,
            ),
          ),
          
          // Animated background particles
          _buildParticleEffect(),
          
          // Main content
          SafeArea(
            child: Column(
              children: [
                // Mode switcher
                ModeSwitcher(
                  isHaloMode: _isHaloMode,
                  onToggle: _toggleMode,
                ),
                
                // Central orb
                Expanded(
                  child: Center(
                    child: AnimatedBuilder(
                      animation: _orbAnimation,
                      builder: (context, child) {
                        return Transform.scale(
                          scale: _orbAnimation.value,
                          child: child,
                        );
                      },
                      child: OrbWidget(
                        isHaloMode: _isHaloMode,
                        onTap: _handleOrbTap,
                        onLongPress: _handleOrbLongPress,
                      ),
                    ),
                  ),
                ),
                
                // Bottom controls
                _buildBottomControls(),
              ],
            ),
          ),
          
          // Suggestion panel
          if (_showSuggestions)
            Positioned(
              bottom: 120,
              left: 20,
              right: 20,
              child: SuggestionPanel(
                isVisible: _showSuggestions,
                onClose: _toggleSuggestions,
              ),
            ),
          
          // Memory graph overlay
          if (_showMemoryGraph)
            Positioned.fill(
              child: MemoryGraph(
                onClose: _toggleMemoryGraph,
              ),
            ),
        ],
      ),
    );
  }
  
  Widget _buildParticleEffect() {
    return Stack(
      children: List.generate(20, (index) {
        return AnimatedPositioned(
          duration: Duration(seconds: 3 + index % 3),
          left: (index * 50.0) % MediaQuery.of(context).size.width,
          top: (index * 30.0) % MediaQuery.of(context).size.height,
          child: Container(
            width: 4 + (index % 3),
            height: 4 + (index % 3),
            decoration: BoxDecoration(
              color: _isHaloMode 
                  ? AppTheme.teenTeal.withOpacity(0.3)
                  : AppTheme.teenPink.withOpacity(0.3),
              shape: BoxShape.circle,
            ),
          ),
        );
      }),
    );
  }
  
  Widget _buildBottomControls() {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _buildGlassButton(
            icon: Icons.mic_none,
            label: 'Voice',
            onPressed: _handleVoiceInput,
          ),
          _buildGlassButton(
            icon: Icons.keyboard,
            label: 'Text',
            onPressed: _handleTextInput,
          ),
          _buildGlassButton(
            icon: Icons.camera_alt,
            label: 'Vision',
            onPressed: _handleVisionInput,
          ),
        ],
      ),
    );
  }
  
  Widget _buildGlassButton({
    required IconData icon,
    required String label,
    required VoidCallback onPressed,
  }) {
    return Glassmorphism(
      blur: 20,
      opacity: 0.1,
      radius: 16,
      child: InkWell(
        onTap: onPressed,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 24, color: Colors.white),
              const SizedBox(height: 4),
              Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  void _handleOrbTap() {
    // Handle orb tap - show quick suggestions
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('✨ What can I help you with today?'),
        backgroundColor: AppTheme.teenTeal,
      ),
    );
  }
  
  void _handleOrbLongPress() {
    // Handle orb long press - switch to focus mode
    setState(() {
      _isHaloMode = false;
    });
    
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('🎯 Focus mode activated'),
        backgroundColor: AppTheme.teenPink,
      ),
    );
  }
  
  void _handleVoiceInput() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('🎤 Listening...'),
        backgroundColor: AppTheme.teenGreen,
      ),
    );
  }
  
  void _handleTextInput() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return Glassmorphism(
          blur: 20,
          opacity: 0.1,
          radius: 20,
          child: Container(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'What would you like to know?',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 16),
                TextField(
                  decoration: InputDecoration(
                    hintText: 'Type your question...',
                    hintStyle: TextStyle(color: Colors.white.withOpacity(0.6)),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                      borderSide: BorderSide.none,
                    ),
                    filled: true,
                    fillColor: Colors.white.withOpacity(0.1),
                  ),
                  style: const TextStyle(color: Colors.white),
                  onSubmitted: (value) {
                    Navigator.pop(context);
                    _handleQuerySubmitted(value);
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }
  
  void _handleVisionInput() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('📷 Camera mode - Coming soon!'),
        backgroundColor: AppTheme.teenOrange,
      ),
    );
  }
  
  void _handleQuerySubmitted(String query) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('🔍 Searching for: $query'),
        backgroundColor: AppTheme.primaryColor,
      ),
    );
  }
}