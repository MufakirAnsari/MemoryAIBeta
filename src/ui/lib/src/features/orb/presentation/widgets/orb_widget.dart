import 'package:flutter/material.dart';
import 'package:glassmorphism/glassmorphism.dart';
import 'package:memoryai_ui/src/core/theme/app_theme.dart';

class OrbWidget extends StatefulWidget {
  final bool isHaloMode;
  final VoidCallback onTap;
  final VoidCallback onLongPress;
  
  const OrbWidget({
    super.key,
    required this.isHaloMode,
    required this.onTap,
    required this.onLongPress,
  });

  @override
  State<OrbWidget> createState() => _OrbWidgetState();
}

class _OrbWidgetState extends State<OrbWidget> with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;
  
  bool _isHovered = false;
  bool _isThinking = false;
  
  @override
  void initState() {
    super.initState();
    
    _pulseController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: true);
    
    _pulseAnimation = Tween<double>(
      begin: 1.0,
      end: 1.1,
    ).animate(CurvedAnimation(
      parent: _pulseController,
      curve: Curves.easeInOut,
    ));
  }
  
  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }
  
  void _handleHover(bool isHovering) {
    setState(() {
      _isHovered = isHovering;
    });
  }
  
  void _simulateThinking() {
    setState(() {
      _isThinking = true;
    });
    
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) {
        setState(() {
          _isThinking = false;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      onEnter: (_) => _handleHover(true),
      onExit: (_) => _handleHover(false),
      child: GestureDetector(
        onTap: widget.onTap,
        onLongPress: widget.onLongPress,
        onDoubleTap: _simulateThinking,
        child: AnimatedBuilder(
          animation: _pulseAnimation,
          builder: (context, child) {
            return Transform.scale(
              scale: _isHovered ? 1.05 : _pulseAnimation.value,
              child: child,
            );
          },
          child: Stack(
            alignment: Alignment.center,
            children: [
              // Outer glow effect
              if (widget.isHaloMode)
                Container(
                  width: 280,
                  height: 280,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        AppTheme.teenTeal.withOpacity(0.3),
                        AppTheme.teenPurple.withOpacity(0.2),
                        Colors.transparent,
                      ],
                      stops: const [0.3, 0.6, 1.0],
                    ),
                  ),
                )
              else
                Container(
                  width: 260,
                  height: 260,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        AppTheme.teenPink.withOpacity(0.4),
                        AppTheme.accentColor.withOpacity(0.2),
                        Colors.transparent,
                      ],
                      stops: const [0.3, 0.6, 1.0],
                    ),
                  ),
                ),
              
              // Main orb
              Glassmorphism(
                blur: 25,
                opacity: 0.15,
                radius: 120,
                child: Container(
                  width: 240,
                  height: 240,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: LinearGradient(
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                      colors: widget.isHaloMode
                          ? [
                              AppTheme.teenTeal.withOpacity(0.3),
                              AppTheme.teenPurple.withOpacity(0.2),
                            ]
                          : [
                              AppTheme.teenPink.withOpacity(0.3),
                              AppTheme.accentColor.withOpacity(0.2),
                            ],
                    ),
                    border: Border.all(
                      color: Colors.white.withOpacity(0.3),
                      width: 2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: (widget.isHaloMode ? AppTheme.teenTeal : AppTheme.teenPink)
                            .withOpacity(0.3),
                        blurRadius: 30,
                        spreadRadius: 10,
                      ),
                    ],
                  ),
                  child: Center(
                    child: _buildOrbContent(),
                  ),
                ),
              ),
              
              // Inner core
              Glassmorphism(
                blur: 15,
                opacity: 0.25,
                radius: 60,
                child: Container(
                  width: 120,
                  height: 120,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    gradient: RadialGradient(
                      colors: [
                        Colors.white.withOpacity(0.4),
                        Colors.white.withOpacity(0.1),
                      ],
                    ),
                  ),
                ),
              ),
              
              // Thinking indicator
              if (_isThinking)
                Positioned(
                  top: 20,
                  child: _buildThinkingIndicator(),
                ),
              
              // Status indicator
              Positioned(
                bottom: 20,
                child: _buildStatusIndicator(),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildOrbContent() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          widget.isHaloMode ? Icons.light_mode : Icons.mode_night,
          size: 48,
          color: Colors.white.withOpacity(0.9),
        ),
        const SizedBox(height: 8),
        Text(
          widget.isHaloMode ? 'Halo' : 'Focus',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          _isThinking ? 'Thinking...' : 'Ready',
          style: TextStyle(
            color: Colors.white.withOpacity(0.7),
            fontSize: 12,
          ),
        ),
      ],
    );
  }
  
  Widget _buildThinkingIndicator() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: List.generate(3, (index) {
        return AnimatedOpacity(
          opacity: _isThinking ? 1.0 : 0.0,
          duration: const Duration(milliseconds: 300),
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 2),
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: AppTheme.teenGreen,
              shape: BoxShape.circle,
            ),
          ),
        );
      }),
    );
  }
  
  Widget _buildStatusIndicator() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.2),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: _isThinking 
                  ? AppTheme.teenOrange
                  : AppTheme.teenGreen,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            _isThinking ? 'Processing' : 'Active',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 10,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}