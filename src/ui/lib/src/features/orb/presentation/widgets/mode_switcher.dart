import 'package:flutter/material.dart';
import 'package:glassmorphism/glassmorphism.dart';
import 'package:memoryai_ui/src/core/theme/app_theme.dart';

class ModeSwitcher extends StatelessWidget {
  final bool isHaloMode;
  final VoidCallback onToggle;
  
  const ModeSwitcher({
    super.key,
    required this.isHaloMode,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(20),
      child: Glassmorphism(
        blur: 20,
        opacity: 0.1,
        radius: 25,
        child: Container(
          padding: const EdgeInsets.all(8),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildModeButton(
                icon: Icons.light_mode,
                label: 'Halo',
                isActive: isHaloMode,
                onTap: isHaloMode ? null : onToggle,
                color: AppTheme.teenTeal,
              ),
              const SizedBox(width: 8),
              _buildModeButton(
                icon: Icons.mode_night,
                label: 'Focus',
                isActive: !isHaloMode,
                onTap: !isHaloMode ? null : onToggle,
                color: AppTheme.teenPink,
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildModeButton({
    required IconData icon,
    required String label,
    required bool isActive,
    required VoidCallback? onTap,
    required Color color,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: isActive ? color.withOpacity(0.3) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isActive ? color : Colors.white.withOpacity(0.3),
            width: isActive ? 2 : 1,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              icon,
              size: 20,
              color: isActive ? Colors.white : Colors.white.withOpacity(0.7),
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                color: isActive ? Colors.white : Colors.white.withOpacity(0.7),
                fontSize: 14,
                fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
              ),
            ),
          ],
        ),
      ),
    );
  }
}