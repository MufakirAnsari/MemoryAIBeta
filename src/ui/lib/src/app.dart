import 'package:flutter/material.dart';
import 'package:memoryai_ui/src/core/theme/app_theme.dart';
import 'package:memoryai_ui/src/features/orb/presentation/pages/orb_page.dart';

class MemoryAIApp extends StatelessWidget {
  const MemoryAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MemoryAI Enterprise',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      darkTheme: AppTheme.darkTheme,
      themeMode: ThemeMode.system,
      home: const OrbPage(),
    );
  }
}