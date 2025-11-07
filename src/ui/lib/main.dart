import 'package:flutter/material.dart';
import 'package:memoryai_ui/src/app.dart';
import 'package:memoryai_ui/src/core/services/service_locator.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize service locator
  await ServiceLocator.init();
  
  runApp(const MemoryAIApp());
}