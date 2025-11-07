part of 'orb_bloc.dart';

abstract class OrbState {}

class OrbInitial extends OrbState {}

class OrbLoading extends OrbState {}

class OrbLoaded extends OrbState {
  final List<Map<String, dynamic>> suggestions;
  final bool isHaloMode;
  
  OrbLoaded({
    required this.suggestions,
    this.isHaloMode = true,
  });
}

class OrbError extends OrbState {
  final String message;
  
  OrbError({required this.message});
}