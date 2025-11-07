part of 'orb_bloc.dart';

abstract class OrbEvent {}

class LoadSuggestions extends OrbEvent {
  final String userId;
  
  LoadSuggestions(this.userId);
}

class SubmitFeedback extends OrbEvent {
  final String suggestionId;
  final bool accepted;
  
  SubmitFeedback({
    required this.suggestionId,
    required this.accepted,
  });
}

class ToggleMode extends OrbEvent {
  final bool isHaloMode;
  
  ToggleMode(this.isHaloMode);
}