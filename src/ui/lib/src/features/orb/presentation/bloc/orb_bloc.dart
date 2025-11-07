import 'dart:async';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:memoryai_ui/src/features/orb/domain/usecases/get_suggestions.dart';

part 'orb_event.dart';
part 'orb_state.dart';

class OrbBloc extends Bloc<OrbEvent, OrbState> {
  final GetSuggestions getSuggestions;
  
  OrbBloc({
    required this.getSuggestions,
  }) : super(OrbInitial()) {
    on<LoadSuggestions>(_onLoadSuggestions);
    on<SubmitFeedback>(_onSubmitFeedback);
    on<ToggleMode>(_onToggleMode);
  }
  
  Future<void> _onLoadSuggestions(
    LoadSuggestions event,
    Emitter<OrbState> emit,
  ) async {
    emit(OrbLoading());
    
    try {
      final suggestions = await getSuggestions(event.userId);
      emit(OrbLoaded(suggestions: suggestions));
    } catch (e) {
      emit(OrbError(message: 'Failed to load suggestions: $e'));
    }
  }
  
  Future<void> _onSubmitFeedback(
    SubmitFeedback event,
    Emitter<OrbState> emit,
  ) async {
    try {
      // In a real app, you would have a repository method for this
      print('Submitting feedback: ${event.suggestionId}, accepted: ${event.accepted}');
      
      // Remove the suggestion from the current list if accepted
      if (state is OrbLoaded) {
        final currentState = state as OrbLoaded;
        final updatedSuggestions = currentState.suggestions
            .where((s) => s['id'] != event.suggestionId)
            .toList();
        
        emit(OrbLoaded(suggestions: updatedSuggestions));
      }
    } catch (e) {
      print('Error submitting feedback: $e');
    }
  }
  
  void _onToggleMode(
    ToggleMode event,
    Emitter<OrbState> emit,
  ) {
    if (state is OrbLoaded) {
      final currentState = state as OrbLoaded;
      emit(OrbLoaded(
        suggestions: currentState.suggestions,
        isHaloMode: event.isHaloMode,
      ));
    }
  }
}