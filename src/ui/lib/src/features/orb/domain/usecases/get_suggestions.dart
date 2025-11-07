import 'package:memoryai_ui/src/features/orb/data/repositories/orb_repository.dart';

class GetSuggestions {
  final OrbRepository repository;
  
  GetSuggestions(this.repository);
  
  Future<List<Map<String, dynamic>>> call(String userId) async {
    return await repository.getSuggestions(userId);
  }
}