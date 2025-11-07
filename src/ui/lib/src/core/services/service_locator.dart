import 'package:get_it/get_it.dart';
import 'package:memoryai_ui/src/features/orb/data/repositories/orb_repository.dart';
import 'package:memoryai_ui/src/features/orb/domain/usecases/get_suggestions.dart';
import 'package:memoryai_ui/src/features/orb/presentation/bloc/orb_bloc.dart';

final GetIt serviceLocator = GetIt.instance;

class ServiceLocator {
  static Future<void> init() async {
    // Register repositories
    serviceLocator.registerLazySingleton<OrbRepository>(
      () => OrbRepositoryImpl(),
    );
    
    // Register use cases
    serviceLocator.registerLazySingleton<GetSuggestions>(
      () => GetSuggestions(serviceLocator<OrbRepository>()),
    );
    
    // Register BLoCs
    serviceLocator.registerFactory<OrbBloc>(
      () => OrbBloc(
        getSuggestions: serviceLocator<GetSuggestions>(),
      ),
    );
  }
}