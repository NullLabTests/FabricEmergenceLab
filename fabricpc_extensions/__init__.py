"""Extensions to the FabricPC library for emergence experimentation."""

__version__ = "0.1.0"

# Lazy imports — fabricpc has circular import issues when loaded via
# certain sys.path configurations (e.g., when a sibling fabricpc/ exists).
# Use __getattr__ to defer imports until first access.
_imports = {
    "WorldModel": "fabricpc_extensions.world_model",
    "SharedMemory": "fabricpc_extensions.shared_memory",
    "CommunicationChannel": "fabricpc_extensions.communication",
    "PCGenome": "fabricpc_extensions.evolution",
    "Population": "fabricpc_extensions.evolution",
    "LLMInterpreter": "fabricpc_extensions.llm_interface",
    "build_experiment_summary": "fabricpc_extensions.llm_interface",
    "PCAgent": "fabricpc_extensions.agent",
    "AssociativeMemory": "fabricpc_extensions.agent",
    "PositionalMemory": "fabricpc_extensions.agent",
    "BehaviorTracker": "fabricpc_extensions.agent",
}

__all__ = sorted(_imports.keys())


def __getattr__(name):
    if name in _imports:
        import importlib

        module = importlib.import_module(_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
