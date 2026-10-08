import pytest


extensions = pytest.importorskip(
    "neuromind.extensions",
    reason="Extension API is not present in neuromind-v0.1 yet",
)


def test_extension_module_imports():
    assert extensions is not None


def test_extension_registry_or_entrypoint_exists():
    candidates = (
        "ExtensionRegistry",
        "NeuroMindExtension",
        "register_extension",
        "register",
    )
    assert any(hasattr(extensions, name) for name in candidates), (
        "neuromind.extensions must expose an extension registry, extension "
        "class, or registration function"
    )
