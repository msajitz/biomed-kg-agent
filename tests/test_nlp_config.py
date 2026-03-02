"""Test NLP configuration loading and management."""

import tempfile
from pathlib import Path

import pytest

from biomed_kg_agent.nlp.config import (
    LinkerConfig,
    NerConfig,
    get_default_config,
    load_ner_config,
)


def test_load_ner_config_default() -> None:
    """Test loading configuration with default values."""
    config = load_ner_config()

    # Check that it returns a NerConfig dataclass
    assert isinstance(config, NerConfig)
    assert isinstance(config.linker, LinkerConfig)

    # Check default values
    assert config.model_priorities["bionlp"] == 3
    assert config.model_priorities["craft"] == 2
    assert config.model_priorities["bc5cdr"] == 1

    # Check linker defaults
    assert config.linker.core_model == "en_core_sci_sm"
    assert config.linker.confidence_threshold == 0.7


def test_load_ner_config_with_custom_yaml() -> None:
    """Test loading configuration from custom YAML file."""
    yaml_content = """
model_priorities:
  bionlp: 2
  craft: 1
  bc5cdr: 3

linker:
  enabled: true
  core_model: "en_core_sci_sm"
  confidence_threshold: 0.8
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        config = load_ner_config(f.name)

        # Check model priorities
        assert config.model_priorities["bc5cdr"] == 3
        assert config.model_priorities["bionlp"] == 2
        assert config.model_priorities["craft"] == 1

        # Check linker config
        assert config.linker.core_model == "en_core_sci_sm"
        assert config.linker.confidence_threshold == 0.8


def test_load_ner_config_missing_file() -> None:
    """Test behavior when config file doesn't exist."""
    config = load_ner_config("/nonexistent/path/config.yaml")

    # Should return default configuration
    assert isinstance(config, NerConfig)
    assert config.model_priorities["bionlp"] == 3
    assert config.linker.core_model == "en_core_sci_sm"


def test_load_ner_config_invalid_yaml() -> None:
    """Test behavior with invalid YAML file."""
    invalid_yaml = "invalid: yaml: content: ["

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(invalid_yaml)
        config_path = f.name

    try:
        config = load_ner_config(config_path)
        # Should return default configuration
        assert isinstance(config, NerConfig)
        assert config.model_priorities["bionlp"] == 3
    finally:
        Path(config_path).unlink()


# Test for unknown use_case removed - parameterization no longer supported


def test_get_default_config() -> None:
    """Test get_default_config function."""
    config = get_default_config()

    assert isinstance(config, NerConfig)
    assert isinstance(config.linker, LinkerConfig)
    assert config.model_priorities["bionlp"] == 3
    assert config.linker.core_model == "en_core_sci_sm"


def test_linker_config_immutability() -> None:
    """Test LinkerConfig validation and modification behavior."""
    config = get_default_config()

    # Pydantic models are mutable, but we can test validation
    # Test that we can modify but validation still works
    config.linker.core_model = "en_core_sci_lg"
    assert config.linker.core_model == "en_core_sci_lg"

    # Test that validation prevents invalid values
    from pydantic import ValidationError

    # Test confidence threshold validation (must be 0-1)
    with pytest.raises(ValidationError):
        LinkerConfig(
            core_model="en_core_sci_sm", confidence_threshold=2.0  # Invalid: > 1.0
        )

    with pytest.raises(ValidationError):
        LinkerConfig(
            core_model="en_core_sci_sm", confidence_threshold=-0.1  # Invalid: < 0.0
        )

    # Test model priorities validation
    with pytest.raises(ValidationError):
        NerConfig(
            model_priorities={"bionlp": 0},  # Invalid: must be >= 1
            linker=LinkerConfig(core_model="en_core_sci_sm", confidence_threshold=0.7),
        )


def test_config_validation() -> None:
    """Test Pydantic validation features."""
    from pydantic import ValidationError

    # Test valid configuration
    valid_config = LinkerConfig(core_model="en_core_sci_sm", confidence_threshold=0.8)
    assert valid_config.confidence_threshold == 0.8

    # Test invalid confidence threshold
    with pytest.raises(ValidationError) as exc_info:
        LinkerConfig(core_model="en_core_sci_sm", confidence_threshold=1.5)  # Too high
    assert "confidence_threshold" in str(exc_info.value)

    # Test empty core model
    with pytest.raises(ValidationError):
        LinkerConfig(core_model="", confidence_threshold=0.7)  # Empty string

    # Test invalid model priorities
    with pytest.raises(ValidationError):
        NerConfig(
            model_priorities={},  # Empty dict
            linker=LinkerConfig(core_model="en_core_sci_sm", confidence_threshold=0.7),
        )


def test_config_type_coercion() -> None:
    """Test that Pydantic handles type coercion correctly."""
    # Test that Pydantic can handle dict input with string values
    config_dict = {"core_model": "en_core_sci_sm", "confidence_threshold": 0.8}

    config = LinkerConfig(**config_dict)

    assert config.confidence_threshold == 0.8
    assert isinstance(config.confidence_threshold, float)


# Tests for get_available_use_cases, get_available_linking_profiles,
# and simplified_linking_profiles removed - parameterization no longer supported


def test_config_attribute_access() -> None:
    """Test that configuration attributes are accessible with proper types."""
    config = load_ner_config()

    # Test model_priorities access
    assert isinstance(config.model_priorities, dict)
    assert all(isinstance(k, str) for k in config.model_priorities.keys())
    assert all(isinstance(v, int) for v in config.model_priorities.values())

    # Test linker config access
    assert isinstance(config.linker.core_model, str)
    assert isinstance(config.linker.confidence_threshold, float)
