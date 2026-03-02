"""Configuration management for biomedical NER."""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "ner_config.yaml"


class LinkerConfig(BaseModel):
    """Configuration for entity linking settings with validation."""

    enabled: bool = Field(True, description="Enable/disable UMLS entity linking")
    core_model: str = Field(..., description="Core spaCy model for entity linking")
    confidence_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for entity linking",
    )

    @field_validator("core_model")
    @classmethod
    def validate_core_model(cls, v: str) -> str:
        """Validate that core model name is reasonable."""
        if not v or not isinstance(v, str):
            raise ValueError("core_model must be a non-empty string")
        return v


class NerConfig(BaseModel):
    """Complete NER configuration with model priorities and linking settings."""

    model_priorities: dict[str, int] = Field(
        ..., description="Priority mapping for NER models (higher = more priority)"
    )
    linker: LinkerConfig = Field(..., description="Entity linking configuration")

    @field_validator("model_priorities")
    @classmethod
    def validate_model_priorities(cls, v: dict[str, int]) -> dict[str, int]:
        """Validate model priorities are positive integers."""
        if not v:
            raise ValueError("model_priorities cannot be empty")

        valid_models = {"bionlp", "craft", "bc5cdr"}

        for model, priority in v.items():
            if model not in valid_models:
                logger.warning(f"Unknown model: {model}")
            if not isinstance(priority, int) or priority < 1:
                raise ValueError(
                    f"Priority for {model} must be a positive integer, got {priority}"
                )

        return v


def _load_yaml(config_path: Path) -> dict[str, Any]:
    """
    Load YAML configuration file with proper error handling.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed YAML configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must contain a YAML dictionary, got {type(config)}"
        )

    return config


def get_default_priorities() -> dict[str, int]:
    """Get hardcoded default priorities as fallback."""
    return {
        "bionlp": 3,  # Most comprehensive
        "craft": 2,  # Good ontology linking
        "bc5cdr": 1,  # Chemical/disease focused
    }


def get_default_linker_config() -> LinkerConfig:
    """Get default linker configuration for development environment."""
    return LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )


def get_default_config() -> NerConfig:
    """Get complete default configuration."""
    return NerConfig(
        model_priorities=get_default_priorities(), linker=get_default_linker_config()
    )


def load_ner_config(config_path: Optional[str] = None) -> NerConfig:
    """
    Load complete NER configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses default location.

    Returns:
        Validated NerConfig object with model priorities and linker configuration

    Raises:
        ValueError: If config file format is invalid
    """
    config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    try:
        config = _load_yaml(config_file)
    except FileNotFoundError:
        logger.warning(
            f"Config file not found: {config_file}. Using hardcoded defaults."
        )
        return get_default_config()
    except (yaml.YAMLError, ValueError) as e:
        logger.error(f"Invalid config file format: {e}. Using defaults.")
        return get_default_config()

    # Load model priorities
    model_priorities = config.get("model_priorities", get_default_priorities())

    # Load linker configuration
    linker_dict = config.get("linker", {})
    try:
        linker_config = (
            LinkerConfig(**linker_dict) if linker_dict else get_default_linker_config()
        )
    except Exception as e:
        logger.error(f"Invalid linker configuration: {e}. Using defaults.")
        linker_config = get_default_linker_config()

    logger.info(f"Loaded config from {config_file}")

    try:
        return NerConfig(model_priorities=model_priorities, linker=linker_config)
    except Exception as e:
        logger.error(f"Config validation failed: {e}. Using defaults.")
        return get_default_config()
