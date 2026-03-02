"""Tests for KG filter configuration."""

import pytest

from biomed_kg_agent.kg.config import FilterConfig


def test_filter_config_defaults() -> None:
    """Test FilterConfig with baseline defaults."""
    config = FilterConfig()
    assert config.docs_count_min == 2
    assert config.sent_count_min == 2
    assert config.stopwords_enabled is True
    assert config.allowed_entity_type_pairs is None
    assert config.stopwords is None


def test_filter_config_validation() -> None:
    """Test Pydantic validation on FilterConfig."""
    # Valid config
    config = FilterConfig(docs_count_min=3, sent_count_min=5)
    assert config.docs_count_min == 3
    assert config.sent_count_min == 5

    # Invalid: negative values
    with pytest.raises(ValueError):
        FilterConfig(docs_count_min=-1)

    with pytest.raises(ValueError):
        FilterConfig(sent_count_min=0)


def test_filter_config_with_entity_type_pairs() -> None:
    """Test FilterConfig with allowed entity type pairs."""
    config = FilterConfig(
        docs_count_min=3,
        allowed_entity_type_pairs=[("Gene", "Disease"), ("Drug", "Disease")],
    )
    assert config.allowed_entity_type_pairs == [
        ("Gene", "Disease"),
        ("Drug", "Disease"),
    ]


def test_filter_config_with_custom_stopwords() -> None:
    """Test FilterConfig with custom stopword set."""
    custom_stopwords = {"test", "example", "demo"}
    config = FilterConfig(stopwords=custom_stopwords)
    assert config.stopwords == custom_stopwords
    assert config.stopwords_enabled is True  # Default


def test_filter_config_stopwords_disabled() -> None:
    """Test FilterConfig with stopwords disabled."""
    config = FilterConfig(stopwords_enabled=False)
    assert config.stopwords_enabled is False
    assert config.stopwords is None
