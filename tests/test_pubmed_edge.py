from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from biomed_kg_agent.data_sources.ncbi import ncbi_utils, pubmed


@pytest.mark.unit
def test_save_to_sqlite_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    pubmed.save_to_sqlite([], str(db_path))
    assert not db_path.exists()


@pytest.mark.unit
def test_save_to_json_empty(tmp_path: Path) -> None:
    json_path = tmp_path / "test.json"
    pubmed.save_to_json([], str(json_path))
    assert not json_path.exists()


@pytest.mark.unit
def test_fetch_abstracts_handles_exception_get(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:  # Changed type hint
    """Test that fetch_abstracts handles exceptions for GET requests."""

    mocker.patch("time.sleep", return_value=None)  # Speed up retries by disabling sleep

    mock_get_request = mocker.patch.object(
        ncbi_utils.requests,
        "get",
        side_effect=ncbi_utils.requests.exceptions.RequestException(
            "network error via GET"
        ),
    )
    # mock_post_request will be a simple spy that doesn't raise an error
    mock_post_request = mocker.patch.object(ncbi_utils.requests, "post")

    # Pass delay=0.001 to speed up the inter-batch delay as well
    result = pubmed.fetch_abstracts(["123456"], delay=0.001)
    assert result == []

    assert mock_get_request.call_count > 0  # GET should be called (and retried)
    assert mock_post_request.call_count == 0  # POST should NOT be called


@pytest.mark.unit
def test_fetch_abstracts_handles_exception_post(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:  # Changed type hint
    """Test that fetch_abstracts handles exceptions for POST requests."""

    mocker.patch("time.sleep", return_value=None)  # Speed up retries by disabling sleep

    # mock_get_request will be a simple spy
    mock_get_request = mocker.patch.object(ncbi_utils.requests, "get")
    mock_post_request = mocker.patch.object(
        ncbi_utils.requests,
        "post",
        side_effect=ncbi_utils.requests.exceptions.RequestException(
            "network error via POST"
        ),
    )

    # Create a list of 200 PMIDs to trigger the POST request
    # (assuming default use_post_threshold=200)
    pmids_for_post = [str(i) for i in range(200)]
    # delay is already 0.001 from previous edit
    result = pubmed.fetch_abstracts(pmids_for_post, delay=0.001)
    assert result == []

    assert mock_post_request.call_count > 0  # POST should be called (and retried)
    assert mock_get_request.call_count == 0  # GET should NOT be called


@pytest.mark.unit
def test_get_pubmed_pmids_handles_exception(
    mocker: MockerFixture,
) -> None:  # Changed monkeypatch to mocker
    """Test that get_pubmed_pmids handles exceptions gracefully."""

    # Mock requests.get to raise a RequestException
    mocked_get = mocker.patch.object(
        ncbi_utils.requests,
        "get",
        side_effect=ncbi_utils.requests.exceptions.RequestException("API error"),
    )

    result = pubmed.get_pubmed_pmids("test", 1)  # Removed delay argument
    assert result == []
    mocked_get.assert_called()  # Verify that the mocked get was indeed called
