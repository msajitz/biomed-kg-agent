"""Mock utilities for NCBI/PubMed API testing.

This module provides mock implementations for NCBI E-utilities API responses
used across PubMed, PMC, and other NCBI-related tests.
"""


class MockResp:
    """Mock HTTP response for NCBI E-utilities API calls."""

    def __init__(self, xml_response: str):
        self._xml_response = xml_response

    def raise_for_status(self) -> None:
        """Mock requests.Response.raise_for_status() - never raises."""
        pass

    @property
    def text(self) -> str:
        """Mock requests.Response.text property."""
        return self._xml_response
