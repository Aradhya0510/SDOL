"""Shared fixtures for Provena tests."""

import pytest

from provena.connectors.executor import MockQueryExecutor
from provena.core.provenance.trust_scorer import TrustScorer


@pytest.fixture
def trust_scorer() -> TrustScorer:
    return TrustScorer()


@pytest.fixture
def mock_executor() -> MockQueryExecutor:
    return MockQueryExecutor(records=[{"id": 1, "value": "test"}])
