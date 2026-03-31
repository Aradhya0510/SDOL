"""Databricks document extensions — vector/semantic search via Databricks Vector Search."""

from sdol.extensions.databricks.document.vector_search import (
    DatabricksVectorSearchConnector,
)
from sdol.extensions.databricks.document.vector_search_query import (
    DatabricksVSQuery,
    build_vs_similarity_query,
)

__all__ = [
    "DatabricksVectorSearchConnector",
    "DatabricksVSQuery",
    "build_vs_similarity_query",
]
