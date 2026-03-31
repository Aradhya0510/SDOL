"""OLAP connectors — analytical / aggregation paradigm."""

from sdol.connectors.olap.base import BaseOLAPConnector
from sdol.connectors.olap.generic import GenericOLAPConnector

__all__ = [
    "BaseOLAPConnector",
    "GenericOLAPConnector",
]
