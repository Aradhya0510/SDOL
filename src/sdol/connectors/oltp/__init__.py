"""OLTP connectors — transactional / point-lookup paradigm."""

from sdol.connectors.oltp.base import BaseOLTPConnector
from sdol.connectors.oltp.generic import GenericOLTPConnector

__all__ = [
    "BaseOLTPConnector",
    "GenericOLTPConnector",
]
