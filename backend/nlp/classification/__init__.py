"""Classification package.

# Diagram omitted - package export surface with no processing logic.
"""

from backend.nlp.classification.arbitration import classify_cluster, classify_clusters
from backend.nlp.classification.types import (
    ClassEvidence,
    ClassificationDecision,
    EntityhoodDecision,
)

__all__ = [
    "ClassEvidence",
    "ClassificationDecision",
    "EntityhoodDecision",
    "classify_cluster",
    "classify_clusters",
]
