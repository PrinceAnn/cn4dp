"""phenotype_encoder

Reusable phenotype feature encoder components.

This package is intentionally named generically (not "student") because it can
be used either for distillation alignment or for direct downstream prediction.
"""

from .model import MLPConfig, PhenotypeMLP

__all__ = ["MLPConfig", "PhenotypeMLP"]
