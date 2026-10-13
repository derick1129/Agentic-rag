"""Public semantic-cache contracts."""

from adaptive.cache.references import ReferenceChecker, has_complete_references
from adaptive.interfaces import CacheEntry, CacheHit, CacheLookup, CacheStore

__all__ = [
    "CacheEntry",
    "CacheHit",
    "CacheLookup",
    "CacheStore",
    "ReferenceChecker",
    "has_complete_references",
]
