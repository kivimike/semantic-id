from typing import List, Optional
from semantic_id.uniqueness.stores import CollisionStore, InMemoryCollisionStore

class UniqueIdResolver:
    """
    Resolves collisions by appending suffixes to semantic IDs.
    """
    
    def __init__(self, store: Optional[CollisionStore] = None):
        if store is None:
            self.store = InMemoryCollisionStore()
        else:
            self.store = store
            
    def assign(self, semantic_ids: List[str]) -> List[str]:
        """
        Assign unique IDs for a list of semantic IDs.
        
        Args:
            semantic_ids: List of raw semantic ID strings.
            
        Returns:
            List of unique IDs (potentially with suffixes).
        """
        unique_ids = []
        for sid in semantic_ids:
            # Get the count/index for this SID
            # 0 -> 1st time -> "sid"
            # 1 -> 2nd time -> "sid-1"
            # ...
            idx = self.store.next_suffix(sid)
            
            if idx == 0:
                unique_ids.append(sid)
            else:
                unique_ids.append(f"{sid}-{idx}")
                
        return unique_ids
