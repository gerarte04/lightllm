from core.sequence import Sequence

class CacheManager:
    """Tracks free KV-cache slot indices (one slot per concurrent sequence)."""
    def __init__(self, max_seqs):
        self.max_seqs = max_seqs
        self.free_slots = list(range(max_seqs))

    def can_allocate(self, seq: Sequence) -> bool:
        return bool(self.free_slots)

    def allocate(self, seq: Sequence):
        seq.slot = self.free_slots.pop()

    def free(self, seq: Sequence):
        self.free_slots.append(seq.slot)
        seq.slot = None
