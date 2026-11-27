from collections import deque

from core.cache_manager import CacheManager
from core.sequence import Batch, BatchPhase, Sequence, SeqStatus


class Scheduler:
    def __init__(self, cache_manager: CacheManager, eos_id: int = 2, max_context: int = None):
        self.cm = cache_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.eos_id = eos_id
        self.max_context = max_context

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> Batch:
        admitted = []
        while self.waiting and self.cm.can_allocate(self.waiting[0]):
            seq = self.waiting.popleft()
            self.cm.allocate(seq)
            seq.status = SeqStatus.RUNNING
            admitted.append(seq)

        if admitted:
            self.running.extend(admitted)
            return Batch(seqs=admitted, phase=BatchPhase.PREFILL)

        return Batch(seqs=list(self.running), phase=BatchPhase.DECODE)

    def update(self, batch: Batch, next_tokens: list[int]):
        for seq, tok in zip(batch.seqs, next_tokens):
            seq.output_ids.append(int(tok))
            if tok == self.eos_id:
                seq.status = SeqStatus.FINISHED
                seq.finish_reason = "stop"
            elif len(seq.output_ids) >= seq.max_tokens:
                seq.status = SeqStatus.FINISHED
                seq.finish_reason = "length"
            elif self.max_context is not None and seq.num_tokens >= self.max_context:
                seq.status = SeqStatus.FINISHED
                seq.finish_reason = "length"
            if seq.status is SeqStatus.FINISHED:
                self.cm.free(seq)
        self.running = [s for s in self.running if s.status is not SeqStatus.FINISHED]
