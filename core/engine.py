import torch

from core.cache_manager import CacheManager
from core.model import Transformer
from core.runner import ModelRunner
from core.scheduler import Scheduler
from core.sequence import Batch, Sequence, SeqStatus


class Engine:
    """Continuous-batching generation loop: schedule -> run -> update."""

    def __init__(self, scheduler: Scheduler, runner: ModelRunner):
        self.scheduler = scheduler
        self.runner = runner
        self._next_id = 0

    @property
    def max_context(self) -> int:
        return self.scheduler.max_context

    def add_request(
            self,
            prompt_ids,
            max_tokens: int = 256,
            temperature: float = 0.0,
            top_k: int = 0,
            top_p: float = 1.0,
            repetition_penalty: float = 1.0,
            no_repeat_ngram_size: int = 0
            ) -> Sequence:
        seq = Sequence(
            seq_id=self._next_id,
            prompt_ids=list(prompt_ids),
            max_tokens=max_tokens,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        )
        self._next_id += 1
        if not seq.prompt_ids or len(seq.prompt_ids) >= self.max_context:
            seq.status = SeqStatus.FINISHED
            seq.finish_reason = "length"
        else:
            self.scheduler.add(seq)
        return seq

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> Batch | None:
        batch = self.scheduler.schedule()
        if not batch.seqs:
            return None
        next_tokens = self.runner.run(batch)
        self.scheduler.update(batch, next_tokens)
        return batch

    def generate(self, prompts, **kwargs) -> list[Sequence]:
        seqs = [self.add_request(p, **kwargs) for p in prompts]
        while self.has_unfinished():
            self.step()
        return seqs


def make_engine(
        model: Transformer,
        max_seqs: int, 
        max_len: int = None,
        eos_id: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
        pad_id: int = None
        ) -> Engine:
    cfg = model.config
    if eos_id is None:
        eos_id = getattr(cfg, "eos_token_id", 2)
    runner = ModelRunner(model, max_seqs, max_len=max_len, device=device, dtype=dtype, pad_id=pad_id)
    scheduler = Scheduler(CacheManager(max_seqs), eos_id=eos_id, max_context=runner.max_len)
    return Engine(scheduler, runner)
