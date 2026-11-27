from dataclasses import dataclass, field
from enum import Enum, auto


class SeqStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class BatchPhase(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclass
class Sequence:
    """One generation request: prompt + sampled output, slot and sampling params."""
    seq_id: int
    prompt_ids: list[int]
    output_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    max_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    slot: int | None = None
    finish_reason: str | None = None

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def last_token(self) -> int:
        return self.output_ids[-1] if self.output_ids else self.prompt_ids[-1]


@dataclass
class Batch:
    seqs: list[Sequence]
    phase: BatchPhase
