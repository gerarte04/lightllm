import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from tests.helpers import MODEL_ID, load_our_model


@pytest.fixture(scope="session")
def env():
    """Shared (config, tokenizer, model, device) for all comparison tests.

    The model is loaded once per session with strict=True weight loading
    (see tests.helpers.load_our_model).
    """
    device = torch.device("cpu")
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_our_model(config, device)
    return config, tokenizer, model, device
