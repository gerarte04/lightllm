#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from core.model import Transformer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def load_our_model(config, device):
    """Instantiate core.model.Transformer and load HF weights with strict=True."""
    model = Transformer(config)

    weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
    state_dict = load_file(weights_path)

    # SmolLM2 ties input/output embeddings, so the checkpoint may omit
    # lm_head.weight. Our model ties them too; supply the key so that
    # strict=True loading sees exactly the same set of parameters.
    if config.tie_word_embeddings and "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    incompatible = model.load_state_dict(state_dict, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ours] loaded {len(state_dict)} tensors ({n_params / 1e6:.1f}M params), "
          f"strict=True: no missing/unexpected keys")

    return model.to(device=device, dtype=torch.float32).eval()


def load_hf_model(device):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hf]   {type(model).__name__} ({n_params / 1e6:.1f}M params), "
          f"attn_implementation=eager")
    return model.to(device).eval()


@torch.no_grad()
def compare_logits(ours, hf, tokenizer):
    assert ours.shape == hf.shape, f"shape mismatch: {ours.shape} vs {hf.shape}"
    diff = (ours - hf).abs()

    print("\n=== Logits comparison (float32) ===")
    print(f"logits shape:            {tuple(ours.shape)}  (batch, seq_len, vocab)")
    print(f"logits range [hf]:       [{hf.min().item():.3f}, {hf.max().item():.3f}], "
          f"std={hf.std().item():.3f}")
    print(f"max abs diff:            {diff.max().item():.3e}")
    print(f"mean abs diff:           {diff.mean().item():.3e}")
    print(f"RMSE:                    {diff.pow(2).mean().sqrt().item():.3e}")
    print(f"max diff / max |logit|:  {(diff.max() / hf.abs().max()).item():.3e}")
    print(f"cosine similarity:       {F.cosine_similarity(ours.flatten().cpu().double(), hf.flatten().cpu().double(), dim=0).item():.8f}")
    print(f"allclose(atol=1e-4):     {torch.allclose(ours, hf, atol=1e-4, rtol=1e-4)}")
    print(f"allclose(atol=1e-3):     {torch.allclose(ours, hf, atol=1e-3, rtol=1e-3)}")

    # Do both models pick the same tokens?
    argmax_match = (ours.argmax(-1) == hf.argmax(-1)).float().mean().item()
    print(f"argmax agreement:        {argmax_match * 100:.2f}% of all positions")

    k = 10
    ours_top = set(ours[0, -1].topk(k).indices.tolist())
    hf_top = set(hf[0, -1].topk(k).indices.tolist())
    print(f"top-{k} overlap (last pos): {len(ours_top & hf_top)}/{k}")

    # Top-5 candidates for the next token, side by side.
    print("\n=== Top-5 next-token candidates at the last position ===")
    print(f"{'token':<15} {'ours logit':>12} {'hf logit':>12} {'diff':>12}")
    for idx in hf[0, -1].topk(5).indices.tolist():
        token = repr(tokenizer.decode([idx]))
        o, h = ours[0, -1, idx].item(), hf[0, -1, idx].item()
        print(f"{token:<15} {o:>12.5f} {h:>12.5f} {o - h:>12.2e}")

    ours_next = tokenizer.decode([ours[0, -1].argmax().item()])
    hf_next = tokenizer.decode([hf[0, -1].argmax().item()])
    print(f"\ngreedy next token: ours={ours_next!r}, hf={hf_next!r} "
          f"-> {'MATCH' if ours_next == hf_next else 'MISMATCH'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is",
                        help="input text for the forward pass")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"model: {MODEL_ID}\ndevice: {device}, dtype: float32\n")
    config = AutoConfig.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    rope_theta = getattr(config, "rope_theta", None) or config.rope_parameters["rope_theta"]
    print(f"config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"intermediate={config.intermediate_size}, vocab={config.vocab_size}, "
          f"rope_theta={rope_theta}, rms_norm_eps={config.rms_norm_eps}, "
          f"tie_word_embeddings={config.tie_word_embeddings}")

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"\nprompt: {args.prompt!r}")
    print(f"input_ids ({input_ids.shape[1]} tokens): {input_ids[0].tolist()}")
    print(f"tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")

    our_model = load_our_model(config, device)
    hf_model = load_hf_model(device)

    with torch.no_grad():
        ours_logits = our_model(input_ids)
        hf_logits = hf_model(input_ids).logits

    compare_logits(ours_logits, hf_logits, tokenizer)


if __name__ == "__main__":
    main()
