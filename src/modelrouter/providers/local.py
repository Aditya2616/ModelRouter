"""Local backend: small open models through transformers, on CPU.

This is the path that is genuinely exercised end to end on this machine.
"""

from __future__ import annotations

import threading
import time

from .base import Completion, Message, ProviderError


class LocalProvider:
    name = "local"

    def __init__(self, threads: int | None = None):
        self._models: dict[str, tuple] = {}
        self._lock = threading.Lock()
        self._threads = threads

    def _load(self, model: str):
        if model not in self._models:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:  # pragma: no cover
                raise ProviderError(f"transformers/torch not available: {e}") from None
            if self._threads:
                torch.set_num_threads(self._threads)
            tok = AutoTokenizer.from_pretrained(model)
            lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
            lm.eval()
            self._models[model] = (tok, lm)
        return self._models[model]

    def complete(self, model: str, messages: list[Message], max_tokens: int = 256,
                 temperature: float = 0.0, prefix: str = "") -> Completion:
        """`prefix` is forced as the start of the assistant turn (the returned text excludes it). The live
        experiment uses it so a short multiple-choice answer is not cut off mid-sentence."""
        import torch

        with self._lock:
            tok, lm = self._load(model)
            t0 = time.perf_counter()
            if prefix:
                text_in = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + prefix
                ids = tok(text_in, return_tensors="pt")
            else:
                ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                              return_dict=True)
            n_prompt = ids["input_ids"].shape[1]
            gen_kwargs = {"max_new_tokens": max_tokens, "return_dict_in_generate": True,
                          "output_scores": True, "pad_token_id": tok.eos_token_id}
            if temperature and temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)
            else:
                gen_kwargs.update(do_sample=False)
            with torch.inference_mode():
                out = lm.generate(**ids, **gen_kwargs)
            seq = out.sequences[0, n_prompt:]
            lps = []
            for step, tid in enumerate(seq):
                logp = torch.log_softmax(out.scores[step][0].float(), dim=-1)[tid].item()
                lps.append(logp)
            eos = {tok.eos_token_id, getattr(tok, "pad_token_id", None)}
            gen_ids = [int(t) for t in seq]
            hit_end = bool(gen_ids) and gen_ids[-1] in eos
            if hit_end:
                lps = lps[:-1]
            text = tok.decode(seq, skip_special_tokens=True).strip()
            return Completion(
                text=text, prompt_tokens=n_prompt, completion_tokens=len(gen_ids), model=model,
                finish_reason="stop" if hit_end else "length",
                mean_logprob=(sum(lps) / len(lps)) if lps else None,
                latency_s=time.perf_counter() - t0,
            )
