"""Model loading (plain Hugging Face + hooks), answer-token handling, random-init control."""
from __future__ import annotations

import gc

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import CACHE, LORA, MODELS, SEED

YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO", " NO"]


def pick_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_tokenizer(name: str):
    if name in LORA:
        name = LORA[name][0]
    tok = AutoTokenizer.from_pretrained(MODELS.get(name, name))
    tok.padding_side = "left"  # so position -1 is always the final prompt token
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# Models whose embedding matrix is kept on disk instead of in (unified) GPU memory.
# Gemma-2-2B has a 256k x 2304 embedding (1.18 GB in float16) tied to the unembedding; with it
# on the GPU the model (5.3 GB) thrashes an 8 GB Mac at 0.07-0.14 items/s. See decision log D12.
DISK_EMBED = {"gemma2b"}


class DiskEmbedding:
    """Embedding / unembedding matrix as a read-only float16 memory-map on disk.

    embed():   row lookup on CPU (touches only the rows used), scaled, moved to the device.
    unembed(): last-position hidden states -> full-vocab logits, computed on CPU in chunks.
    Numerically this is the same matrix the model would use; only where it lives changes.
    """

    def __init__(self, path, scale: float, device, dtype):
        self.W = np.load(path, mmap_mode="r")
        self.scale, self.device, self.dtype = scale, device, dtype

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        e = torch.from_numpy(np.ascontiguousarray(self.W[ids.cpu().numpy()]))
        return (e.to(self.device, self.dtype) * self.scale)

    def rows(self, ids: list[int]) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.W[ids])).float()

    def unembed(self, h: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
        """Full-vocab logits [B, V] (float32, CPU). Chunks of the matrix are moved to the device
        one at a time. Touches the whole 1.2 GB matrix, so use sparingly (generation only)."""
        hd = h.to(self.device, self.dtype)
        out = []
        for s in range(0, self.W.shape[0], chunk):
            w = torch.from_numpy(np.ascontiguousarray(self.W[s:s + chunk])).to(self.device)
            out.append((hd @ w.T).float().cpu())
            del w
        return torch.cat(out, 1)


def load_model(name: str, device: str | None = None, dtype=None, random_init: bool = False,
               disk_embed: bool | None = None):
    """Load an instruct model. `random_init=True` gives an untrained copy with the same
    architecture and tokenizer (the 'random model' probing control)."""
    if name in LORA:            # base model + trained LoRA adapter, merged into plain weights
        from peft import PeftModel
        base, adapter = LORA[name]
        assert not random_init
        device = device or pick_device()
        model = load_model(base, device="cpu", dtype=torch.float32)
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        model.to(dtype or (torch.float32 if device == "cpu" else torch.float16)).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    hf_id = MODELS.get(name, name)
    device = device or pick_device()
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else DEFAULT_DTYPE.get(name, torch.float16)
    if disk_embed is None:
        disk_embed = name in DISK_EMBED and device != "cpu"
    if random_init and not disk_embed:
        torch.manual_seed(SEED)
        cfg = AutoConfig.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_config(cfg).to(dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=dtype)
        if random_init:
            # Building a 2.6B-parameter model from config allocates float32 (10 GB). Instead
            # re-initialise the loaded half-precision weights in place with the model's own
            # initialiser - the same function from_config() applies.
            torch.manual_seed(SEED)
            model.apply(model._init_weights)
    if disk_embed:
        emb = model.model.embed_tokens
        path = CACHE / f"embed_{name}{'-rand' if random_init else ''}.npy"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb.weight.detach().to(torch.float16).numpy())
        scale = float(getattr(emb, "scalar_embed_scale", 1.0))
        assert model.lm_head.weight is emb.weight or torch.equal(model.lm_head.weight[:8], emb.weight[:8]), "untied"
        tiny = torch.nn.Parameter(torch.zeros(1, emb.weight.shape[1], dtype=dtype), requires_grad=False)
        emb.weight = tiny
        model.lm_head.weight = tiny
        gc.collect()
        model._disk = DiskEmbedding(path, scale, device, dtype)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# Filled in after the MPS-vs-CPU verification (scripts/00_verify_env.py); see decision log.
DEFAULT_DTYPE: dict = {}


def answer_token_ids(tok) -> dict:
    """Single-token ids for each Yes/No surface variant ('Yes', ' yes', ...).
    Variants that are not a single token for this tokenizer are dropped."""
    def ids(words):
        out = {}
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                out[w] = t[0]
        return out
    yes, no = ids(YES_VARIANTS), ids(NO_VARIANTS)
    assert yes and no
    return {"yes": yes, "no": no}


def answer_logit_diff(logits: torch.Tensor, ans: dict) -> torch.Tensor:
    """logsumexp over Yes-variants minus logsumexp over No-variants (last dim = vocab)."""
    y = torch.logsumexp(logits[..., list(ans["yes"].values())].float(), -1)
    n = torch.logsumexp(logits[..., list(ans["no"].values())].float(), -1)
    return y - n


def body_forward(model, input_ids, attention_mask=None, **kw):
    """Run the transformer body (everything except the unembedding)."""
    dev = next(model.parameters()).device
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)
    disk = getattr(model, "_disk", None)
    if disk is not None:
        return model.model(inputs_embeds=disk.embed(input_ids), attention_mask=attention_mask, **kw)
    return model.model(input_ids=input_ids.to(dev), attention_mask=attention_mask, **kw)


def _softcap(model, logits):
    cap = getattr(model.config, "final_logit_softcapping", None)
    return torch.tanh(logits / cap) * cap if cap else logits


def logits_from_hidden(model, h: torch.Tensor) -> torch.Tensor:
    """Post-final-norm hidden state [B, d] -> full-vocab logits [B, V] (float32, on CPU)."""
    disk = getattr(model, "_disk", None)
    logits = disk.unembed(h) if disk is not None else model.lm_head(h).float().cpu()
    return _softcap(model, logits)


def final_logits(model, input_ids, attention_mask=None, **kw) -> torch.Tensor:
    """Next-token logits at the LAST position only, [B, vocab], float32 on CPU.

    Calling model(...).logits would unembed every position (B x T x 256k floats for Gemma),
    which exhausts memory on an 8 GB machine. We run the transformer body, take the final
    hidden state at the last position and unembed just that. Requires left padding.
    """
    h = body_forward(model, input_ids, attention_mask, **kw).last_hidden_state[:, -1]
    return logits_from_hidden(model, h)


def answer_logits_from_hidden(model, h: torch.Tensor, ans: dict) -> torch.Tensor:
    """Post-norm hidden state [B, d] -> logits of the answer tokens only [B, n_yes + n_no]
    (float32, CPU). Needs just a handful of unembedding rows."""
    if not hasattr(model, "_W_ans"):
        ids = list(ans["yes"].values()) + list(ans["no"].values())
        model._W_ans = unembed_rows(model, ids).to(h.device)
    return _softcap(model, h.float() @ model._W_ans.T).cpu()


def has_cheap_full_vocab(model) -> bool:
    return getattr(model, "_disk", None) is None


def unembed_rows(model, ids: list[int]) -> torch.Tensor:
    """Rows of the unembedding matrix for a few token ids (float32, CPU)."""
    disk = getattr(model, "_disk", None)
    return disk.rows(ids) if disk is not None else model.lm_head.weight[ids].detach().float().cpu()


def get_layers(model):
    """The list of transformer blocks (Qwen2 and Gemma2 both expose model.model.layers)."""
    return model.model.layers


def answer_logit_diff_from_resid(model, resid: torch.Tensor, ans: dict) -> torch.Tensor:
    """Pre-norm residual vector(s) [..., d] -> Yes-No answer logit difference, using only the
    answer-token rows of the unembedding (cheap; stays on the model's device; differentiable)."""
    if not hasattr(model, "_W_ans"):
        ids = list(ans["yes"].values()) + list(ans["no"].values())
        model._W_ans = unembed_rows(model, ids).to(resid.device)
    lg = _softcap(model, model.model.norm(resid.to(next(model.parameters()).dtype)).float() @ model._W_ans.T)
    ny = len(ans["yes"])
    return torch.logsumexp(lg[..., :ny], -1) - torch.logsumexp(lg[..., ny:], -1)


def answer_logit_diff_from_normed(model, h: torch.Tensor, ans: dict) -> torch.Tensor:
    """Post-norm hidden state [B, d] -> Yes-No logit difference, on-device and differentiable
    (used for attribution patching)."""
    if not hasattr(model, "_W_ans"):
        ids = list(ans["yes"].values()) + list(ans["no"].values())
        model._W_ans = unembed_rows(model, ids).to(h.device)
    lg = _softcap(model, h.float() @ model._W_ans.T)
    ny = len(ans["yes"])
    return torch.logsumexp(lg[..., :ny], -1) - torch.logsumexp(lg[..., ny:], -1)
