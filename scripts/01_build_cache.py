"""Build the activation cache for one model (resumable; safe to re-run).

  python scripts/01_build_cache.py --model qwen0.5b            # trained model, all tags + generation
  python scripts/01_build_cache.py --model qwen0.5b --random   # random-init control (probe sets only)
"""
import argparse
import json
import time

from causal_probe import cache, data, models

TAGS = ["c2c_test", "c2c_probe", "cladder", "c2c_test_renamed",
        "c2c_test_sub_fewshot", "cladder_sub_fewshot"]
RANDOM_TAGS = ["c2c_probe", "cladder"]
GEN_TAGS = ["c2c_test_sub", "cladder_sub", "c2c_test_sub_fewshot", "cladder_sub_fewshot"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--tags", default=None)
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N items per tag (smoke tests only)")
    ap.add_argument("--gen-limit", type=int, default=None, help="generate answers for the first N items only (smoke tests)")
    ap.add_argument("--max-minutes", type=float, default=85.0)
    a = ap.parse_args()
    t0 = time.time()
    tok = models.load_tokenizer(a.model)
    model = models.load_model(a.model, random_init=a.random)
    tags = a.tags.split(",") if a.tags else (RANDOM_TAGS if a.random else TAGS)

    def log(msg):
        print(msg, flush=True)
        if (time.time() - t0) / 60 > a.max_minutes:
            raise SystemExit("time budget reached; re-run to resume")

    for tag in tags:
        cache.build_cache(a.model, tag, random_init=a.random, model=model, tok=tok, log=log, limit=a.limit)
    if not a.random and not a.no_gen:
        for tag in GEN_TAGS:
            out = cache.cache_dir(a.model, tag) / "gen.json"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            fs = tag.endswith("_fewshot")
            items = data.get_dataset(tag.removesuffix("_fewshot"))[: a.gen_limit or a.limit]
            shots = cache.fewshot_examples(items[0]["dataset"]) if fs else None
            kw = {}
            if not models.has_cheap_full_vocab(model):
                # Gemma (embedding on disk): each decoding step pages the 1.2 GB unembedding
                # through memory (~6 s). Decode only the first token, on every 5th item.
                items, kw = items[::5], {"max_new_tokens": 1}
            texts = cache.generate_answers(model, tok, items, shots,
                                           budget=cache.token_budget(a.model, next(model.parameters()).device), **kw)
            out.write_text(json.dumps({"uids": [i["uid"] for i in items], "text": texts}))
            log(f"[gen] {a.model}/{tag} done")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
