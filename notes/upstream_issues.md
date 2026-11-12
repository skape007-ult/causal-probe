# Upstream issues (not filed; for the owner to verify and file)

Each entry: what, where, minimal reproduction, status of my own verification.

## U1. Corr2Cause and CLadder dataset cards are empty (HF datasets)
- `https://huggingface.co/datasets/causal-nlp/corr2cause/raw/main/README.md` ends with `# Dataset card for corr2cause` / `TODO`. CLadder's card is YAML only. Neither states a licence (`license` tag absent in `https://huggingface.co/api/datasets/causal-nlp/corr2cause`). The GitHub code repos (causalNLP/corr2cause, causalNLP/cladder) are MIT; whether that covers the data is not stated.
- Repro: `curl -sL https://huggingface.co/datasets/causal-nlp/corr2cause/raw/main/README.md | tail -3`

## U2. Corr2Cause dev shares premises with train
- 2 of the 22 distinct dev premises (42 of 1,076 dev items) also occur as premises in train. Test has no premise overlap.
- Repro:
```python
import csv; csv.field_size_limit(10**9)
prem = lambda r: r["input"].split("\nHypothesis: ")[0]
tr = {prem(r) for r in csv.DictReader(open("train.csv"))}
dev = list(csv.DictReader(open("dev.csv")))
print(sum(prem(r) in tr for r in dev), len({prem(r) for r in dev} & tr))   # 42 2
```

## U3. Corr2Cause: `child` template has no positive examples in any split
- `template == "child"`: 34,291 train / 177 dev / 194 test items, all label 0; `parent` is 35% positive. By symmetry one would expect "B directly causes A"-type hypotheses to be valid about as often. May be intentional (e.g. deduplication of symmetric pairs) but is undocumented, and makes the template a strong label shortcut.
- Repro: `collections.Counter((r["template"], r["label"]) for r in csv.DictReader(open("train.csv")))`

## U4. Corr2Cause perturbation files do not align with test.csv
- `perturbation_by_refactorization_test.csv` and `perturbation_by_paraphrasing_test.csv` have 2,246 rows vs 1,162 in `test.csv`; the paraphrasing file's `label` column is 3-class (0/1/2) while the default config is binary; the refactorization file has no `template`/`num_variables` columns. So a paired original-vs-perturbed comparison on the released test set is not possible from these files. (We therefore apply our own A,B,C→Z,Y,X renaming to `test.csv`.)
- Repro: `wc -l test.csv perturbation_by_*_test.csv`; `Counter(r["label"] ...)`.

## U5. TransformerLens 4: `HookedTransformer` import fails with an ImportError that points at a doc path
- `from transformer_lens import HookedTransformer` raises `ImportError: 'HookedTransformer' was removed in TransformerLens 4.0 ... See docs/source/content/migrating_to_v4.md` — a repo-relative path that does not exist in the installed wheel, with no URL. Minor doc gap. (transformer-lens 4.4.0.)
