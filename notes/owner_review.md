# Owner review: the five-minute quick check (23 November 2026)

**PR:** https://github.com/skape007-ult/causal-probe/pull/1 (branch `claude/loving-davinci-sgvou5`, open, not merged).

## What it adds

- `repro/quickcheck.ipynb`: a CPU-only notebook (no model weights) that downloads the official Corr2Cause `train.csv` and `test.csv` from Hugging Face at the pinned revision, md5-checked. It then recomputes these numbers with the repository's own code and prints each beside the committed value, with PASS/FAIL:
  - claim 1's hypothesis-only AUROC;
  - the debiased-pool shortcut checks;
  - the prompt-length leak within a hypothesis type;
  - claim 2's model-free lookup (0.765).
  It also checks that `src/`, `scripts/` and `results/` are identical to tag v1.0.1. It is committed with the outputs of a clean run, so GitHub shows the results.
- `README.md`: a "Check the headline in 5 minutes" section (Colab badge, one paragraph) above the Status paragraph, and a link to it in the navigation line. The build disclosure (laptop plus free Kaggle and Colab; eighteen reviews by fresh reviewer agents) is unchanged.
- Nothing in `RESULTS.md`, `report/`, `scripts/`, `src/` or `results/` changed.

## What was checked

| Check | Recomputed | Committed |
|---|---|---|
| Hypothesis-only, official test split, AUROC | 0.895 [0.863, 0.919] | 0.895 [0.863, 0.919] |
| Same classifier, accuracy | 0.855 | 0.855 |
| Debiased pool: hypothesis-only string prior, AUROC and accuracy | 0.500, 0.500 | 0.500, 0.500 |
| Debiased pool: hypothesis-only TF-IDF, AUROC and accuracy | 0.500, 0.500 | 0.500, 0.500 |
| Debiased pool: premise-only TF-IDF, AUROC and accuracy | 0.500, 0.500 | 0.500, 0.500 |
| Debiased pool: whole prompt, additive, AUROC and accuracy | 0.512 [0.491, 0.535], 0.495 | 0.512, 0.495 |
| Prompt length per hypothesis type, report half: AUROC | 0.571 [0.539, 0.603] | 0.571 [0.539, 0.603] |
| Prompt length per hypothesis type, report half: accuracy | 0.5625 | 0.5625 |
| Claim 2 lookup: AUROC and accuracy | 0.834, 0.765 | 0.834, 0.765 |

The notebook also checks three things that have no committed number:
- the pool's construction;
- that every shortcut interval contains 0.5 and the length interval lies above it;
- that the length and lookup rows are identical in all six models' result files.

Every recomputed value equals the committed one exactly (|diff| = 0). All 28 checks pass.

## Runs and run time

All runs below were alone on the machine, limited to two CPUs, from a fresh clone with an empty Hugging Face cache.

| Run | Environment | Result | Time |
|---|---|---|---|
| Clean run, at `84080ee` | Python 3.12 venv with only the `requirements.txt` pins | 28 of 28 | 63.8 s (notebook: 61 s) |
| Earlier clean run, at `0f1679b` | same | 28 of 28 | 68.1 s (notebook: 65 s) |
| Independent agent, at `42eefa9` (shown none of my numbers) | its own venv | 28 of 28, exact | 63.5 s (notebook: 61 s) |
| Colab simulation | older stack (scikit-learn 1.6.1, numpy 2.0.2), stub `google.colab`, empty folder, so the notebook cloned from GitHub | installed scikit-learn 1.9.1, then 28 of 28 | 82 s |

The independent agent also hand-checked the committed values in the JSON files and confirmed that `git diff v1.0.1 HEAD -- src scripts results` is empty.

## What did not match, or needs your attention

1. **The prompt-length accuracy depends on the scikit-learn version.** Nothing in the clean run mismatched; this showed up only under other versions.
   - Under scikit-learn 1.6.1, 1.7.2 and 1.8.0 the `template_x_prompt_length` accuracy is 0.5561, not 0.5625, and its AUROC interval is [0.540, 0.602], not [0.539, 0.603]. The AUROC estimate is 0.571 either way, and the conclusion (CI above 0.5) holds.
   - 1.9.0 and 1.9.1 reproduce the committed values.
   - Under 1.6.1 every other check passes.
   - So the notebook installs scikit-learn 1.9.1 in Colab, and warns when run locally on another version. Only `requirements.txt` pins it; `pyproject.toml` does not.
2. **Colab itself was not run** from here. Before relying on the badge, run it once yourself; the branch link is in the PR.
3. **The badge points at `main`,** so it 404s until the PR is merged.
4. **README.md is generated** from `report/README_template.md`, which I did not touch because `report/` is closed. Rerunning `scripts/13_build_docs.py` would drop the new section unless you copy it into the template.
5. **0.5625 is a rounding tie.** It prints as 0.562 (Python rounds the binary value half to even). Conventional rounding would say 0.563. Both sides of the check use the same rounding, so the pass is sound.
6. **Network.** The session's environment had to allow huggingface.co and its CDN hosts (`us.aws.cdn.hf.co` and others). I created a personal cloud environment named `causal-probe` for this; your Default environment is unchanged.
7. **Commit history.** Commit `94c4340` says it applies wording fixes but only stripped the notebook's outputs (an error in my edit script). `84080ee` applies the fixes, and `3f7ff5c` restores the outputs. I left the history as it is rather than rewriting it.
8. **Commit author.** The commits are authored as `Claude <noreply@anthropic.com>`, which is this environment's git identity. There is no co-author line. Change the author before merging if you prefer.
9. **There is no `CLAUDE.md`** in the repository or its history, so I worked from `README.md` and `RESULTS.md` only.
