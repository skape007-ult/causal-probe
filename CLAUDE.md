# Standing orders: fine-tuning study (branch `exp/finetune`, worktree `causal-probe-ft`)

Written 16 November 2026 on the owner's instruction. These bind every session that works in this worktree.

## Scope
- v1.0 is closed. Never edit `RESULTS.md`, `report/`, the v1.0 scripts in `scripts/`, `src/causal_probe/`, or anything
  they depend on. Import v1.0 code; do not change it. Everything new lives under `finetune/`; `notes/log.md` and
  `notes/progress.md` are appended to as before.
- Analysis code runs in the v1.0 environment (`../causal-probe/.venv/bin/python`), which is not modified. The
  worktree's own `.venv` holds only the Kaggle CLI.

## Claims
1. **Audit before anything enters `finetune/RESULTS_FT.md`.** Every result is audited by a fresh reviewer who is given
   code and result files, not conclusions. Unaudited results are reported as "in progress" or not at all.
2. **Pre-register and power-check every test** in `finetune/PREREG.md` before the run that could decide it:
   hypothesis, prediction, what each outcome means, exact test and threshold, power for a plausible effect. If power
   is below 0.6, redesign or drop the test. Changes after the fact go under "Amendments" with date and reason.
3. **Before describing any result as beyond its controls:** check multiplicity (how many comparisons could have been
   the one reported), replication (other seeds, the other half of the data), and loading (what else the direction or
   feature carries: prompt length, adjacency, the model's own answer).
4. **`RESULTS_FT.md` wording changes wait for the owner.** Queue the proposed wording with its reason in
   `finetune/notes/results_ft_queue.md`; do not enter it. Numbers that regenerate from result files are not wording.
5. The owner's parts (`finetune/MY_PARTS.md`) are not implemented by Claude: the balancing function, the
   within-string AUROC with length removed, the interpretation section. Reference implementations stay in separate
   `*_reference.py` files.
6. In `RESULTS_FT.md` the decisive comparison for "what did it learn" is each model against the memorisation
   reference of its own training set (copy from similar training premises; PREREG E5b). A difference between
   conditions is never presented as evidence of learned structure without it. The owner's balancing function is
   reported as "independently implemented; passes the same property checks; core runs used the reference sets".
7. **Blinding (owner, 17 Nov):** no registered A-vs-B test and no A-vs-B comparison of any kind on the evaluation
   sets until all 5 seeds of A and of B (A_s0-4, B_s0-4) are ingested in `finetune/runs/`. Until then analysis code
   is tested only on synthetic scores and on the base model. Analysis scripts enforce this in code.
8. Never rush a claim to meet a date. The page reports audited parts and says "in progress" for the rest.

## Machines and quota
- Laptop: 8 GB, CPU only. Cap threads (`OMP_NUM_THREADS=2`), one heavy process at a time, nothing that swaps, no
  model larger than Qwen2.5-0.5B loaded locally.
- Run every long local job and the Kaggle queue runner under `caffeinate -i` (for example
  `caffeinate -i .venv/bin/python finetune/scripts/kaggle_run.py poll`), so the Mac stays awake exactly while work is
  running and can sleep otherwise. At the end of each day tell the owner whether anything will run overnight.
- Kaggle: every GPU run goes through `finetune/scripts/kaggle_run.py` (queue file, at most 2 concurrent, no automatic
  relaunch, halt and report on any failure, dry run first). **Cap: 26 GPU hours per quota week; the binding limit is what Kaggle's own page
  says is left (recorded with `kaggle_run.py quota`); the week resets Saturday 00:00 UTC.** Colab is for overflow runs
  the owner launches by hand, never more than one at a time. Ask the owner
  before spending quota on anything that is not in the queue as planned, and before clearing `finetune/kaggle/HALT`.
- No training run before the pre-registration has passed its independent review.

## Housekeeping
- Commits: no Claude co-author line. Notebooks are pinned to a pushed commit of `exp/finetune`.
- The owner asks for "status" each morning; answer from `notes/progress.md`, updated first. No scheduler.
- Batch questions for the owner; do not block on anything these orders let you decide.
- From 17 Nov: anything meant only for the owner (study notes, questions with model answers, specs of the owner's
  parts, verify-yourself checklists, notes addressed to the owner) is written under `private/` (git-ignored on
  `exp/finetune` and `main`). Files already committed elsewhere stay where they are until the owner triggers the
  cleanup.
