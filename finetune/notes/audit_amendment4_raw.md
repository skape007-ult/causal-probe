# Independent review of `finetune/notes/amendment4_draft.md` (Amendment 4, E7/E8 deferred)

Reviewer: fresh, independent; 23 Nov 2026. Scope: the text only, meaning whether each statement is true and supported
by the files and whether the text does what it says. The experimental design is out of scope.

Read: `finetune/PREREG.md` (Amendment 3, Sections 0-6), `finetune/e7e8/pilot_prepare.json`,
`finetune/scripts/e7e8_prepare.py`, `finetune/ftstudy/e7e8.py` (function list), `finetune/notes/audit_amendment3_third_raw.md`,
`finetune/kaggle/queue.json`, `finetune/kaggle/run_loop.log`, `finetune/kaggle/halt_history.txt`,
`finetune/scripts/kaggle_run.py` (launch and HALT logic), `finetune/kaggle/push/` and `downloads/`, `git log`.

## 1. Numbers

All values are recomputed from `pilot_prepare.json`.

| Draft | File | Verdict |
|---|---|---|
| 36% (72 of 198) | `decided...pilot_share_items_without_both_donors` = 0.3636; n_pilot_items 198; 0.3636 × 198 = 72 | correct (the third review gives 36.4%) |
| 35.6–38.3% at every candidate N_items | third review §5(iii): N=300 38.3%, 600 37.8%, 1,000 35.6%, all 2,162 35.6% | correct |
| r: A_s1 0.097, A_s3 0.089, B_s2 0.079, B_s3 0.060 | 0.0970, 0.0886, 0.0793, 0.0599. The other six are ≥ 0.10 (lowest B_s4 0.104) | correct and complete |
| A_s4, B_s4 separation along g below the median control's | A_s4 0.0433 vs 0.0609; B_s4 0.0463 vs 0.0552. No other model is below | correct |
| B own-answer S at site 0.974–0.984 | 0.9739 (s3) to 0.9838 (s1) | correct |
| B at final 0.994–0.996 | 0.9943 to 0.9963 | correct |
| "within 0.02 in four of five seeds (B_s0, B_s1, B_s2, B_s4)" | gaps 0.0145, 0.0109, 0.0142, **0.0204 (B_s3)**, 0.0136; matches `upstream_qualifier_applies_in` | correct |
| A site 0.84–0.89, final 0.96–0.99 | site 0.844–0.890, final 0.958–0.985 (gaps 0.088–0.117) | correct (0.958 rounds to 0.96) |
| A_s4/B_s4 median-|cos| trigger not cited | max median |cos| is 0.832 (B_s1) < 0.9 | correctly not cited |

All numbers are right.

## 2. Claims of fact

- **No intervention has run.** This is true. `ft_e7e8_pilot` in `queue.json` has `"state": "queued"` and has no
  `launched_at`, push output or commit. `finetune/kaggle/push/` and `downloads/` have no `ft_e7e8_pilot` directory, and
  `run_loop.log` never mentions it (its last entry is 03:48 UTC, E6). No E7/E8 result file exists other than
  `finetune/results/e7e8_pilot_synthetic.json`, which is synthetic. Steering and patch vectors were **computed** and
  committed (`pilot_inputs.npz`) but were not applied to any model. The draft's wording ("no steering or patching
  vector has been applied") is accurate.
- **"The pilot stays queued ... and is not launched": misleading as a safeguard.** HALT was **cleared** at
  04:41:05 UTC on 23 Nov (`halt_history.txt`), and `finetune/kaggle/HALT` does not exist. `kaggle_run.py launch` and
  `run` push the first entry whose state is `"queued"` (line 200). The only things that block it are the cap and the
  quota. So a routine `kaggle_run.py run` would launch the pilot, which contradicts the amendment. The queue entry's
  note ("waits for the owner to clear HALT") is also stale. The runner has no withdraw command, so the text has to name
  a real mechanism (condition 1).
- **Commits.** Both are correct. 0d44366 (2026-11-22 23:35:57 −0500 = 23 Nov 04:35 UTC) created `pilot_prepare.json`.
  01e6908 (23:54:29 −0500 = 04:54 UTC) re-ran it with the diagnostics for all ten models and the decided labels. Both
  are on 23 Nov in UTC and 22 Nov in local time, and the draft does not say which (condition 5).
- **"Labels and activations only": not quite true.** `e7e8_prepare.py` also uses each model's stored answer logit
  difference (`meta.json` `answer_logit_diff`, checked against `scores.npz`), which gives `own` and the answer loading.
  It also uses the text stack's predictions (`oof_text_sign`, `stack_rep`), token counts and adjacency. These are all
  outputs of unmodified forward passes or of text-only models, and none is an intervention outcome, so the draft's
  point stands. The list is still incomplete (condition 3).
- **"The registered-run analysis code is not written."** This is true. `ftstudy/e7e8.py` has no registered test, no
  Holm step, no outlier rule and no half swap (third review M10, still open).
- **"The independent review of 23 Nov"** is `audit_amendment3_third_raw.md`. The draft should name the file (condition 6).

## 3. Applying Amendment 3's rules

- **"Uninformative" (Section 2, Degenerate g).** The rule applies "if, in any of its models: r < 0.10; or the
  report-half separation ... is not above that of the median control; or the median |cos| > 0.9". The r trigger alone
  fires for A (A_s1, A_s3) and for B (B_s2, B_s3), so the conclusion holds. One caveat: `separation_g_report` is
  computed over **all** report-half items (`H[ri]`), pilot premises included. The registered items exclude the pilot
  premises, and the JSON itself notes that the separation trigger "needs report-half numbers per run". The A_s4/B_s4
  sentence therefore rests on a choice of item set that Amendment 3 leaves open. The conclusion does not depend on it
  (condition 4).
  The label applies only to a "not shown". The draft says this correctly.
- **Upstream qualifier.** The rule is "site value ≥ final-position value − 0.02". It does not say whether this is
  judged per model, on the seed mean, or in any model; the third review left this as open item 4(f). The draft counts
  seeds ("four of five"), which is not a rule. B carries the qualifier under the seed mean (0.9805 vs 0.9952 − 0.02 =
  0.9752) and under "any model". It does not carry it under "every model", because B_s3 misses by 0.0004. The text
  should give this instead of an implied per-seed count (condition 2). For A, no reading triggers it. The draft is
  correct there.
- **30% rule.** The rule reads "more than 30% of items skipped in **any model**". The donor rule uses only item
  metadata (string, gold label, adjacency, premise, token count), and donors and items are shared across all ten
  models, so the skip share is identical in every model. The rule applies to the **registered** items, not to the pilot
  items. The draft leads with the pilot figure (36%), and the JSON flag is explicitly conditional
  (`E8_descriptive_only_if_this_holds_in_the_registered_items`). The figure that decides the rule is the third
  review's 35.6–38.3% for every candidate N_items. The draft should lead with that figure and give the
  model-independence argument (condition 2). The conclusion itself, E8(A) and E8(B) descriptive with p = 1, is
  correct.
- **"At most a claimed positive E7": overclaims.** Section 6 also allows "claimed < 0 → reported as found". The draft
  also leaves out that E7 can still be dropped for power. Replace the sentence (condition 2).

## 4. Misleading, overclaimed or missing

- **Misquoted qualifier.** "upstream in position, not in content" is not the registered wording. It comes from
  reviewer remark 91 of the first Amendment 3 review. The registered text is "at a site where the model's answer is
  already as decodable as at the answer position" (condition 2).
- **Date and state line.** The other amendments give a UTC time and the state of the runs. This one gives the date
  only (condition 5).
- **What is deferred.** The draft does not say that the third review's open items are not registered: no Amendment 3a
  exists in PREREG.md. These are M5 (merge precision), M8 (donor order and the pilot donor pool), M9 (whether a
  descriptive E8 counts towards N_items) and 4(f) (aggregation for the qualifier). The draft should say they pass to
  the redesign and are not settled here (condition 6).
- **What stays registered.** Item 5 covers Section 3 and E1–E6. It does not say whether Amendment 3 itself stays in
  force as text, or whether the withdrawal of the L4 row (Amendment 3, Section 0) still holds. A reader needs both
  (condition 6).
- **Page wording.** Item 4 is clear and correctly neutral.

## 5. Wording that could read as evidence about use

Item 4's last sentence correctly rules out reading the diagnostics as evidence about use. One phrase is borderline:
the heading "**B's site already reads out the model's answer**". "Reads out" suggests the model uses the site to
produce its answer, which is a claim about use. The diagnostic shows only that a linear probe decodes the answer sign
equally well at the site and at the final position (condition 2). "Uninformative" and "keeps less than 10% of its
norm" are fine.

## Verdict: PASS WITH CONDITIONS

1. **Make "not launched" true by mechanism.** Before or with the commit, mark `ft_e7e8_pilot` in `queue.json` with a
   state other than `"queued"`, such as `"withdrawn"` with `withdrawn_at` and the reason "Amendment 4", and update the
   stale note. Alternatively, write `finetune/kaggle/HALT` with the reason. The text must name the mechanism, for
   example: "Its queue entry is marked withdrawn (state `withdrawn`), so the runner cannot launch it; HALT was cleared
   at 04:41 UTC for E6 and does not block it."
2. **Rule-application wording in item 2:**
   - **(a)** Lead E8 with the registered-items figure: "Under Section 3's donor rule, 35.6–38.3% of the registered
     items lack an opposite-label or a same-label donor at every candidate N_items (third review); in the pilot items,
     72 of 198 (36%). The donor rule uses only item metadata and the items and donors are shared, so the share is the
     same in every model, and the 30% rule ('in any model') applies to both conditions."
   - **(b)** Replace "four of five seeds" with "Amendment 3 does not say whether the qualifier is judged per model or
     on the seed mean; it applies on the seed mean (0.981 vs 0.995) and in four of five models (B_s3 misses by
     0.0004), so B carries it under any reading except 'every model'."
   - **(c)** Quote the registered qualifier: "at a site where the model's answer is already as decodable as at the
     answer position".
   - **(d)** Rename the heading to "At B's site, the model's own answer is already about as decodable as at the
     answer position".
   - **(e)** Replace "could give at most a claimed positive E7" with "the only tests that could still be claimed are
     E7(A) and E7(B), in either direction (B with the qualifier), and either could still be dropped for power".
3. **Diagnostic inputs.** Replace "labels and activations only" with "gold labels, item metadata, the models' stored
   (unsteered) answer scores, the text stack's predictions, and E6's stored activations".
4. **Separation trigger.** Add: "(computed over all report-half items, pilot premises included; the r trigger alone
   decides both conditions)".
5. **Heading.** Add a UTC time and a state line, for example "23 November 2026, hh:mm UTC, after
   `pilot_prepare.json` was seen (commits 0d44366, 04:35 UTC, and 01e6908, 04:54 UTC) and before any intervention",
   and date the diagnostics in UTC.
6. **Name what is deferred and what stays.**
   - Cite `finetune/notes/audit_amendment3_third_raw.md` by name.
   - State that its open items (M5, M8, M9, 4(f)) were not registered and pass to the redesign.
   - State that Amendment 3's text stays as registered (not withdrawn), and that its withdrawal of the L4 row still
     holds.
