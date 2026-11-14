# Claims check of the 14 November text (commit b13f9e9)

Nothing in the repository was written; `git status` is clean. All templated numbers in the checked sections match their result files; the problems are in hand-typed sentences. Line numbers refer to the built files.

## Must fix

1. **report §6, ninth/tenth audits:** "…in `notes/log.md` (14 Nov 01:40, 02:45, 07:56), written before any run." **Wrong.** The 07:56 entry was written after the Qwen 1.5B launch (log 07:56; fifteenth review F6). §4.1 has it right. → "The test and threshold were fixed before any run (01:40, 02:45); the sentence for each outcome was recorded at 07:56, during the first run and before any output."

2. **report §7 Steering:** "No matched controls for Gemma-2-2B yet." **Stale; contradicts the same bullet and §4.** Evidence: `steering_controls_gemma2b_*.json`. → delete.

3. **report §6 "What the audits did not cover":** "Gemma-2-2B's second-round rows have not arrived" and "The paragraphs added … on 14 November have had a claims pass". **Stale / wrong.** The rows are in (eleventh audit); log 11:29 says no claims pass was done on the new paragraphs. → "Gemma's second-round rows were audited (eleventh). The paragraphs added on 14 November after the eighth audit had no claims pass when written."

4. **report summary 5:** "two cases remain undecided … and are being put to a held-out test". **Stale.** Qwen 1.5B has returned (`steering_settle_qwen1.5b…/primary_test`: −0.112, p 0.120). → "The Qwen2.5-1.5B case kept its sign at under half the size and lay inside its 500 controls (p = 0.12); the Qwen2.5-3B case is untested."

5. **report §3.6 (l.636):** "…which earlier versions of this report called "that fixed stack"". **Wrong (search-and-replace damage).** They called it "the strongest text-only model" (table note, l.634). → "…called "the strongest text-only model"; below, "the fixed stack"."

6. **report §3.6:** "A small gain in AUROC … from Qwen2.5-1.5B upward over every text model tried … Neither is in a result file". **Wrong and overstated.**
   - `incremental_value/over_tuned_text_only/gain_auroc` is in all five files.
   - Over the tuned stack the interval includes zero for 1.5B (+0.0013 [−0.0005, +0.0031]) and Gemma (+0.0015 [−0.0001, +0.0032]); it excludes zero for 3B and 7B only.
   - The review tried richer text models for 7B and 3B only.
   → "Over the tuned stack the AUROC gain is +0.001 to +0.002; its interval excludes zero for Qwen2.5-3B and 7B only (result files). The end-of-hypothesis gain is the review's and is in no result file."

7. **"adjacency, not validity":**
   - report summary 1: "the four models' answers follow adjacency, not validity"
   - report §3.1 Reading: "what these answers follow inside a string is adjacency, not validity"
   - report §7: "follow adjacency and not validity"
   - RESULTS/README claim 2 caveat: "depend on adjacency of the queried pair, not on validity"

   **Overstated; contradicts §4.1.** `answer_bias.json`: Gemma within-string 0.473 (p 0.0025); with adjacency fixed, Qwen 1.5B 0.436 and Gemma 0.467 (both p 0.0005). These answers do depend on the label, negatively (thirteenth review W3.3). The RESULTS wording "depend on … not on validity" is stronger than the reviewed "follow". → "do not follow validity (within-string AUROC 0.47 to 0.51); they depend on adjacency, with signs differing by type, and in Qwen2.5-1.5B and Gemma-2-2B run against the label once adjacency is fixed."

8. **EXPLAINER Q48:** "the audit found the mechanism, and it is not use of validity"; "…which is why the two cases at that token are the ones in the held-out test". **Overstated (W3.4) and wrong order.**
   - The account covers the answer position on the debiased pool, 4 of the 16 comparisons behind p = 0.001.
   - The two cases were chosen at 00:55–02:45, before the eleventh audit.
   → "Because for the part the audit could explain (answer position, debiased pool, three of four models) the comparison does not show use of validity … At the end of the hypothesis it is unexplained; two cases there had already been put to a held-out test."
   - Same fault in **progress.md:** "explained by behaviour, not by use of validity".

9. **EXPLAINER Q46 and D34(3):** outcomes quoted as "held up…" / "did not hold up". **Stale and misquoted.**
   - The registered sentences (log 07:56) were "replicated on held-out data…" / "did not replicate on held-out data".
   - Q46 still treats the test as pending.
   - Log 09:18 says "did not hold up" has the same fault as "did not replicate".
   → add to both: "Outcome, Qwen2.5-1.5B: p = 0.12 (category 'p > 0.05'); the slope kept its sign at under half the size and lay inside its 500 controls. The registered sentence 'did not replicate' was replaced after the fifteenth review."

10. **Runs called launched or running.** **Unsupported.**
    - report summary 9 and §4 status box: Qwen 3B test, 240-pair patching and Gemma-2-9B "were launched on 14 November"
    - RESULTS/README "The four": "a test against 500 controls … is running"
    - README Status: "Three GPU runs launched"

    The log and progress.md record launches of `r3s_qwen15b` and `r3_gemma9b` only; the rest is "Next". Study "Still to be added" says "if it is run". The README list is four runs (two patching). → "queued; not yet returned" unless a launch is logged.

11. **README Status:** "every sentence of it was checked … by a reviewer before it was entered". **Overstated.** The thirteenth review checked the draft; the settle sentence, the 7B parenthetical, "Fifteen reviews" and "Checked by" were added later. → "the draft was checked sentence by sentence (thirteenth review); sentences added afterwards rest on the fourteenth and fifteenth reviews."

12. **progress.md, status 08:55.** **Stale:**
    - "RESULTS.md: unchanged" (replaced 11:29).
    - Option B/C "with an independent reviewer" (decided: C, power about 0.2).
    - `r3s_qwen15b` "running" (returned, audited, p 0.12).
    - The 07:56 wording (replaced).
    - No mention of the fourteenth review.
    → rewrite the block or head it "superseded 11:29".

## Should fix

13. **RESULTS "Run" line; report §6 intro:** fifteen reviews "given code and results, not my conclusions". **Overstated.** The thirteenth reviewed my proposed page, the twelfth my note, the sixth RESULTS.md. → "fifteen reviews by fresh reviewer agents; those of results were given code and files, not my reading."

14. **report §3.1 (l.151); study Q53:** "permutation p 0.000". **Wrong display**; the file and table give 0.0005. → "0.0005".

15. **report §7 Numerics:** "most probe fits behind the steering directions". **Overstated (W10):** 129 of 264; "most" holds for Gemma only. → W10's sentence.

16. **report §7 Multiplicity:** "(two of them at sites that do not respond)". **Inconsistent with §4.1**, which drops four comparisons at two sites (`summary/all/sites_not_responding`). → "(four at two sites whose positive control does not exceed its controls)".

17. **report §7 Adjacency:** "one informative layer per token" — Gemma has none at the end of the hypothesis. **§4.4 Limits:** "four of the eight sites of the two GPU runs" omits Gemma's three non-converged sites (eleventh audit F5).

18. **EXPLAINER D36 Outcome:** "a pool balanced for length inside the cells would [help]". **Overstated.** Balancing removes the confound, not the low power; the review's main reason is the share one site carries. → "…would remove the length confound, not the low power."

19. **EXPLAINER Q44:** "even a real effect would not be an effect of validity" (W5: "cannot separate"). **Primer:** "largely through adjacency" (inference, W3.3). **D34:** "launched the held-out test of the two steering cases" (one was launched).

20. **report §6:** "announced a replication in 57% of runs". This is Qwen 3B; it was 28% for Qwen 1.5B (ninth audit F2). → give both.

21. **report summary 1:** "Qwen2.5-3B is the only model above chance on CLadder". **Inconsistent with §3.1** (7B 0.544 [0.520, 0.568]). → "the only one of the four".

22. **report §4.1:** mean rank 4.88 "(0.001)". The file has 0.0015; the audit says 0.002.

23. **README l.160:** AUROC gains "present from Qwen2.5-1.5B upward" beside the tuned stack — see 6. **README Patching:** "(four models) … the two larger models". Unclear which two.

24. **power_7b_option_b.md:** "every number here is in `results/power_option_b_qwen7b.json`". False for the appended part. The appendix also omits review F5 (share capped at 0.2–0.3 by the 7B data) and F6 (g falls to 0.26–0.36).

## Optional

25. "about 240 pairs" (summary 9, README) against §4.2 "about 210 to 230 per kind".
26. Patching recovers nothing "from about layer 18" (report §4.1) against "layer 16" (EXPLAINER Q49, study Q55).
27. Status box "three readings" against §6 "four".
28. Summary 9: "everything for … Qwen2.5-7B ran on Kaggle GPUs" — the probe ladder ran on the laptop.
29. "strongest text-only model" survives in study/01 l.97, study/02 l.119, 123, 183 and EXPLAINER l.25.
30. Study "Still to be added" lists the held-out test and the 7B probe comparison, both now in the report.
31. EXPLAINER primer "12 on the debiased pool … 2 were seen" (now 16 and 3).
32. §4.4 heading still says "all three Qwen models".

## Checked and found correct

- **RESULTS.md:**
  - W1–W6 are not reintroduced as listed; item 7 is the one regression.
  - Every number matches its file.
  - The settle sentence matches the fifteenth review.
  - The 7B parenthetical matches the fourteenth review and `over_tuned_text_only` (lower bound exactly 0.0).
  - The "Checked by" lists are right.
  - Fig. 12 marks "NO TEST"; the fig. 13 caption says "Pooled".
- **report §3.1:** the table, 33/48, by-type values, 53.6%, the three cautions, and the 7B comparisons with other models.
- **report §3.6:** 18 net items, p 0.015 / 0.077, selection half +0.009, half-data about half a point, 0.865 → 0.887 > 0.880, the five tuned gains ("one of five excludes zero" holds), 0.895, and the three exclusions.
- **report §4.1:**
  - Gemma: 0.475 against 0.462, 96th percentile, a fifth of resamples, ranks 6/12/5/4, positive control rank 7.
  - Counts: 4/32, 3/16, 4/28, 3/14 with their probabilities.
  - Mean ranks: 6.44, 8.31, 10.00; 15/16 positive.
  - Alignment: r values, predicted and observed slopes, cosine ranks.
  - The three W3 limits are stated.
  - Held-out test: every number; 98% of items and every block; 0.53 × −0.251; ratio 9 against about 4; p 0.044; cosines 0.997 and about 0.8; the within-cell observation with its caveats; the 07:56 timing.
- **report §4.2 and §4.4, points 15–17:** all numbers match the eleventh audit.
- **report §6:** the eleventh-audit table and the twelfth to fifteenth rows.
- **report §9:** implies no results. 1,200 premises / 2,400 items / 96.7% six-variable; 205,734 items, 2,338 premises, pool all from train; 94.74, 39.29, 62.30; no hypothesis-only baseline in the paper; power about 0.2; the length confound.
- **EXPLAINER:** D35, Q47 (says "pooled"), Q49, the "pooled score" primer, and D36's numbers.
- **Study:** Session 8 ideas; the numbers in Q53–55.
- **Power note:** part 1 matches its JSON; part 2 matches the review's verdict.
