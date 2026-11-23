**Amendment 4, 23 November 2026, 12:15 UTC: E7 and E8 deferred; no intervention has been run.** Reviewed
independently (`finetune/notes/audit_amendment4_raw.md`, PASS WITH CONDITIONS; every fix applied below).

**State at the time of writing.**
- No steering or patching vector has been applied to any fine-tuned model.
- The power pilot `ft_e7e8_pilot` has never been pushed to Kaggle. No pilot or E7/E8 result exists.
- The pilot stays in state "queued" in `finetune/kaggle/queue.json`, as the owner instructed (23 Nov). Its note now
  reads "deferred by Amendment 4; do not launch".
- HALT is not set, so nothing mechanical prevents a launch. It stays unlaunched because no queue-runner loop is
  started. Anyone who runs `kaggle_run.py run` or `launch` must withdraw the entry first.

1. *What was seen, and when.*
   - The pre-intervention diagnostics of Amendment 3 were computed and seen on 23 Nov, first at about 04:35 UTC and in their final form by 04:54 UTC
     (`finetune/scripts/e7e8_prepare.py` → `finetune/e7e8/pilot_prepare.json`; commits 0d44366 and 01e6908).
   - Every one of them was seen before any intervention run.
   - They use E6's stored activations, the gold labels, the models' stored answer scores, and the tuned text stack's
     predictions. From these they compute directions, controls, donor availability and own-answer decodability at the
     site. They contain no intervention outcome.
2. *What they showed, read under Amendment 3's own rules:*
   - **E8 would be descriptive only.** The third independent review (`finetune/notes/audit_amendment3_third_raw.md`)
     found that 35.6–38.3% of items lack an opposite-label or a same-label donor at every candidate N_items for the
     registered items (in the pilot, 72 of 198, 36%).
     - Section 3's 30% rule applies "in any model". Donor availability depends only on item metadata, and the items
       and donors are shared by all models, so the share is the same in every model.
     - E8(A) and E8(B) would therefore enter the family with p = 1.
   - **A "not shown" would be uninformative for both A and B** (Section 2, "Degenerate g").
     - The r trigger alone decides this for both conditions: the projected gold direction keeps less than 10% of its
       norm in A_s1 (r = 0.097), A_s3 (0.089), B_s2 (0.079) and B_s3 (0.060).
     - Separately, A_s4's and B_s4's report-half separation along g is below their median control's. That
       separation was computed over all report-half items, pilot premises included.
   - **B's site: the model's answer is already about as decodable there as at the answer position.**
     - At (content_end, layer 16) the own-answer probe scores 0.974–0.984 within string, against 0.994–0.996 at the
       final position.
     - Amendment 3 registers the qualifier "at a site where the model's answer is already as decodable as at the answer
       position" when the site value is ≥ the final-position value − 0.02.
     - Amendment 3 does not say how that rule is aggregated over seeds. It holds on the seed mean (0.981 against 0.995)
       and in four of five seeds (B_s0, B_s1, B_s2, B_s4). It misses in B_s3 by 0.0004 (0.9739 against 0.9743).
     - At A's site (layer 14) it holds in no seed (0.84–0.89 against 0.96–0.99).
   - **So, as registered:**
     - E8 could not be tested.
     - A "not shown" from E7 in either condition would be uninformative.
     - Only a claimed E7, positive or negative, could have been informative, and E7 could still have been dropped by
       the power rule.
3. *Decision (owner, 23 Nov):*
   - E7 and E8 are **deferred**. No intervention runs under Amendment 3.
   - They will be redesigned **after 30 November 2026**. A redesign is a new amendment with its own independent
     review, registered before any intervention runs.
   - The redesign may use what the diagnostics showed. That is why they are recorded here, and no Amendment 3 rule is
     to be changed to rescue the current design.
   - Open items pass to the redesign: those of the third review (M5 the merge precision, M8 the donor order, M9 whether
     a descriptive-only E8 counts towards N_items, 4(f) how the upstream qualifier is aggregated), and the
     registered-run analysis code, which is not written.
   - Amendment 3's text, including its withdrawal of Section 0's L4 row, stays as registered.
4. *On the page:* E7 and E8 are reported as "registered (Amendment 3), then deferred before any intervention, because
   the pre-intervention diagnostics showed the design could not give an informative null and could not test E8
   (Amendment 4)". Nothing in Amendment 3 or in the diagnostics is evidence about whether the label direction is used.
5. *Unchanged:* every registered test of Section 3 and every result of E1–E6.
