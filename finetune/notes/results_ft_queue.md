# Wording queue for `finetune/RESULTS_FT.md` (waits for the owner; CLAUDE.md item 4)

## 1. Sat 21 Nov: first version of the page (behaviour, E1–E5)
- **Proposed text:** `finetune/notes/RESULTS_FT_draft.md`, all of it.
- **Reason:** the registered analysis is complete and audited. The audit verdict was PASS WITH CONDITIONS, no blocking
  finding (`finetune/notes/audit_behaviour_raw.md`), and the draft meets each condition.
- **Decisions for the owner:**
  - **Claim headlines:** the three headline sentences and their wording.
  - **A's rung:** whether A's rung sentence ("L2, above the registered copy reference only; not described as beyond
    copying") is how you want it.
  - **E3:** whether E3 stays outside the three claims.
  - **Interpretation:** the section, which is yours.
- **Status:** awaiting the owner. Due to the owner by Sun 22 Nov 12:00 EST (17:00 UTC).

## 2. Mon 23 Nov: claim 3's scope, and the registered Table-5b analogue (found by the defence-pack session)
- **Proposed text:** in the E4 claim, "not shown to be due to the reversal of letter order" → "not shown to be due to
  the reversal of letter order within a hypothesis string".
- **Reason:** S is a within-string statistic, so E4b cannot see a string-level letter-order cue. Separately, the
  descriptive table registered beside E4 (pooled F1 on ID, ID renamed, ID paraphrased and the authors' sets; PREREG
  Section 4, E4) was not computed, and neither the audit nor the draft lists it as a deviation. The scores exist in
  finetune/runs/*/scores.npz.
- **Decisions for the owner:** the wording; whether to compute the table and have it audited, or list it as a
  deviation.
- **Status:** owner, 23 Nov: wording approved for the draft ("within a hypothesis string" added to claim 3's heading
  in RESULTS_FT_draft.md); the table is to be computed, audited and added to the draft with a deviation line.
  Nothing entered in RESULTS_FT.md.

## 3. Mon 23 Nov: E6 (representations), descriptive section
- **Proposed text:** a short descriptive section built from `finetune/notes/e6_summary.md`, keeping its "Read this
  first" paragraph word for word. That paragraph says the probe is trained on in-distribution gold labels, so "probe
  above answer" is not "present and unused", and that the text stack fitted on the same labels beats the probe
  everywhere. It also states that for B the probe is at most 0.01 to 0.02 above B's answers, with ρ ≈ 0.9.
- **Reason:** E6 is audited (PASS WITH CONDITIONS; all seven met in e6_summary.md). PREREG keeps it descriptive.
- **Decisions for the owner:** whether E6 appears on the page at all; the wording; whether you approve the use of a
  run marked FAILED by a notebook bug (see e6_summary.md, "Run status").
- **Status:** awaiting the owner. Nothing entered.
