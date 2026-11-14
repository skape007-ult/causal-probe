# RESULTS.md: changes waiting for the owner's answer

Rule (user, Sat 14 Nov, about 12:25): "any change to RESULTS.md wording waits for my answer; queue it with the reason instead of entering it." Nothing in this file has been applied. `RESULTS.md` is at 1,105 words (commit `b6b32b0`).

## Q1. Cuts to bring the page to one printed page (asked for by the user; proposed 12:30)

Word counts now: header 77, claim 1 77, claim 2 212, claim 3 332, retraction table 253, limitations 132. Each cut says where the removed content already is in the report, so nothing is lost.

| Cut | What goes | Where it already is | Words saved |
|---|---|---|---|
| A | The retraction table, replaced by one sentence: "The claims and readings retracted on the way, each with the review that caught it, are in report Section 6." | Report Section 6, review by review | about 225 |
| A′ (instead of A) | Keep three rows of the table (the first positive verdict, audit 1; the first steering summary, audit 2; the adjacency result "beyond all controls", audits 4 and 5) and drop the rest with the same sentence | Report Section 6 | about 140 |
| B | "The four" as one sentence with the slopes only: "Qwen2.5-3B, end of the hypothesis, class-mean difference (+0.738); Qwen2.5-1.5B, same token, probe direction (−0.251; inside its 500 controls in a test fixed beforehand, p = 0.12); Gemma-2-2B, answer position, probe direction (+0.475, first by 3%); Qwen2.5-0.5B, original pool, class-mean difference (+0.865). Three are undecided and one was not confirmed." | Report 4.1 (what each direction is, natural scale, the held-out test) | about 55 |
| C | The two adjacency sentences of claim 3's caveats become one: "Shifting items along the encoding of adjacency before the answer position is outside its controls in one of four models (figure); at the answer position it moves the answer the same way for every hypothesis type, which locates the answer." | Report 4.4 | about 45 |
| D | Claim 2: the parenthesis on the 7B probe's gain over the text model, and "equal accuracy is not equal content (…)" | Report 3.6 | about 50 |
| E | Limitations: the clause "(a test was costed and not run; report Section 9)"; in the second bullet "one layer … an erasure that removes about half of the readable adjacency on held-out items" | Report Sections 7 and 9 | about 35 |
| F | The three "Checked by" lines become one line at the end of the claims | – | about 10 |

- **A + B + C + D + E + F: about 685 words.** This fits one printed page as text. It cannot keep three full-width figures on the same page: with this package I would keep the ladder figure inline and turn the other two into links.
- **A′ + B + C + D + E + F: about 770 words.**
- **B + C + E + F only (table and claim 2 untouched): about 960 words.**

What I would not cut, because the page misleads without it: the 7B within-string sentence with 53.6%; the four named cases; the mean-rank caveat with "does not show that validity is used"; "'No intervention shows' is not 'unused'"; the limitation that every corrected analysis was introduced after seeing results.

My recommendation: A′ + B + C + D + E + F. The retraction table is the part of the page a reader is least likely to find elsewhere, so I would keep its three most consequential rows.

## Q2. Status of the runs (the user's item 2 of 12:25)

The page has no status line at present (the sentence "a test … is running" was removed at 11:40 because I could not confirm the launch). The user's message asked me to state what is live, finished and not launched, in a bracket that reads like a placeholder to be filled in. What I can see from this machine: one round-3 download has arrived (`r3s_qwen15b`, ingested and audited); nothing for `r3_gemma9b`, the two patching runs or `r3s_qwen3b`; no branch `exp/finetune` in this repository or on its remote, and no other session of this project in the app. I cannot see Kaggle. **Waiting for the user to say what is live.** Proposed line once known, under the Run line: "Not in this page: <runs>, <state>."
