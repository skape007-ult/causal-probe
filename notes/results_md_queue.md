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

## Update 16:53

- **Q1 (cuts): the user chose A′ + B to F, with the ladder figure inline and the other two as links, and asked to see the final page before it is committed.** The page is prepared (793 words) as `notes/RESULTS_cut_preview.md`; the template that produces it is in the scratchpad and is not installed. Waiting for the user's go.
- **Q2 (status of the runs): answered by the user.** Gemma-2-9B returned (ingested, audit pending). The Qwen 3B settle run and the Qwen 3B all-pairs patching run were launched at about 16:50. The Gemma all-pairs patching run is not launched. No fine-tuning pilot exists. Proposed status line, under the Run line, if the user wants one: "Not in this page: Gemma-2-9B (returned 14 Nov, under audit); the held-out test of the Qwen2.5-3B case and patching on all available pairs for Qwen2.5-3B (launched 14 Nov)."

## Q3. A second model whose answers follow validity within a hypothesis string (queued 16:53; depends on the audit of the 9B run)

The page says of the 7B model that its answers "follow validity on average, weakly". Gemma-2-9B does the same by the same statistic: mean AUROC within hypothesis string 0.542 [0.522, 0.560] (p 0.001), 0.541 with adjacency held fixed, valid item of a premise higher in 52.6% of premises; ranked across all items at once it is at chance (0.512 [0.490, 0.535]). By type it differs from the 7B model (with the label for direct causation and common cause, against it for indirect causation). Reason to change the page: without it the page presents the 7B model as the single exception, which the result files no longer support. Proposed, if the audit confirms the numbers: in the Run line "Qwen2.5-7B (4-bit) and Gemma-2-9B: answers, and probes on the debiased pool"; in claim 2's caveat "The 7B and 9B models' follow validity on average, weakly (AUROC within string 0.549 and 0.542) …"; in the first limitation "The 7B and 9B models follow it weakly … and were not intervened on". Whether claim 2's probe numbers get a 9B clause depends on the probe ladder (running).

## Update 16:56: Q1 and Q2 applied on the user's instruction

The user (16:55): "Commit the cut page with two changes: add the status line you proposed; and remove first person ('not my reading' → 'not the conclusions'; any other 'I/my') so the page reads neutrally." Applied: package A′ + B to F, ladder figure inline, the other two figures as links, the status line under the Run line, "not the conclusions", and the heading "What was retracted, and why" (it was "What we retracted"). 834 words. Q3 (a second model, Gemma-2-9B) remains queued until its audit.

## Q3, revised 17:33 after the sixteenth review (replaces the Q3 above; nothing applied)

The user asked: if the 9B result survives its audit, queue a revision from "the 7B model" to "the two largest models (7B, 9B; two families)". **It did not survive** (not distinguishable from following prompt length), so that revision is not proposed. What the audit supports, for the user's decision:

- **Q3a. Status line.** Now: "Not in this page: Gemma-2-9B (returned 14 Nov, under audit); …". Proposed: "Not in this page: the held-out test of the Qwen2.5-3B case and patching on all available pairs for Qwen2.5-3B (launched 14 Nov)." Reason: the 9B run is audited; what it showed goes into the caveat below or nowhere.
- **Q3b. Claim 2's caveat, the 7B sentence.** Now: "The 7B model's follow validity on average, weakly, and go against it for direct-causation hypotheses (0.442): AUROC within string 0.549 [0.530, 0.569]; the valid item of a premise gets the higher answer logit in 53.6% of premises (Qwen2.5-3B: 52.0%)." Proposed: "The 7B model's follow validity on average, weakly: AUROC within string 0.549 [0.530, 0.569], about 0.53 with prompt length removed; the valid item of a premise gets the higher answer logit in 53.6% of premises (Qwen2.5-3B: 52.0%). Gemma-2-9B has the same raw figure (0.542) and it is not shown beyond a response to prompt length (0.52 with length removed)." Reasons: (1) the clause "go against it for direct-causation hypotheses (0.442)" is not robust (5 of 14 strings; 0.530 within adjacency cells); (2) the 7B figure needs the control that the 9B failed, or the page invites the same objection; (3) a reader who knows a 9B model was run should find what it showed. The numbers "about 0.53" and "0.52" would be placeholders read from `answer_bias.json` once the control is in it.
- **Q3c. First limitation.** Now: "The 7B model follows it weakly, was run in 4-bit quantisation, and was not intervened on." Proposed addition: "Inside a hypothesis type the pool still leaks through prompt length, which Gemma-2-9B's answers follow." Reason: it is a defect of the debiased pool that the page's claim 2 rests on.
- **Not proposed:** any sentence about the type-by-type difference between the two models. It is in the report and the handoff; on the page it would need three qualifications.

Cost in words if all three are accepted: about +45.

## Update 17:44: Q3a to Q3c and the review count applied on the user's approval (commit `997471b`; 863 words)

## Q4. Claim 2's headline (queued 17:44 on the user's request; not applied; may need revising once the length-augmented comparison is audited)

Now: "With every single-input shortcut removed, a probe on the activations of the models from 1.5B to 7B is about as accurate as …"

Proposed: "With the hypothesis-only and premise-only shortcuts removed (prompt length still leaks within a hypothesis type), a probe on the activations of the models from 1.5B to 7B is about as accurate as …"

Reason: on this pool the length of the prompt predicts the label inside a hypothesis type, and a reader of "every single-input shortcut removed" would take it that nothing in the text predicts the label without the causal reasoning. One correction to the premise of the request, for the record: the leak is not strictly single-input. Every premise appears once with a valid and once with an invalid hypothesis, so any function of the premise alone, its length included, predicts nothing (the token count pooled over all items has AUROC 0.494); it is premise length *together with* the hypothesis type that predicts the label. The headline's literal statement is therefore true and misleading, and the parenthesis fixes the misleading part.

Length: +10 words. Compensating cuts, so that the page stays at 863 words or fewer: the Run line's last sentence "Numbers are from `results/*.json`; details and all caveats are in `report/report.md`." → "Numbers: `results/*.json`; caveats: `report/report.md`." (−7); claim 1's figure link text "probe against text, four of the eight cells" → "four of the eight cells" (−4). Net −1.

If the length-augmented comparison (running) moves the probe below the lookup, "about as accurate as a lookup" will need changing as well; that would be a separate queued item.

## Q5. Run line (queued 17:44; depends on the 9B probe ladder, which has not finished)

Now: "Qwen2.5-7B (4-bit): answers, and probes on the debiased pool." The caveat of claim 2 now mentions Gemma-2-9B, which the Run line does not. Proposed, once the 9B probes exist: "Qwen2.5-7B (4-bit) and Gemma-2-9B: answers, and probes on the debiased pool." (+3 words)

## Q6. The Qwen 3B held-out test and the end of the status line (queued 19:07; waits for the seventeenth review of both Qwen 3B runs)

- **"The four", now:** "Qwen2.5-3B, end of the hypothesis, class-mean difference (+0.738); Qwen2.5-1.5B, same token, probe direction (−0.251; inside its 500 controls in a test fixed beforehand, p = 0.12); Gemma-2-2B, answer position, probe direction (+0.475, first by 3%); Qwen2.5-0.5B, original pool, class-mean difference (+0.865). Three are undecided and one was not confirmed."
- **Proposed:** "Qwen2.5-3B, end of the hypothesis, class-mean difference (+0.738) and Qwen2.5-1.5B, same token, probe direction (−0.251) were tested again by a rule fixed beforehand, refitted on the other half of the data: both kept their sign and lay inside 500 controls (p = 0.08 and 0.12). Gemma-2-2B, answer position, probe direction (+0.475, first by 3%) and Qwen2.5-0.5B, original pool, class-mean difference (+0.865) are undecided." (about the same length)
- **Status line:** remove it (both runs it names are in). −22 words.
- **Counts:** "Sixteen reviews" → the number on file when this is applied; "Checked by … claim 3, audits 2, 4, 5, 6, 8, 11 and 15" → add the seventeenth.
- **Reason:** the Qwen 3B test returned the registered outcome "p > 0.05" (p = 0.082; slope +0.569, 77% of the first, which a response unrelated to the label predicts from the overlap of the two halves' directions). The page currently calls that case undecided and says a test is running.
- **Not proposed:** anything about patching; the page does not mention it. The report (Section 4.2) gets the all-pairs result.
