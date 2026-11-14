# Power of option B's test on the 7B model, computed before anything is built

Written Sat 14 Nov, about 08:50, at the user's request, to decide between option B (a pre-registered intervention on Qwen2.5-7B) and option C (none). **Not yet reviewed when first written; the review's verdict is appended at the end.** Script: `scripts/18_power_option_b.py`; every number here is in `results/power_option_b_qwen7b.json`.

## The test

At the last token of the hypothesis, steer along the gold label's direction beyond adjacency: the label minus its mean in its (hypothesis string, adjacency) cell, projected on the centred activations of a fitting half, with that half's adjacency direction projected out. Compare its slope, one-sided, with 100 control directions built the same way from labels shuffled within those cells. "Detects" means p = (1 + controls at least as large) / 101 ≤ 0.05. This is the comparison the tenth audit found calibrated (2 to 3% false positives when the model responds to adjacency only). 150 steered items, one step of one unit of s.

## What had to be assumed

A power needs an effect size in the units of the test (a slope in logits per unit of s). The user's effect sizes are in units of behaviour: (a) "an effect the size of the 7B answers' dependence on validity", (b) half that. The translation between the two is the whole calculation, and it is an assumption:

- **A1.** The answer depends on the activation at the site through one linear direction (the true label-beyond-adjacency direction), by as much as the answers depend on the label within cells: moving an item across the within-cell class gap along that direction changes the answer by g = +0.440 logits (the within-cell regression coefficient of the 7B answer logit on the label; within string only it is +0.259). That is share f = 1; half is f = 0.5.
- **A2.** The response is linear up to the step of the test. The class gap along the direction is 0.2 to 0.3 natural SDs, far below one unit of s, so the slope at f = 1 comes out at **9.5 to 16.5 logits per unit of s**, depending on the layer.
- **A3.** The true direction is not known; a direction fitted on the other half stands in for it (cosine between the two halves' directions 0.76 to 0.88). The stand-in's error cancels for the tested direction.
- **A4.** Everything else is label-free noise in the slope, the same for the tested direction and its controls, with SD σ = 2 × 0.376 logits per s: twice the largest SD of class-mean-difference control slopes at the end of the hypothesis on the debiased pool in the three Qwen models (0.196, 0.099, 0.376). Doubling is the user's instruction; ×1 and ×4 are given for sensitivity.
- **A5.** The site responds at all.

## Result

Power, range over the layers 8 to 28 (every second), eight fitting halves:

| Share of the behavioural dependence carried by the direction (f) | σ × 1 | **σ × 2** | σ × 4 |
|---|---|---|---|
| **1 (the user's case a)** | 1.00 | **1.00** | 0.97 – 1.00 |
| **0.5 (the user's case b)** | 1.00 | **0.97 – 1.00** | 0.70 – 0.83 |
| 0.25 | 0.96 – 1.00 | 0.70 – 0.84 | 0.34 – 0.52 |
| 0.1 | 0.56 – 0.73 | 0.28 – 0.44 | 0.14 – 0.21 |
| 0.05 | 0.29 – 0.42 | 0.14 – 0.21 | 0.09 – 0.12 |
| 0 (nothing label-specific) | 0.04 – 0.06 | 0.05 – 0.06 | 0.04 – 0.05 |

**By the rule as fixed (power for (a) ≥ 0.6 → build B), the answer is B: 1.00.**

## Why I do not trust that answer as it stands

The calculation says the test cannot miss an effect of size (a). I think that says more about (a) than about the test.

1. **The slope that (a) implies is far larger than anything measured in this project.** 9.5 to 16.5 logits per unit of s, where the positive control (a step along the model's own answer direction) at the same token in the three Qwen models gave 0.51 to 0.81, and 5.3 at the answer position of Qwen 3B. Either A1 or A2 fails well before that.
2. **In the smaller models a site of this kind carried a small share of a dependence that is certainly there.** Their answers depend on adjacency. A natural-scale shift along the adjacency class-mean difference at the best end-of-hypothesis layer moved the answer by 11.5%, 5.8% and 0.6% of that dependence (Qwen 0.5B, 1.5B, 3B; contrast over benchmark of the adjacency test). If the 7B model's validity dependence is carried at such a site to a similar extent, the row to read is f = 0.05 to 0.1: power 0.14 to 0.44.
3. **A5 is not in the calculation.** One of four models (Gemma-2-2B) had an end-of-hypothesis layer, chosen for decodability, that responded to nothing. A dead site turns any f into power 0.05.
4. **Nothing has run on a 4-bit model**, and its arithmetic (bfloat16) is noisier than the other models'.

So: under the literal reading of (a) the rule says B. Under the share that the comparable measurements in the smaller models suggest, the power is below 0.5. Which reading is the right one for the rule is what the independent review is asked.

## After the independent review (09:10; raw report: `notes/audit_power_option_b_raw.md`)

The reviewer made their own estimate before reading the script or this note, then reproduced the script's numbers (they reproduce) and reviewed both.

**Verdict: the honest power for (a) is about 0.2 (range 0.12 to 0.32) with the control spread doubled, about 0.1 for (b); with the spread not doubled, about 0.35 and 0.19. Below 0.6.** The reading the reviewer defends is the one the user chose for the rule: the share that comparable end-of-hypothesis sites carried in the smaller models (0.115, 0.058, 0.006, and one site that does not respond). Share needed for power 0.6: 0.14 to 0.23; one of four smaller sites reached 0.115.

What the review found wrong or missing in my calculation:

1. **The tested direction is mostly prompt length, and its controls are not** (critical). Conditioning on adjacency breaks the balance of length against the label: inside (string, adjacency) cells the residual label correlates −0.109 with the token count. The tested direction has cosine −0.55 to −0.75 with the length direction; at most 3 of 100 controls load on length as strongly. The 7B answers depend on length inside cells more than on the label (correlation −0.26 against +0.14). So my row "share 0 → power 0.05" does not hold: a site that responds to length alone would return p ≤ 0.05, and a positive result could not separate validity from prompt length. The calibration I cited from the tenth audit (adjacency-only response) does not cover this.
2. **The "slope at full share" is an identity** (g × s_unit / the norm of the fitted mean difference). The stand-in direction cancels exactly, so the number says nothing about whether a label direction exists; a shuffled label would give twice the slope.
3. **Which assumption fails:** A1, not A2. The full-mediation formula predicts the measured slopes of gold class-mean differences at the final token (Qwen 3B: 3.27 predicted, 4.02 measured) and fails at the end of the hypothesis (4.31 predicted, 0.74 measured); responses are linear in the step where measured.
4. **My note and my code disagreed** about the controls (the code adds a share-proportional term to them, the note says pure noise). The code is the more defensible; at share 1 it implies control spreads ten times anything measured, another sign that share 1 is not a realistic case.
5. **The dead-site risk** (one in four) was in my text and not in my table.

**Would more items help? No.** An unlimited fitting pool raises the slope by about 12%; item noise is under 1% of the control variance; 100 controls are enough. What limits the test is the share one layer at one token carries, the model's response to arbitrary directions, and whether the site responds. A new pool could remove the length confound and the one-class cells if prompts are balanced for length inside the cells: a change of design, not of size.

**Decision (the user's rule, as the user clarified it at about 09:00): option C.** No intervention on the 7B model in this phase. The reason the user had pencilled in ("it needs more items than the official debiased pool provides") is not what the review supports; the reason written into the report is the reviewed one.
