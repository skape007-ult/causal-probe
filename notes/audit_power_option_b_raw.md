# Twelfth review: the power calculation for option B on the 7B model (raw report)

## Summary (under 400 words)

**The honest power for (a) is below 0.6: about 0.2 at σ × 2 (range 0.12 to 0.32), and about 0.1 for (b) (0.07 to 0.15).** At σ × 1, (a) is about 0.35 (0.23 to 0.50).

**Report file not written.** The harness refused the write to `audit12/report.md` ("subagents should return findings as text"), and I did not work around it. The full report is below; my scripts and outputs are in the `audit12` folder.

**The author's numbers reproduce.** Share 1: 0.996 to 1.000; share 0.5: 0.966 to 0.996 (σ × 2, layers 8 to 28). The direction and control construction, seeds, splits and p-value match 07d.

**Reading I defend:** the 7B dependence is carried at the tested site to the extent adjacency was at the smaller models' end-of-hypothesis sites (0.115, 0.058, 0.0056, and a dead Gemma site). The literal reading (power 1.00) needs share 1, which those measurements contradict by a factor of 9 to 300.

**Errors and gaps found:**
- **The tested direction is mostly prompt length.** Its cosine with the length direction is −0.55 to −0.75 in layers 10 to 28, and at most 3 of 100 controls load on length as strongly. The within-cell residual label correlates −0.109 with token count; the 7B answers correlate −0.26 with it, against +0.14 with the label. The test is not calibrated for this, so the "share 0 → 0.05" row does not hold.
- **The "slope at full share" is an identity:** g · s_unit / |fitted mean difference|. The stand-in cancels exactly, and a shuffled label would give twice the slope.
- **The code and note disagree on controls.** The code adds a share-proportional term to them; at share 1 its SD is 3.1 to 7.9 logits per s, against 0.10 to 0.38 measured.
- **A1 fails, not A2.** Full mediation predicts measured slopes at the final token (3B: 3.27 predicted, 4.02 measured) and fails at the end of the hypothesis (4.31 predicted, 0.74 measured). Linearity ratios are 0.99 to 1.05.
- The dead-site risk (1 in 4) is not in the table.

**Given the model does use such a representation:** P(p ≤ 0.05) ≈ 0.3 (0.1 to 0.5). A positive result would not separate validity from prompt length; a null would show almost nothing.

**More items would not help.** An unlimited fitting pool raises the slope by about 12%, and item noise is under 1% of control variance. The limit is the share one site carries against the model's response to arbitrary directions. The rule's sentence "needs more items than the pool provides" would be wrong.

**Recommendation:** below 0.6, do not run option B as designed. The main reason is the measured share at such sites.

**Disclosure:** four early scripts wrote output files into the repository root. I moved them to `audit12` within minutes; the root is clean and nothing else was touched.

---

# Full report — Audit 12: power of "option B" on the 7B model

Independent review, Sat 14 Nov. All numbers were computed with my own scripts (CPU, cached activations and answers only, no model loaded). Order of work as instructed: my own estimate first, then `scripts/18_power_option_b.py` and its JSON, then `notes/power_7b_option_b.md`.

## My power table

Test: p = (1 + controls ≥ real) / 101 ≤ 0.05, i.e. at most 4 of 100 controls at or above the real slope. With normal controls this needs expected slope / control SD ≈ 1.94 for power 0.6 and 2.53 for 0.8.

Full-share slope (layers 16 to 20, where the gold probe is best at this token): 10.6 logits per s for g = 0.440. σ = 0.376 logits per s (Qwen 3B, the largest of the three responding sites).

| Reading | g (logits) | (a) σ × 2 | (b) σ × 2 | (a) σ × 1 | (b) σ × 1 |
|---|---|---|---|---|---|
| R1 literal: one site carries all of it (share 1 / 0.5) | 0.440 | 1.00 | 1.00 | 1.00 | 1.00 |
| R1 literal | 0.259 (within string) | 1.00 | 0.99 | 1.00 | 1.00 |
| R2: share as measured for adjacency at the four smaller end-of-hypothesis sites, author's σ | 0.440 | **0.20** | **0.10** | 0.38 | 0.19 |
| R2 | 0.356 (net of prompt length) | 0.16 | 0.09 | 0.32 | 0.15 |
| R2 | 0.259 | 0.12 | 0.07 | 0.23 | 0.12 |
| R2′: scale-free version of R2 | 0.440 | 0.32 | 0.15 | 0.50 | 0.32 |
| R2′ | 0.259 | 0.18 | 0.09 | 0.37 | 0.18 |

R2 per template at σ × 2, g = 0.440, case (a): Qwen 0.5B share 0.115 → 0.48; Qwen 1.5B share 0.058 → 0.20; Qwen 3B share 0.0056 → 0.06; Gemma (dead site) → 0.05. Mean 0.20.

R2′ takes the z that the adjacency translation actually reached against norm-matched controls in each smaller model (+1.67, +0.72, +0.28; Gemma dead) and rescales it by the full-share slope in units of each model's within-string answer SD. This avoids carrying a control SD in one model's logit scale to another's.

Share needed for power 0.6 at σ × 2 with Qwen 3B's σ: 0.14 (g = 0.440) to 0.23 (g = 0.259). Only one of four smaller sites reached 0.115.

Sensitivity, in order of importance: the share (20-fold range, 0.0056 to 0.115, moves power from 0.06 to 0.48); σ (×1 vs ×2 roughly doubles power in the middle range); g (0.259 vs 0.440 changes power by a factor of about 1.6); layer (flat from 10 to 28); item noise and number of controls (negligible).

## Findings

### F1 (critical). The tested direction is largely a prompt-length direction, and the controls do not share that property

- At the last hypothesis token the largest within-cell principal component is prompt length (token position): 17 to 26% of within-cell variance in layers 16 to 28, correlation with token count ±0.86 to 0.89.
- Every premise has one valid and one invalid item, so length is exactly uncorrelated with the label within string. Conditioning on adjacency breaks that balance: within (string, adjacency) cells the residual label correlates with token count at **−0.109** (shuffled labels: SD 0.03).
- The tested direction therefore has cosine **−0.55 to −0.75** with the length direction in layers 10 to 28. Its projection correlates −0.87 to −0.91 with token count.
- In the primary split, at most 3 of 100 controls load on "shorter" as strongly as the real direction (layers 10 to 28). The real direction's natural SD is also twice its controls' (0.28 vs 0.14 s at layer 24).
- The 7B answers depend on length within cells more strongly than on validity: correlation of the answer logit with token count −0.26, with the label +0.14.
- Consequence: if the site responds to length in the observed direction (shorter → Yes) and to nothing label-specific, the test returns p ≤ 0.05 with near certainty. If it responds with the other sign, the real direction ranks last whatever validity does. The "share 0 → power 0.05" row in the note holds only if the real direction is exchangeable with its controls, and it is not.
- The same structure is present in the smaller models: the option-B direction's |cosine| with the length component is 0.64 to 0.76 at six of eight sites, beyond 95 to 100% of its controls.
- The calibration the note cites (2 to 3% false positives "when the model responds to adjacency only") does not cover this.

### F2 (critical for the decision). The literal reading of (a) assumes a share that this project has measured to be 0.3 to 12%

- Adjacency translation at the best end-of-hypothesis layer, as a share of the full-mediation benchmark: 0.115 (Qwen 0.5B), 0.058 (Qwen 1.5B), 0.0056 (Qwen 3B), 0.003 (Gemma). At the final token the same quantity is 0.56 to 0.98.
- I checked the full-mediation formula against measured slopes of the gold class-mean difference. At the final token it predicts well (1.5B: 2.26 predicted, 1.89 measured; Gemma: −1.91, −1.80; 3B: 3.27, 4.02). At the end of the hypothesis it fails (1.5B: 2.59 predicted, 0.095 measured; 3B: 4.31, 0.74; Gemma: −2.48, +0.008).
- So A1 fails at this kind of site, not A2. Linearity holds where measured: small-step over large-step slope ratios are 0.99 to 1.05 up to 0.5 s at the three responding sites.
- Adjacency in those models was a stronger and far better separated property than validity is in the 7B model, and was steered with 500 items. Even so its translation was beyond all 20 norm-matched controls in the predicted sign at only 2 of 4 sites.

### F3 (major). The "slope at full share" contains no information about whether a label direction exists

- The formula slope = f·g·(u_E·d)·s_unit / gap_F(u_E) reduces algebraically to **g·s_unit / |fitted mean difference on F|**. The stand-in u_E cancels identically, not statistically. I confirmed numerically (layer 20: 10.4 from the identity, 10.8 from the script; the difference is the adjacency projection).
- A shuffled-label direction put through the same formula gives about twice the slope (its norm is half: 0.027 vs 0.054 s).
- The cross-half cosine of 0.76 to 0.88 quoted under A3 overstates how well the direction is identified. Shuffled-label directions from the two halves have cosines with SD 0.23 to 0.44, because both halves are dominated by the length component. The real cosine is 2 to 2.7 SDs out.

### F4 (major). At share 1 the model's own controls would have an SD ten times anything measured

- In the script, controls get a systematic slope f·g·(ctrl·u_E)·s_unit / gap_F. At f = 1 its SD over the 100 controls is **3.1 to 7.9 logits per s**, against 0.10 to 0.38 in the smaller models.
- The note describes A4 as "label-free noise, the same for the tested direction and its controls". The code is not that; it adds this share-proportional term to the controls. The code is the more defensible of the two, and it lowers power at small shares (share 0.1, σ × 2: 0.28 to 0.43 with it, 0.35 to 0.69 without).
- The real direction gets one fixed overlap per split and no share-proportional scatter. With 25% relative scatter, power at share 1 falls from 1.00 to 0.84 to 0.96. Not decisive.

### F5 (major). The 7B model's own cached data cap the share along the tested direction at roughly 0.2 to 0.3

- If the answer depended on the tested direction as A1 says at share 1, the within-cell correlation between the answer logit and the projection would have to be about 0.9 to 1.2 (the answer's label separation, 0.34 SD, over the projection's, 0.29 to 0.39 SD).
- The held-out correlation is 0.19 to 0.27. It is −0.06 to +0.09 in layers 10 to 26 once token count is removed from both.
- This is an upper bound for this direction only, and most of it is length. It does not bound a different, better-separated direction.

### F6 (moderate). Choices in the effect size and σ

- g = 0.440 is the larger of the two candidates (0.259 within string). Block-bootstrap SE 0.07. It falls to 0.356 controlling for token count and to 0.255 controlling for token count plus the shallow structural features.
- σ: the script doubles the SD; the rule's wording is "control variance … doubled" (SD × 1.41). The author's choice is the conservative one, and it takes the largest of three models.
- The control family differs from the one σ came from. I rebuilt the option-B family in the smaller models: natural SD 0.10 to 0.14 s against 0.10 to 0.15 for the measured class-mean-difference controls. The factor 2 covers the difference.
- σ in logits per s does not transfer cleanly between models (answer-logit SD within string: 0.28, 0.88, 2.60 in the Qwen models, 1.34 in 7B). R2′ is the scale-free alternative.

### F7 (moderate). Things the calculation leaves out, all of which lower power

- Dead site: 1 of 4 smaller end-of-hypothesis sites. The note says so; the table does not include it.
- The run uses one split (split 1). The script averages eight heavily dependent splits; split 1 alone gives the same range at share 1 and 0.5.
- The one-sided sign is assumed positive. No earlier 7B run exists to fix it.
- 4-bit, bfloat16 steering has never been run.

### F8 (minor, confirmed correct)

- Direction and control construction match `build()` in 07d exactly, including the control generator seed and the split generator.
- p-value and threshold match `one_sided_p` and `second_question`.
- Item noise is negligible: item SE of a control slope is 0.005 to 0.023 logits per s against a direction-to-direction SD of 0.10 to 0.38.
- One-class cells: 391 of 2,400 items (16%) pool-wide, 239 to 269 per half (21%), in 24 to 25 of 96 cells. Effective pairs per half 186 to 203 of a possible 288 to 312. This costs fitting precision only.
- 100 controls are enough: z for power 0.6 is 1.94 against about 1.90 with unlimited controls.

## Step 3: the note

- **Complete and fair?** A1 to A5 are stated honestly and the note does not hide that the headline is fragile. It omits F1 (length), F3 (the identity), F4 (control SD implied by its own model) and F5.
- **"Why I do not trust that answer":** right in direction, and understated.
  - Point 1 is right, but the positive control is a probe direction, so it is not a ceiling for a class-mean difference. The better evidence is the predicted-versus-measured comparison in F2.
  - "Either A1 or A2 fails": the measurements say A1.
  - Point 2 quotes 0.14 to 0.44 from two of the three shares. Using all four sites the mean is 0.20.
  - It does not say that a positive result would be uninterpretable (F1).
- **Which reading for the rule:** R2. The rule tells the author to take the control variance from the smaller models; the same models are the only evidence on the share.
- **Probability of p ≤ 0.05 given the 7B model does use such a representation at some end-of-hypothesis layer:** best single estimate **0.3, range 0.1 to 0.5**. This sits above R2's 0.2 because the condition makes larger shares more likely than in the smaller models, and because the length loading can produce a hit on its own.
- **If p ≤ 0.05:** it would show that adding this direction moves the answer more than 96% of its controls do. It would not separate validity from prompt length. I put the chance of p ≤ 0.05 with no label-specific effect at roughly 0.1 to 0.15, not 0.05; that figure is a judgement, since the 7B response to length at this site is unmeasured.
- **If p > 0.05:** almost nothing. A dead site, a small share, or an adverse length response each produce it with a real effect present.

## Step 4: would more items help?

No, not materially.

- **Items for fitting:** the fitted mean difference is already about 75% signal in squared norm (0.054 s against 0.027 s for shuffled labels). An unlimited pool raises the full-share slope by about 12%.
- **Items steered:** item noise is under 1% of the control variance.
- **Number of controls:** not limiting (F8).
- **What items cannot fix:** the share one layer at one token carries, the model's response to arbitrary directions (the control SD, which does not shrink with n because every direction is a unit vector at a fixed step), and whether the site responds.
- **What a new pool could fix:** the length confound, if prompts are length-balanced within (string, adjacency) cells, and the one-class cells. That is a change of design, not of size.

## What I could not check

- Any response of the 7B model to steering: control SD, sign and size of its length response, whether a site is dead, 4-bit behaviour.
- Which layer option B would select (no selection file for 7B; the gold probe is best at layers 16 to 21, accuracy 0.80).
- The tenth audit's calibration claim; I did not read that audit.
- The low-rank response fits I used to look at length effects in the smaller models are good at the final token (leave-one-out R² 0.90 to 0.96) and poor at the end of the hypothesis (−0.01 to 0.32). I draw no conclusion from the latter about the sign of a length response.

## Files

All in `/private/tmp/claude-501/-Users-SahilPaliwal-Documents-causal-probe/bf0bfd13-4d48-4df5-becf-71aee174b8a6/scratchpad/audit12/`:

- `s1_behaviour.py`, `s1_surface.py`, `s1_small_len.py`: 7B answers, cell structure, length and structural features
- `s1_geom7b.py`, `s1_length7b.py` (outputs `geom_qwen7b-4bit.json`, `length7b.json`): direction geometry and length content by layer
- `s1_small_table.py`, `s1_small_recon.py` (outputs `small_table.json`, `small_recon_*.json`): smaller-model slopes, rebuilt directions, predicted-versus-measured
- `s1_power.py`, `s3_mixture.py`, `s3_final_table.py` (output `power_table_mine.json`): power tables
- `s2_reproduce.py`, `reproduce18.json`: reproduction of script 18 and variants
