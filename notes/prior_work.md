# Prior-work check for `report/benchmark_text_baselines.md`

Checked on 13 Nov 2026. Scope: text-only / partial-input baselines, shortcut analyses and probing
studies on Corr2Cause (Jin et al., ICLR 2024) and CLadder (Jin et al., NeurIPS 2023), plus the
canonical method references.

## How to read this file

- Every entry below was opened in this session at the link given. Nothing is cited from memory.
- "Opened" means the page was fetched through an automated fetch tool that converts the page to
  text and answers questions about it with a small model. I did not read raw page text myself.
  Numbers below are therefore second-hand readings of the source and should be checked against
  the source before they are quoted. Where two fetches of the same page were made, I say so.
- For long arXiv HTML pages the tool reads the first ~100,000 characters per call. Where the tail
  was not read, the entry says so.
- PDFs could not be parsed (no PDF text tool on this machine; none was installed). Papers with no
  arXiv HTML version are therefore "abstract only".
- OpenReview and Google Scholar could not be opened (browser-verification / redirect pages).
- Status tags: `opened: abstract only` or `opened: full text`.
- Wording is paraphrased; no passage is quoted verbatim.

## Headline: does arXiv:2610.38100 check out?

**Yes.** https://arxiv.org/abs/2610.38100 exists. Title: "Evaluating Causal Reasoning in LLMs: A
Case Study in the Challenges of Procedural Generation". Authors as listed on the abstract page:
Justin Clarke, Isha Karn, Patrick Murray, KJ Hill, David Jensen. Submitted 28 Oct 2026 (v1). No
venue listed; it is a preprint. The full HTML (v1) was fetched three times with different
questions and the answers agreed on the following:

- Random forest on hand-built surface features: 86.8 ± 0.7 overall (Table 3, App. A.4); Sec. 3.2
  states it as 87%. The Figure 3 caption says mean accuracy over 10-fold cross-validation.
- Per type, RF row of Table 3: collider bias 100.0 ± 0.4 (n = 432), backdoor adjustment
  62.2 ± 10.8 (n = 74), det-counterfactual 48.6; ATE 96.8, marginal 93.8, correlation 90.4,
  exp-away 92.2, NDE 81.2, NIE 81.5, ETT 99.0.
- Data: 9,866 unique question–answer pairs regenerated with the CLadder code (App. A.3), not the
  released Hugging Face file. The paper does not mention `full_v1.5_default.csv` or 10,112 items.
- The paper does not say how folds were built (random or grouped by story). No story-held-out
  split is described.
- Features are listed in App. A.8 (probability values, pairwise comparisons and products, about
  30 keyword flags, query length, two query-opening flags). No feature importances.
- The account of generator rules is in App. A.7 (parameter signs fixed per story, the collider
  graph forcing some answers, collider-bias answer set by polarity alone). The fetch found no
  statement about how backdoor-adjustment answers are determined.
- Also reported: RF + BERT embeddings 76.2 ± 0.8; fine-tuned LLMs above 94% with 16,000 training
  examples (Sec. 3.2, App. A.5); removing background/graph text changes frontier-model accuracy by
  0.6–4.2 points (Sec. 3.4); RF reaches 79.1 ± 2.3 on CounterBench (Table 4, App. A.9).
- No linear probing of activations. Corr2Cause is not mentioned.

Two small points for the note's wording, to check by hand:

1. The note cites "Sec. 3.2 and App. A.4, Table 3" for both the numbers and the account of
   generator rules. The numbers are there; the generator-rule account is App. A.7 and the feature
   list App. A.8.
2. The note says ten-fold cross-validation "on a freshly generated 9,866-item set". App. A.3 says
   the 9,866 pairs were used for all experiments, and the Figure 3 caption gives the 10-fold
   protocol; the fetch found no single sentence in Sec. 3.2 joining the two. The fetch also
   flagged that Sec. 3.2 describes the fine-tuning split as coming from the public CLadder
   dataset while App. A.5 describes the generated set as the test set. Worth a direct read.

## 1. Papers and public reports

### 1a. The two benchmark papers

| Paper | Link fetched | What it reports that matters here | Status |
|---|---|---|---|
| Jin, Liu, Lyu, Poff, Sachan, Mihalcea, Diab, Schölkopf. "Can Large Language Models Infer Causation from Correlation?" arXiv v3 17 Apr 2024; arXiv comment: ICLR 2024 | https://arxiv.org/abs/2306.05836 and https://arxiv.org/html/2306.05836v3 | Table 3: 207,972 items, 18.57% positive; 205,734 / 1,076 / 1,162. Table 4: best off-the-shelf F1 33.38 (BART MNLI); always-majority F1 0.0, accuracy 84.77. Table 5a: RoBERTa-Large MNLI 94.74 fine-tuned. Sec. 4.5 / Table 5b: drops of up to 39.29 (paraphrase) and up to 62.30 (variable refactorization, i.e. reversed alphabet). Sec. 3.3: nodes in topological order, edges only from lower to higher index. Sec. 3.7 footnote and App. D: v2.0 is de-duplicated and half the size; authors expect little change in results. App. E, Table 11: PMI between n-grams and labels; some cues come from hypothesis framing, some from variable names. No hypothesis-only, premise-only or other partial-input baseline; the words "shortcut" and "artifact" do not occur ("spurious" does, App. E). No premise counts per split. | opened: full text |
| Jin, Chen, Leeb, Gresele, Kamal, Lyu, Blin, Gonzalez Adauto, Kleiman-Weiner, Sachan, Schölkopf. "CLadder: Assessing Causal Reasoning in Language Models". arXiv v3 17 Jan 2024; arXiv comment: NeurIPS 2023, updated with dataset v1.5 | https://arxiv.org/abs/2312.04350 and https://arxiv.org/html/2312.04350v3 | Table 1: 10,112 items (3,160 / 3,160 / 3,792 by rung), 50% positive in every column. Sec. 3.3 / App. A.4: roughly balanced over query types, graphs, stories and answers. Table 2: only a random baseline (49.27) besides LLMs; no text-only classifier; no fine-tuning. App. A.2: collider-bias answer is always "no" by construction. App. A.4: "balanced" set is the v1.5 main benchmark (10,112); "aggregate" is the v1.0 one (10,560). No analysis of predictability from surface features in the part read. Read up to App. A.5 only (page tail past ~100k characters not read; total story count not seen). | opened: full text |

### 1b. Corr2Cause: follow-ups, critiques, re-evaluations

| Paper | Link fetched | What it reports that matters here | Status |
|---|---|---|---|
| Sgouritsa, Aglietti, Teh, Doucet, Gretton, Chiappa. "Prompting Strategies for Enabling Large Language Models to Infer Causation from Correlation". arXiv, 18 Dec 2024 (no venue on page) | https://arxiv.org/html/2412.13952 | Evaluates on the v1.0 test partition (2,246 items; footnote 2). Sec. 5: 15.23% of the test set is positive; F1 is the main metric; always-majority has high accuracy and zero F1. Sec. 5 / Fig. 6a: per-hypothesis-group breakdown of LLM performance. PaLM 2 F1 0.30 (few-shot CoT) to 0.64 (PC-SubQ). No hypothesis-only baseline; no label-cue discussion; no label issues flagged. Tail of page not read. | opened: full text |
| Bagheri, Alinejad, Dehshiri, Bello, Akhondi-Asl. "C²P: Featuring Large Language Models with Causal Reasoning". arXiv v4, 7 Nov 2025 (no venue on page) | https://arxiv.org/html/2407.18069v4 | Sec. 4.2 states that most Corr2Cause answers are "No" (81.5%) and most items have six variables (96.06%), and for that reason evaluates on a hand-balanced sample: 60 Yes / 60 No with equal counts per number of variables (120 items), plus a 300-item five-variable set. Balances labels and variable count, not hypothesis strings or premises. No hypothesis-only baseline; no variable-name cue analysis. | opened: full text |
| Sun, Nogueira, Silva. "Structured Thinking Matters: Improving LLMs Generalization in Causal Inference Tasks". arXiv v2, 27 May 2025 | https://arxiv.org/html/2505.18034v2 | Sec. 4: about 80% No / 20% Yes, so F1(Yes) is emphasised. Related work: fine-tuned RoBERTa-Large reaches about 95% F1 in-distribution and falls under paraphrase / renaming; attributed in prose to superficial cues, with no experiment isolating them. Qwen3-32B F1 32.71 to 48.26. No partial-input baseline. | opened: full text |
| Kadziolka, Salehkaleybar. "Causal Reasoning in Pieces: Modular In-Context Learning for Causal Discovery". arXiv, 31 Jul 2025 | https://arxiv.org/html/2507.23488 | Uses the full 1,162-item test split (App. B); about 85% negative (Sec. 4.1). Sec. 2.2 restates that edges are generated only for i < j but does not treat this as a label cue. Table 2: o3-mini F1 66.28 zero-shot, 83.83 with the modular pipeline. App. E: bootstrap interval for one baseline (the fetch did not say whether resampling is by item or by premise). No partial-input baseline; no label issues flagged. | opened: full text |
| Roy, Parbhoo. "Why LLMs Fail at Causal Discovery and How Interventional Agents Escape". arXiv, 26 May 2026 | https://arxiv.org/html/2605.27567v1 | Theoretical argument plus an agentic method; baselines include RoBERTa-Large SFT and DPO. Repeats the in-distribution-high / perturbed-low pattern from the original paper without analysing label priors, templates or variable names. Introduces "Extended Corr2Cause" (7–24 variables). No partial-input baseline, no probing. | opened: full text |
| Yang, Shirvaikar, Clivio, Falck. "A Critical Review of Causal Reasoning Benchmarks for Large Language Models". arXiv, 10 Jul 2024; arXiv comment: AAAI 2024 workshop | https://arxiv.org/html/2407.08029 | Sec. 2.2 only: remarks that Corr2Cause uses letters as variables, far from real-world interpretation; that CLadder may allow use of pre-existing knowledge of a causal direction. No spurious-cue or shortcut analysis of either dataset; no baselines. | opened: full text |
| Sun, Nogueira, Verchere, Acher, Silva. "Externalized CPDAG Summaries Improve LLM Causal Deduction". arXiv, 25 Sep 2026; arXiv comment: accepted at NeurIPS 2026 | https://arxiv.org/abs/2609.31071 | Abstract: on the full Corr2Cause test set, F1(Yes) 73.0 to 86.4 for Qwen3.5-27B; scrambling the emitted graph costs 12.0 points. Body not read (no HTML version; PDF not parseable), so whether it discusses label imbalance or partial-input baselines is unknown. | opened: abstract only |
| Vashishtha, Kumar, Pandey, Reddy, Ahuja, Balasubramanian, Sharma. "Teaching Transformers Causal Reasoning through Axiomatic Training". arXiv v3, 24 Oct 2025 | https://arxiv.org/abs/2407.07612 | Abstract: fine-tuning Llama-3-8B-Instruct on axiomatic data improves Corr2Cause and CLEAR scores. No numbers or shortcut discussion in the abstract. | opened: abstract only |

### 1c. CLadder: shortcut / surface-feature analyses, critiques, re-evaluations

| Paper | Link fetched | What it reports that matters here | Status |
|---|---|---|---|
| Clarke, Karn, Murray, Hill, Jensen. "Evaluating Causal Reasoning in LLMs: A Case Study in the Challenges of Procedural Generation". arXiv, 28 Oct 2026 (preprint) | https://arxiv.org/abs/2610.38100 and https://arxiv.org/html/2610.38100v1 | See "Headline" above. The closest prior result to the note's CLadder section. Sec. 2.2 cites Pacchiardi et al. and describes them as concluding that CLadder test sets are among the least susceptible to generator-specific shortcuts. | opened: full text |
| Pacchiardi, Tesic, Cheke, Hernández-Orallo. "Leaving the barn door open for Clever Hans: Simple features predict LLM benchmark answers". arXiv, 15 Oct 2024 (no venue on page) | https://arxiv.org/abs/2410.11672 and https://arxiv.org/html/2410.11672 | CLadder is one of the benchmarks (Table 1: 8,917 instances, 2 choices). Classifier: L1 logistic regression on word-level 2-gram term frequencies (Table 2); train/validation/test split (Sec. 3.4); metric is Cohen's kappa on test. The CLadder value is shown only in the Figure 1 heatmap and was not read. Uses n-grams only, not the quoted numbers. | opened: full text |
| Yu, Zhou. "Caliper: Probing Lexical Anchors versus Causal Structure in LLMs". arXiv, 3 Jun 2026 | https://arxiv.org/abs/2606.04915 and https://arxiv.org/html/2606.04915v1 | Behavioural, not activation probing. Replaces variable names with placeholders on 1,500 commonsensical CLadder items (500 per rung; Sec. 4.1), plus CRASS and e-CARE. CLadder interventional accuracy drops by 7.6 points on local models (Tables 1, 4). No text-only classifier; no claim that answers are predictable from numbers. | opened: full text |
| Xun. "Evidence-Type Competition: When Can Interventional Data Teach Language Models Causal Direction?". arXiv, 31 Jul 2026 | https://arxiv.org/abs/2607.29484 and https://arxiv.org/html/2607.29484v1 | Small transformers trained from scratch on symbolic worlds. Sec. 5.7–5.8: transplants the structure of CLadder v1 SCMs (not the natural-language questions) into its own format and finds a learned positive-effect offset. App. H: activation patching on a 25.7M-parameter model. No linear probes, no text-only baseline. | opened: full text |
| Chen, Singh, Ma, Tang. "CounterBench: Evaluating and Improving Counterfactual Reasoning in Large Language Models". arXiv v2, 11 Apr 2026 | https://arxiv.org/abs/2502.11008 | Abstract: 1K formal counterfactual questions; most models near chance. Relevant because Clarke et al. and Ding & Zhang use it as a second benchmark. | opened: abstract only |
| Vamosi, Forkert. "CRAwDAD: Causal Reasoning Augmentation with Dual-Agent Debate". arXiv v2, 9 Mar 2026 | https://arxiv.org/abs/2511.22854 | Abstract: CLadder accuracy 78.03 to 87.45 (DeepSeek-R1) and 84.16 to 89.41 (Qwen3) with debate. Shows current zero-shot levels; no shortcut analysis in the abstract. | opened: abstract only |

### 1d. Probing and mechanistic studies (these benchmarks or closely related tasks)

| Paper | Link fetched | What it reports that matters here | Status |
|---|---|---|---|
| Ding, Zhang. "Causal Tongue-Tie: LLMs Can Encode Causal Direction, But Their Yes/No Outputs Fail to Express". arXiv 25 May 2026; EMNLP 2026 main, pp. 4976–4998 | https://arxiv.org/html/2605.25891v1 (both halves) and https://aclanthology.org/2026.emnlp-main.217/ | L2 logistic-regression probes on last-prompt-token states, trained on 160 commonsense CLadder rows, tested on 80 anti-commonsense items. Probe accuracy at least 0.969 on every model against Yes/No output accuracy 0.350–0.525 (Sec. 5.1). The probed label is causal direction in paired prompts, not the CLadder gold answer. Controls: item-disjoint CV with pairs kept together; Hewitt–Liang-style control task (selectivity 0.302–0.361; App. D, Table 8); logit lens; sham and random directions; activation patching on 42 wrong items (to 0.571). No bag-of-words, TF-IDF or other text-only classifier for the probe label was found in either half. Limitations section says the direction label is partly confounded with question format. Corr2Cause not used. The arXiv v1 and the Anthology abstract list slightly different model sets; only the arXiv text was read in full. | opened: full text |
| Yu. "Vernier: Probing Representational Misalignment Behind Lexical Gaps in Causal Reasoning". arXiv, 14 Jun 2026 | https://arxiv.org/abs/2606.15733 and https://arxiv.org/html/2606.15733 | CLadder is the in-distribution benchmark (items 0–1499 for training, 1500–1699 held out; Sec. 4.1); CRASS and e-CARE out of distribution. Probes (Sec. 4.4) predict the variable-name family (12 classes), not the gold answer; activation patching with a random-donor control. No text-only baseline for probes. Corr2Cause not mentioned in the part read (tail past ~100k characters not read). | opened: full text |
| Sahoo, Jain, Chadha, Chaudhary. "Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States". arXiv v2, 3 Jun 2026; arXiv comment: TrustNLP workshop at ACL 2026 | https://arxiv.org/abs/2606.02907 | Abstract: probes for reasoning type on Qwen3-14B reach 100% cross-validated accuracy at layer 32 of 40; after residualising source, option count and response length, accuracy is at chance. Datasets are LogiQA 2.0, ARC-Challenge and αNLI, not the two benchmarks here. Closest in spirit to the note's point about probes. | opened: abstract only |
| Butkus, Kriegeskorte. "Causal Discovery and Inference through Next-Token Prediction". NeurIPS 2025 | https://papers.neurips.cc/paper_files/paper/2025/hash/6291ca1d98c37d12db760ec0c1f6cb25-Abstract-Conference.html | Abstract: a GPT-style model trained on text from linear-Gaussian SCMs; an implicit SCM is decoded from residual-stream activations and manipulated with predictable output changes. Synthetic task, model trained from scratch. | opened: abstract only |
| Zhong, He, Li, Mesgarani. "LLMs Think with DAGs: Probing the Latent Graph Structure of Reasoning" (earlier title "From Chains to DAGs: Probing the Graph Structure of Reasoning in LLMs"). arXiv v4, 15 Oct 2026 | https://arxiv.org/abs/2601.17593 | Abstract: probes for node depth, pairwise distance and adjacency of a reasoning DAG on logic / math / code benchmarks. Not causal-inference benchmarks. Controls not described in the abstract. | opened: abstract only |
| Xun. "Located but Not Releasable: Silent Gate Inversion and Bounded Linear Release". arXiv, 12 Aug 2026 | https://arxiv.org/abs/2608.11822 | Abstract: 25.7M-parameter transformer trained on a causal-evidence discrimination task; localisation by intervention works, gated and linear release do not reach preregistered margins. Custom task, not these benchmarks. | opened: abstract only |

### 1e. Method references: partial-input baselines, probe controls, balancing by construction

| Paper | Link fetched | Relevance | Status |
|---|---|---|---|
| Poliak, Naradowsky, Haldar, Rudinger, Van Durme. "Hypothesis Only Baselines in Natural Language Inference". *SEM 2018, pp. 180–191 | https://aclanthology.org/S18-2023/ | Hypothesis-only model beats majority class on many of 10 NLI datasets. | opened: abstract only |
| Gururangan, Swayamdipta, Levy, Schwartz, Bowman, Smith. "Annotation Artifacts in Natural Language Inference Data". NAACL-HLT 2018 (short papers), pp. 107–112 | https://aclanthology.org/N18-2017/ | Hypothesis-only classifier: about 67% on SNLI, about 53% on MultiNLI. | opened: abstract only |
| Kaushik, Lipton. "How Much Reading Does Reading Comprehension Require? A Critical Investigation of Popular Benchmarks". EMNLP 2018, pp. 5010–5015 | https://aclanthology.org/D18-1546/ | Question-only and passage-only baselines on five reading-comprehension datasets. | opened: abstract only |
| McCoy, Pavlick, Linzen. "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference". ACL 2019, pp. 3428–3448 | https://aclanthology.org/P19-1334/ | HANS; shallow heuristics learned from MNLI. | opened: abstract only |
| Tsuchiya. "Performance Impact Caused by Hidden Bias of Training Data for Recognizing Textual Entailment". LREC 2018 | https://aclanthology.org/L18-1239/ | Title, author and venue confirmed; the page showed no abstract and no page numbers, so the content is not verified. | opened: abstract only |
| Hewitt, Liang. "Designing and Interpreting Probes with Control Tasks". EMNLP-IJCNLP 2019, pp. 2733–2743 | https://aclanthology.org/D19-1275/ | Control tasks and selectivity for probes. | opened: abstract only |
| Belinkov. "Probing Classifiers: Promises, Shortcomings, and Advances". Computational Linguistics 48(1), 2022, pp. 207–219 | https://aclanthology.org/2022.cl-1.7/ | Review of probing-classifier methodology and its limits. Optional. | opened: abstract only |
| Goyal, Khot, Summers-Stay, Batra, Parikh. "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering". arXiv v3, 15 May 2017 (venue not shown on the arXiv page) | https://arxiv.org/abs/1612.00837 | Precedent for removing a single-input prior by construction: each question is paired with two similar images that have different answers. Analogous to the note's debiased pool. Optional. | opened: abstract only |
| Le Bras, Swayamdipta, Bhagavatula, Zellers, Peters, Sabharwal, Choi. "Adversarial Filters of Dataset Biases". arXiv v3, 11 Jul 2020; arXiv comment: ICML 2020 | https://arxiv.org/abs/2002.04108 | AFLite; filtering instead of pairing. Optional. | opened: abstract only |

### 1f. "Represents the answer but outputs another": canonical references

| Paper | Link fetched | Relevance | Status |
|---|---|---|---|
| Burns, Ye, Klein, Steinhardt. "Discovering Latent Knowledge in Language Models Without Supervision". arXiv v2, 2 Mar 2024; arXiv comment: ICLR 2023 | https://arxiv.org/abs/2212.03827 | Contrast-consistent search; beats zero-shot by 4% on average over 6 models and 10 datasets. | opened: abstract only |
| Azaria, Mitchell. "The Internal State of an LLM Knows When It's Lying". arXiv v2, 17 Oct 2023 (no venue on the arXiv page) | https://arxiv.org/abs/2304.13734 | Classifier on hidden activations, 71%–83% on balanced true/false sets. | opened: abstract only |
| Marks, Tegmark. "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets". arXiv v3, 19 Aug 2024; arXiv comment: Conference on Language Modeling 2024 | https://arxiv.org/abs/2310.06824 | Linear truth directions; probe transfer and interventions. | opened: abstract only |
| Orgad, Toker, Gekhman, Reichart, Szpektor, Kotek, Belinkov. "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations". arXiv v4, 18 May 2025 (no venue on the arXiv page) | https://arxiv.org/abs/2410.02707 | Cases where the correct answer is encoded internally and a wrong one is output; error detectors do not generalise across datasets. | opened: abstract only |
| Farquhar, Varma, Kenton, Gasteiger, Mikulik, Shah. "Challenges with unsupervised LLM knowledge discovery". arXiv v2, 18 Dec 2023 | https://arxiv.org/abs/2312.10029 | Unsupervised probes can latch on to the most prominent feature in the activations instead of knowledge; proposes sanity checks. Relevant caution for the note's probing argument. | opened: abstract only |

### 1g. Public reports that are not papers (unreviewed)

| Source | Link fetched | What it says | Status |
|---|---|---|---|
| Course-project README, repository `abhijeetnst/dl-aml` (titled as a DL/AML project) | https://github.com/abhijeetnst/dl-aml and the raw README | States that 3 validation premises also occur in test and 2 in train; states that the reversed "directly causes" template is never positive because causes are ordered alphabetically, and calls this a shortcut; builds a renamed-variable test set. Reports a "pattern features, then logistic regression" baseline with F1 0.617 on test and 0.279 on the renamed test (n = 1,162 each), and fine-tuned DistilBERT at 0.605 / 0.264. The README does not define the pattern features and labels the metric as triple extraction, so it is not clear that this is valid/invalid F1. No hypothesis-only or premise-only experiment. No date visible on the page. | opened: full text |
| Hugging Face discussion #6 on `causal-nlp/corr2cause` | https://huggingface.co/datasets/causal-nlp/corr2cause/discussions/6 | Reports 6,943 labels (about 3.3% of 207,972) as incorrect because of a generator bug: train 6,897, validation 28, test 18. Open, no maintainer reply visible. Page shows "April 20" without a year. Does not mention hypothesis-only predictability, the child template or premise overlap. | opened: full text |
| GitHub `causalNLP/corr2cause` issues #3, #5, #9 | https://github.com/causalNLP/corr2cause/issues/5 (also /3, /9) | A comment dated 5 Sep 2026 on #5 reports that an independent re-derivation reproduces 1,144 of 1,162 test labels, that 6,958 of 205,734 train items do not reproduce, and that the `child` relation has no positive instances; the commenter says an audit paper is being written. Comments on #3 and #9 concern label encoding of the perturbation files and reproducibility. None mentions hypothesis-only predictability or split overlap. | opened: full text |
| GitHub `causalNLP/cladder` issues #15, #16 | https://github.com/causalNLP/cladder/issues/15 and the issue list | #15 (20 Sep 2026): ground truth appears to multiply parent marginals; 83 of 10,560 questions affected by the reporter's count; unanswered. #16 seen by title only. No issue about surface-feature predictability. | opened: full text |

### 1h. Opened and judged not directly relevant (so they need not be re-checked)

All `opened: abstract only`. None names a partial-input baseline or a probing study on either
benchmark in its abstract.

- Chen, Chen, Lu, "Can Post-Training Transform LLMs into Causal Reasoners?", https://arxiv.org/abs/2602.06337 (CauGym / CaLM).
- Lu et al., "Generalization of RLVR Using Causal Reasoning as a Testbed", https://arxiv.org/abs/2512.20760 (own dataset).
- Srivastava et al., "Realizing LLMs' Causal Potential Requires Science-Grounded, Novel Benchmarks", https://arxiv.org/abs/2510.16530 (BNLearn memorisation).
- Chi et al., "Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?", https://arxiv.org/abs/2506.21215 (CausalProbe-2024).
- Xun, "Causal Structure is Inducible but Functionally Decoupled", https://arxiv.org/abs/2608.11767.
- Zhang et al., "Probing Causality Manipulation of Large Language Models", https://arxiv.org/abs/2408.14380 (behavioural).
- Wang, Zhu, Li, "CausalFlip: A Benchmark for LLM Causal Judgment Beyond Semantic Matching", https://arxiv.org/abs/2602.20094 (pairs similar questions with opposite answers; a design precedent for paired balancing, different data).
- Behnam, Wang, "Reasoning Models Are Accurate but Unsound on Identification", https://arxiv.org/abs/2610.03519.
- Vaidya et al., "When Many Answers Are Valid, Voting Fails", https://arxiv.org/abs/2608.03506 (CLEAR).
- Dominici et al., "Causal Chain of Thought: Mitigating Attention Leakage via Context Isolation", https://arxiv.org/abs/2610.31639.
- Chang, Geng, "Epistemic Regret Minimization", https://arxiv.org/abs/2602.11675.
- Wang et al., "Do LLMs Have the Generalization Ability in Conducting Causal Inference?", https://arxiv.org/abs/2410.11385.
- Xu, Fu, "NoisyCausal", https://arxiv.org/abs/2605.04313 (transfers a method to CLadder).
- Gui, Ji, "Mitigating Spurious Correlations in LLMs via Causality-Aware Post-Training", https://arxiv.org/abs/2506.09433 (fine-tunes and evaluates on CLadder; body not read).
- Mondal et al., "Implicit Probabilistic Reasoning Does Not Reflect Explicit Answers in Large Language Models", https://arxiv.org/abs/2406.14986.
- Xu et al., "RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation", https://arxiv.org/abs/2506.15455.
- Singh, "Measuring and Mitigating Shortcut Reliance in Language Models with Probe-Based Representation Entanglement", ACL 2026 SRW, https://aclanthology.org/2026.acl-srw.59/ (SST-2, MNLI).
- Patel et al., "Stress-Testing the Robustness of LLM-Based Causal Inference", https://arxiv.org/abs/2611.03656 (the fetch reported a submission date that does not match the identifier's month; abstract names no benchmark).

## 2. What is and is not new in our note

### 2.1 Corr2Cause hypothesis-only result (test AUROC about 0.895; 72 of 120 strings never valid)

**Not found in the literature after these searches.**

- No paper opened here reports a hypothesis-only, premise-only or other partial-input baseline on
  Corr2Cause. Checked in full text: the original paper, Sgouritsa et al., C²P, Sun et al. 2025,
  Kadziolka & Salehkaleybar, Roy & Parbhoo, Yang et al. Checked by abstract only: Sun et al. 2026,
  Vashishtha et al.
- Searches that returned nothing on this: "Corr2Cause" with "hypothesis-only" / "hypothesis only"
  / "premise-only" / "partial input"; Corr2Cause with shortcut, spurious, artifact, leakage;
  Corr2Cause with alphabetical / variable order / topological order; Corr2Cause with "never
  valid" / "always invalid" / label distribution; the arXiv listing of all papers mentioning
  Corr2Cause in title or abstract (8 results, all listed above or in section 3); a title scan of
  248 citing papers from Semantic Scholar.
- Material that implies part of it without stating it:
  - Original paper, Sec. 3.3 (edges only from lower to higher index) and App. D / Sec. 3.7
    footnote (v2.0 de-duplication). The derivation in the note that a directed claim can be valid
    only from an earlier to a later letter rests on these plus the public code; I did not
    re-derive it or read the code.
  - Original paper, App. E, Table 11: some label-correlated n-grams are tied to variable names;
    Sec. 4.5 refactorization is designed around such cues.
  - Unreviewed: the `dl-aml` README states the reversed "directly causes" template is never
    positive and calls it a shortcut; a GitHub comment on corr2cause #5 states `child` has no
    positive instances. Neither measures a hypothesis-only classifier.
- Not found anywhere: the count of 120 distinct strings, the 72 / 0 / 48 breakdown, or any
  AUROC / accuracy / F1 for a hypothesis-only predictor.

### 2.2 Corr2Cause parsed-feature result (test accuracy about 0.938)

**Not found in the literature after these searches**, with one unreviewed near neighbour.

- Published and related: fine-tuned full-input models reach F1 94.74 in-distribution (original
  paper, Table 5a) and fall under renaming and paraphrase (Table 5b). That shows the task is
  learnable in-distribution by a model that sees everything; it does not say shallow parsed
  features suffice.
- Unreviewed near neighbour: the `dl-aml` README reports a pattern-feature logistic regression at
  F1 0.617 on test (0.279 on a renamed test). The features and the metric are not defined clearly
  enough there to compare with the note's 0.802 valid-class F1.
- Searches that returned nothing: Corr2Cause with "logistic regression" / "random forest" /
  heuristic baseline; Corr2Cause with fine-tuning and shortcut; the listings above.

### 2.3 Premise counts per split and premise overlap between splits

**Not found in any paper after these searches; the overlap counts are stated in one unreviewed
README.**

- The original paper gives item counts per split and per number of variables, not premise counts
  (confirmed in the full-text fetch).
- The `dl-aml` README gives the same overlap counts as the note (3 validation–test, 2
  validation–train). The note already cites it.
- No follow-up paper opened here mentions premise-level overlap or resamples by premise. Kadziolka
  & Salehkaleybar report a bootstrap interval for one baseline (App. E); whether it resamples
  items or premises was not determined.
- Label imbalance itself is already published: original paper (18.57% positive overall), Sgouritsa
  et al. (15.23% of the v1.0 test set), C²P (81.5% No; 96.06% six-variable), Sun et al. 2025
  (about 80 / 20), Kadziolka & Salehkaleybar (about 85% negative).
- The earlier check's statement that 2,338 + 22 + 21 minus overlaps equals the 2,376 equivalence
  classes of Table 1 was not re-verified here (I confirmed 6,325 DAGs and 5,984 for N = 6, not
  the class count).

### 2.4 CLadder numbers + polarity result (about 0.835, story-held-out)

**Already published in substance: Clarke et al. (arXiv:2610.38100).**

- Clarke et al. report 86.8 ± 0.7 for a random forest on quoted probabilities, comparisons
  between them and keyword flags, with a per-type table and an account of the generator rules.
- CLadder App. A.2 already states the collider-bias answer is fixed.
- Pacchiardi et al. included CLadder in an n-gram predictability study (2-gram logistic
  regression); the CLadder value was not read, and Clarke et al. describe that study as finding
  CLadder among the least susceptible. This is consistent with the note's chance-level
  bag-of-words rows, and is a second citation the note could carry for that row.
- What these searches did not find elsewhere, i.e. what remains specific to the note:
  - an estimate on the released `full_v1.5_default.csv` (Clarke et al. regenerate 9,866 items);
  - story-held-out folds (Clarke et al. do not state fold construction);
  - `backadj` at 1.00 with a feature for which method conditions case by case (Clarke et al.:
    62.2 ± 10.8, n = 74; their appendix on determinism does not cover backdoor adjustment, per the
    fetch). Searches for CLadder with "backdoor adjustment" and "case by case" / "method 1" /
    shortcut returned only dataset viewer pages.
- Note the collider-bias count differs between sources: 158 items in the released file per the
  note, 432 in Clarke et al.'s regenerated set.

### 2.5 The debiased Corr2Cause pool

**Not found in the literature after these searches.**

- Nearest published item: C²P (Sec. 4.2) evaluates on a hand-balanced sample (60 Yes / 60 No,
  equal per variable count). It balances labels and variable counts only.
- The original paper's variable-refactorization set addresses name cues by renaming, not by
  resampling.
- Method precedents from other tasks (not Corr2Cause): Goyal et al. (paired images with different
  answers), CausalFlip (paired questions with opposite answers), Le Bras et al. (adversarial
  filtering).
- Searches that returned nothing: Corr2Cause with balanced / debiased / resampled / re-split;
  Corr2Cause with premise-level re-split.

### 2.6 Probes compared against a text-only baseline on these benchmarks (the note's motivation)

**Not found in the literature after these searches.**

- Two probing studies use CLadder: Ding & Zhang (probe label: causal direction; Hewitt–Liang
  control task; no text-only classifier; states a format confound as a limitation) and Yu's
  Vernier (probe label: variable-name family; no text-only classifier).
- No probing, activation-patching, steering or SAE study on Corr2Cause was found. Searches:
  Corr2Cause with probe / probing / activation patching / sparse autoencoder / steering;
  conditional-independence statements with attention heads / circuit / probing; the listings
  above.
- Sahoo et al. make the same kind of argument (probe accuracy explained by format features) on
  other datasets.

## 3. Differences from `notes/literature_check.md` found while re-verifying

- **arXiv:2610.38100**: confirmed (see Headline).
- **Pacchiardi et al. (2410.11672)**: confirmed that CLadder is included. Added detail: 2-gram
  logistic regression, kappa metric, and Clarke et al.'s description of its conclusion.
- **Yu 2026 (2606.15733)**: listed in the earlier check under "probing on CLadder". Confirmed
  that it uses CLadder, but its probes predict the variable-name family, not the answer. The
  arXiv abstract names only CRASS and e-CARE; CLadder appears in the body.
- **Ding & Zhang (2605.25891)**: confirmed; now also in the EMNLP 2026 proceedings. The probe
  target is causal direction on 80 anti-commonsense items, which is narrower than "the answer".
- **Sun et al. 2026 (2609.31071)**: exists; the claim that it notes the label imbalance could not
  be verified (abstract only). The same first author's 2025 paper (2505.18034) does state it.
- **Sgouritsa et al.**: confirmed that it notes the imbalance; it evaluates on v1.0 (2,246 test
  items), not the v2.0 files the note uses.
- **Hugging Face discussion #6**: figures confirmed; the year is not shown on the page.
- **GitHub comment that `child` has no positive instances**: confirmed; it is on corr2cause issue
  #5 (5 Sep 2026).
- **Wording to re-read in the note**: the note says the Corr2Cause paper "notes that its
  experiments were not rerun" on v2.0. The fetch of v3 found that the paper expects minimal
  change and suggests future work may re-train, and found no explicit statement either way about
  rerunning. Please read the Sec. 3.7 footnote and App. D directly.

## 4. Leads not verified

- **OpenReview reviews** for Corr2Cause (forum id vqIH0ObdqL) and for CLadder: blocked by a
  browser check, including the API endpoint. Reviewer discussion of spurious cues (which may be
  the origin of the paper's App. E) is unread.
- **Google Scholar "cited by"** pages for both papers: redirect, not opened.
- **Semantic Scholar citation lists**: returned 248 citing papers for Corr2Cause and 213 for
  CLadder (titles only, scanned by eye). The lists are incomplete (for example, Ding & Zhang
  cites CLadder but is absent). The keyword-search endpoint was rate-limited and returned nothing.
- **Sun et al. 2026 (2609.31071)**, body: not read. Possible dataset statistics or baselines
  there.
- **Pacchiardi et al.**, Figure 1: the CLadder kappa value is in a heatmap and was not read.
- **CLadder paper**, appendices after A.5 (including App. D with v1.0 results): not read.
- **Sgouritsa et al.** and **Vernier**: last part of each HTML page not read.
- **An audit paper on Corr2Cause labels** announced in a GitHub comment (issue #5, Sep 2026): no
  such paper found on arXiv by search; the commenter's repository was not opened.
- **Kadziolka master's thesis** (theses.liacs.nl, PDF) behind 2507.23488: seen in search results,
  not opened.
- **Jin, "Causality for Natural Language Processing"** (arXiv:2504.14530, thesis): seen in search
  results, not opened; may discuss the benchmarks' limits.
- **"When Should LLMs Trust Their Own Revisions?"** (arXiv:2609.35832): appears in the arXiv
  listing for "Corr2Cause"; not opened.
- **Butkus & Kriegeskorte**: a search snippet says the decoders are linear and MLP probes; the
  abstract page does not, and the paper body was not read.
- **Search-snippet claim** that a summary site describes alphabetical variable order as a pattern
  fine-tuned Corr2Cause models exploit (lacuna.tiptreesystems.com): secondary page, not opened,
  not found in the original paper.
- **Tsuchiya (LREC 2018)**: bibliographic data confirmed, abstract not shown on the page.
- **Zečević et al., "Causal Parrots"** (arXiv:2308.13067, TMLR 2023 per search snippet): seen in
  search results only, not opened.
- **Sahoo et al.**, **Zhong et al.**, **CRAwDAD**, **CounterBench**, **Gui & Ji**: abstracts
  only; bodies may contain controls or dataset remarks not visible in the abstract.

## 5. Searches run

Web search (several in extended mode):

1. Corr2Cause "hypothesis-only" baseline shortcut "Can Large Language Models Infer Causation from Correlation"
2. CLadder benchmark shortcut dataset artifacts surface features classifier predicts answers critique
3. Corr2Cause benchmark spurious correlations label leakage variable names fine-tuned models overfit re-evaluation 2025
4. linear probe activations Corr2Cause OR CLadder causal reasoning LLM hidden states probing mechanistic interpretability
5. "Corr2Cause" "premise" "hypothesis" shortcut OR "spurious" OR "artifact" OR "leakage" fine-tuning generalization analysis
6. Corr2Cause dataset "label" bias "alphabetical" OR "variable order" OR "topological order" hypothesis valid only classifier without premise
7. "Corr2Cause" probe OR probing OR "activation patching" OR "sparse autoencoder" OR steering hidden representations causal discovery language model
8. "CLadder" "random forest" OR "logistic regression" OR "bag-of-words" OR "n-gram" predict answer without causal reasoning surface patterns
9. mechanistic interpretability causal graph representation LLM linear probe d-separation OR "causal structure" synthetic task transformers internal representation 2025 2026
10. Corr2Cause fine-tuning "in-distribution" shortcut "hypothesis" template "never valid" OR "always invalid" OR "label distribution" analysis re-split premise-level
11. "Corr2Cause" OR "CLadder" linear probes "latent" causal answer encoded residual stream small models Qwen Gemma "held-out" probe accuracy 2026
12. "Corr2Cause" label errors OR "incorrect labels" OR "annotation errors" OR "ground truth" bug dataset audit
13. "CLadder" "Corr2Cause" benchmark validity critique "construct validity" OR "shortcut" OR "spurious" synthetic causal benchmarks LLM 2026
14. Zečević "Causal Parrots" OR "correlations of causal facts" LLM causal shortcut; "Probing Causality Manipulation of Large Language Models" arXiv 2408.14380
15. "Corr2Cause" "hypothesis only" OR "hypothesis-only" OR "premise-only" OR "partial input" OR "partial-input"
16. "CLadder" "question-only" OR "numbers-only" OR "partial input" OR "answer prior" OR "text-only baseline" probe causal
17. probing LLM hidden states "causal direction" OR "causal graph" OR "collider" OR "confounder" linear probe decodes structure "text baseline" OR "bag-of-words baseline" OR "control task" 2025 2026 arXiv
18. "Causal Discovery and Inference through Next-Token Prediction" probe OR decode structural causal model transformer
19. "corr2cause" "directly causes" "never" valid OR positive "child" template shortcut alphabetical github OR blog OR kaggle
20. "CLadder" sparse autoencoder OR "activation patching" OR "steering vector" OR "linear probe" causal reasoning rungs hidden states
21. fine-tuned BERT RoBERTa Corr2Cause "variable refactorization" explanation shortcut why models fail "variable names" causal discovery text benchmark analysis
22. synthetic causal reasoning benchmark LLM "answer can be predicted" OR "trivially predictable" OR "heuristic baseline" numbers comparison "CLadder" OR "CausalBench" OR "CaLM" shallow heuristics
23. Corr2Cause "balanced" subset OR "debiased" OR "resampled" OR "re-split" evaluation hypothesis premise causal inference from correlation LLM
24. probe accuracy vs "input-only" OR "text-only" baseline "TF-IDF" truthfulness probe confound "what the probe" decodes surface features LLM activations 2025
25. LLM "conditional independence" statements causal discovery "attention heads" OR "circuit" OR "probing" OR "representations" PC algorithm natural language premises mechanistic analysis Corr2Cause-style
26. causal reasoning benchmark LLM "question-only" OR "context-free" OR "no-premise" baseline dataset artifacts synthetic templated benchmark predictable labels 2025 2026 "causal"
27. "CLadder" "backdoor adjustment" questions answer determined "case by case" OR "method 1" OR "method 2" shortcut predictable
28. Corr2Cause audit label errors "compute_relation" OR "6,958" OR "6,943" paper arXiv 2026 "child" template no positive instances
29. "Corr2Cause" audit OR "re-examination" OR "revisiting" OR "a closer look" benchmark 2026 arXiv dataset issues labels shortcuts

Listings:

- arXiv API, `search_query=all:Corr2Cause`, sorted by submission date: 8 results.
- arXiv API, `search_query=all:CLadder`, sorted by submission date: 12 results.
- Semantic Scholar Graph API, `paper/arXiv:2306.05836/citations` (248 rows) and
  `paper/arXiv:2312.04350/citations` (213 rows), fields title / year / externalIds / venue.
- Semantic Scholar keyword search for "Corr2Cause", "CLadder causal", "Corr2Cause shortcut",
  "CLadder probe": rate-limited, no results.
- Google Scholar query `"Corr2Cause" "hypothesis-only"`: redirect, not opened.
- OpenReview forum and API for vqIH0ObdqL: blocked.
- GitHub issue lists for `causalNLP/corr2cause` and `causalNLP/cladder`; Hugging Face discussion
  #6 on `causal-nlp/corr2cause`.
