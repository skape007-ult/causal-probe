# Verify yourself (fine-tuning study)

Continues the checklist of `notes/handoff.md`, part (c). Tick items off here.

## 1. The Corr2Cause paper on fine-tuning (Jin et al., ICLR 2024, https://arxiv.org/abs/2306.05836, v3)

Everything below was read on 16 Nov through a tool that fetches the arXiv HTML and has a small model answer questions
about it. It is not a copy of the page. **Open the page and check each line before citing any of it.**

- [ ] Sec. 4.3 lists the fine-tuned models as: BERT-Base/Large and RoBERTa-Base/Large, each with and without MNLI;
      GPT2, GPT2-Large, GPT2-XL; LLaMA-7B, LLaMA2-7B; GPT-3 Ada, Babbage, Curie, Davinci (17 in all).
- [ ] Sec. 4.3, quoted: "RoBERTa-Large MNLI, achieves 94.74% F1 score on this task".
- [ ] Sec. 4.5, quoted: "we simply paraphrase the hypothesis by changing the text template for each causal relation".
- [ ] Sec. 4.5, quoted: "we reverse the alphabet of the variable names, namely flipping A, B, C, to Z, Y, X".
- [ ] Sec. 4.5, quoted: "all the models drop drastically, by up to 39.29 on the paraphrased test set".
- [ ] Sec. 4.5, quoted: "it is very important to accompany adversarial attack together with the i.i.d. test set."
- [ ] Sec. 4.5, quoted: "We open-source our perturbed test sets" (the tool's rendering; check the full sentence).
- [ ] Table 5, F1 original / paraphrased / refactored, as the tool extracted them:
      GPT-3 Ada 79.85 / 61.73 / 41.57; Babbage 78.19 / 62.34 / 43.28; Curie 81.23 / 64.93 / 45.32; Davinci 85.52 / 65.01 / 46.96;
      GPT2 89.18 / 56.76 / 31.70; GPT2-Large 94.29 / 55.95 / 31.99; GPT2-XL 94.30 / 60.32 / 43.95;
      LLaMA-7B 91.98 / 56.41 / 53.92; LLaMA2-7B 92.92 / 52.24 / 49.47;
      BERT-Base 69.29 / 61.13 / 35.20; BERT-Large 85.26 / 63.64 / 38.54; RoBERTa-Base 87.60 / 65.58 / 53.12; RoBERTa-Large 89.10 / 65.05 / 60.20;
      BERT-Base MNLI 89.88 / 65.56 / 31.50; BERT-Large MNLI 90.19 / 67.24 / 52.04; RoBERTa-Base MNLI 94.27 / 57.42 / 62.83; RoBERTa-Large MNLI 94.74 / 55.45 / 67.87.
- [ ] **A discrepancy to settle.** With those numbers the largest paraphrase drop is LLaMA2-7B, 92.92 − 52.24 = 40.68,
      larger than the 39.29 the text calls the largest (which is RoBERTa-Large MNLI, 94.74 − 55.45). Either the
      extraction of the LLaMA2-7B row is wrong, or the paper's sentence does not cover it. Until you have looked,
      cite "39.29 (the paper's text)" and not "the largest drop in Table 5".
- [ ] The 62.30 under refactorisation is GPT2-Large (94.29 − 31.99): check the row.
- [ ] Sec. 4.4, quoted: the model is "several points weaker on the Has-Collider relations" (F1 92.19 in Table 6a).
- [ ] Appendix A, quoted: "which is usually around ten epochs"; "We use the validation set to tune the learning rate"
      over {2e-6, 5e-6, 1e-5, 2e-5, 5e-5}. The tool found no sentence saying every model used all 205,734 items.
- [ ] Appendix E, quoted: "some spurious correlations are rooted in the framing of the hypothesis". (v1.0's handoff
      says the paper does not use the words "shortcut", "artifact" or "hypothesis-only"; this sentence is the
      nearest it comes. Read the appendix before repeating either statement.)
- [ ] The abstract, quoted: "they still fail to generalize".

## 2. What I checked directly against the released files (code, not the tool)

- [ ] Hugging Face `causal-nlp/corr2cause` holds `perturbation_by_paraphrasing_test.csv` and
      `perturbation_by_refactorization_test.csv` (also dev and train versions); both test files have 2,246 rows on the
      21 test premises. `finetune/itemsets/build_report.json`, keys `paraphrase_table_against_authors_file` and
      `renaming_against_authors_file`: all 1,162 official test items, paraphrased or renamed by our code, occur in
      the authors' files with the same label. Rerun: `../causal-probe/.venv/bin/python finetune/scripts/build_data.py --check-only`.
- [ ] The paraphrase file is three-way labelled (`relation`: entailment 342, neutral 712, contradiction 1,192); we
      take valid = entailment. The paper's binary F1 on it presumably does the same; that is an assumption.
- [ ] The authors' paraphrase gives "parent" and "child" one wording ("X directly affects Y") and the two indirect
      relations one wording ("X influences Y through some mediator(s)"), so it changes more than surface form.
