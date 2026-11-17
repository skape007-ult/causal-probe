"""Item sets of the fine-tuning study: how they are built, stored and rebuilt identically on Kaggle.

A set is stored as a small JSON (finetune/itemsets/sets/<name>.json): the uids of official items, an optional transform
(rename, paraphrase, premise swap with the donor item's uid), and the sha256 of the materialised texts and labels.
`materialise` rebuilds the items from the raw CSVs and refuses a set whose hash does not match.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np

from causal_probe import data as D

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / "finetune" / "itemsets" / "sets"
csv.field_size_limit(10 ** 9)

# ---------------------------------------------------------------------------- graph classes
_STATEMENT = [
    (re.compile(r"^([A-F]) correlates with ([A-F])$"), "c"),
    (re.compile(r"^(?:However, )?([A-F]) is independent of ([A-F])$"), "i"),
    (re.compile(r"^(?:However, )?([A-F]) and ([A-F]) are independent given (.+)$"), "i"),
]


def parse_premise(premise: str):
    """-> (number of variables, frozenset of statements). A statement is (kind, pair, conditioning set).
    Raises ValueError on a sentence it does not recognise: the classes must rest on every statement."""
    n = int(re.search(r"closed system of (\d+) variables", premise).group(1))
    st = set()
    for s in premise.split("as follows:", 1)[1].split(". "):
        s = s.strip().rstrip(".")
        if not s:
            continue
        for rx, kind in _STATEMENT:
            m = rx.match(s)
            if m:
                cond = frozenset(re.findall(r"\b[A-F]\b", m.group(3))) if m.lastindex == 3 else frozenset()
                st.add((kind, frozenset(m.groups()[:2]), cond))
                break
        else:
            raise ValueError("unrecognised statement: " + s[:80])
    return n, frozenset(st)


def graph_class(premise: str) -> str:
    """Canonical form of the premise's statement set under every relabelling of its variables: two premises have
    the same class exactly when one is the other with the variables renamed."""
    n, st = parse_premise(premise)
    V = "ABCDEF"[:n]
    best = None
    for perm in itertools.permutations(V):
        m = dict(zip(V, perm))
        key = tuple(sorted((k, tuple(sorted(m[x] for x in a)), tuple(sorted(m[x] for x in c))) for k, a, c in st))
        if best is None or key < best:
            best = key
    return hashlib.md5(repr((n, best)).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------- transforms
# The authors' paraphrase templates, read from their released perturbation_by_paraphrasing_test.csv (one per causal
# relation; `check_paraphrase_against_authors` verifies this table against that file).
_PARAPHRASE = [
    (re.compile(r"^([A-F]) directly causes ([A-F])\.$"), r"\1 directly affects \2."),
    (re.compile(r"^([A-F]) causes something else which causes ([A-F])\.$"), r"\1 influences \2 through some mediator(s)."),
    (re.compile(r"^([A-F]) is a cause for ([A-F]), but not a direct one\.$"), r"\1 influences \2 through some mediator(s)."),
    (re.compile(r"^There exists at least one collider \(i\.e\., common effect\) of ([A-F]) and ([A-F])\.$"),
     r"\1 and \2 together cause some other variable(s)."),
    (re.compile(r"^There exists at least one confounder \(i\.e\., common cause\) of ([A-F]) and ([A-F])\.$"),
     r"Some variable(s) cause(s) both \1 and \2."),
]


def paraphrase(item: dict) -> dict:
    for rx, rep in _PARAPHRASE:
        if rx.match(item["query"]):
            return dict(item, uid=item["uid"] + "-par", query=rx.sub(rep, item["query"]))
    raise ValueError("no paraphrase template for: " + item["query"])


KEEP_ORDER_MAP = dict(zip("ABCDEF", "UVWXYZ"))
_VAR_AF = re.compile(r"\b([A-F])\b")


def rename_keep_order(item: dict) -> dict:
    """A..F -> U..Z: new letters, alphabetical order kept. The authors' refactorisation (v1.0 `rename_variables`,
    A..F -> Z..U) also REVERSES the order, and in this benchmark a directional hypothesis whose first letter is the
    later one is never valid; this version changes the names only."""
    sub = lambda t: _VAR_AF.sub(lambda m: KEEP_ORDER_MAP[m.group(1)], t)
    return dict(item, uid=item["uid"] + "-rko", context=sub(item["context"]), query=sub(item["query"]))


def swap_strata(premises: dict[str, tuple[int, int]], min_size: int = 10) -> dict[str, int]:
    """premise -> stratum id. premises: group -> (number of variables, premise tokens). Inside each variable count the
    premises are sorted by length and cut into consecutive strata of at least `min_size` premises (the last one
    absorbs the remainder); a variable count with fewer than min_size premises is one stratum."""
    out, sid = {}, 0
    for nv in sorted({v[0] for v in premises.values()}):
        gs = sorted((g for g, v in premises.items() if v[0] == nv), key=lambda g: (premises[g][1], g))
        k = max(1, len(gs) // min_size)
        for j, g in enumerate(gs):
            out[g] = sid + min(j // min_size, k - 1)
        sid += k
    return out


def nearest_premise_scores(train_items: list[dict], test_items: list[dict], k: int = 1) -> np.ndarray:
    """The nearest-training-premise reference: for each test item, the label its hypothesis STRING has in the training
    premise (same number of variables, among those that carry this string in `train_items`) whose statement set
    differs least from the test premise's (size of the symmetric difference, same letters). Score = that label, minus
    a small multiple of the distance so that ties are ordered by closeness of a valid neighbour; 0.5 if no training
    premise carries the string. No model, no fitting: what memorising the training items and copying gives."""
    st, lab = {}, {}
    for it in train_items:
        if it["group"] not in st:
            st[it["group"]] = parse_premise(it["context"])
        lab.setdefault(it["query"], {})[it["group"]] = it["label"]
    cache, out = {}, np.full(len(test_items), 0.5)
    for i, it in enumerate(test_items):
        if it["group"] not in cache:
            cache[it["group"]] = parse_premise(it["context"])
        n, s = cache[it["group"]]
        cand = [(len(s ^ st[g][1]), g) for g in lab.get(it["query"], {}) if st[g][0] == n]
        if cand:
            cand.sort()
            top = cand[:k]
            out[i] = float(np.mean([lab[it["query"]][g] for _, g in top])) - 1e-4 * top[0][0] * (1 if lab[it["query"]][top[0][1]] else -1)
    return out


MEMORY_FAMILY = [(mode, kind, par) for mode in ("all", "queried") for kind, pars in (("k", (1, 3, 5, 9)), ("exp", (1.0, 2.0, 4.0))) for par in pars]


def memorisation_family(train_items: list[dict], test_items: list[dict]) -> dict[str, np.ndarray]:
    """The registered family of copy-from-training references (PREREG E1, E5b). For a test item, the candidate
    neighbours are the training premises with the same number of variables that carry its hypothesis string.
    Distance: size of the symmetric difference of the two statement sets, over all statements ("all") or only over
    statements that name one of the two queried variables ("queried"). Score: the mean label of the k nearest
    (k = 1, 3, 5, 9; ties in distance broken by premise id), or the mean label of all candidates weighted by
    exp(-distance / tau) (tau = 1, 2, 4). 0.5 when there is no candidate. Fourteen members."""
    st, lab = {}, {}
    for it in train_items:
        if it["group"] not in st:
            st[it["group"]] = parse_premise(it["context"])
        lab.setdefault(it["query"], {})[it["group"]] = it["label"]
    out = {f"{m}|{k}|{p}": np.full(len(test_items), 0.5) for m, k, p in MEMORY_FAMILY}
    pcache = {}
    for i, it in enumerate(test_items):
        if it["group"] not in pcache:
            pcache[it["group"]] = parse_premise(it["context"])
        n, s = pcache[it["group"]]
        qv = set(re.findall(r"\b[A-Z]\b", it["query"]))
        cands = sorted(g for g in lab.get(it["query"], {}) if st[g][0] == n)
        if not cands:
            continue
        y = np.array([lab[it["query"]][g] for g in cands], float)
        for mode in ("all", "queried"):
            f = (lambda x: x) if mode == "all" else (lambda x: {t for t in x if (t[1] | t[2]) & qv})
            mine = f(s)
            d = np.array([len(mine ^ f(st[g][1])) for g in cands], float)
            order = np.argsort(d, kind="stable")
            for k in (1, 3, 5, 9):
                out[f"{mode}|k|{k}"][i] = y[order[:k]].mean()
            for tau in (1.0, 2.0, 4.0):
                w = np.exp(-(d - d.min()) / tau)
                out[f"{mode}|exp|{tau}"][i] = float((w * y).sum() / w.sum())
    return out


def premise_counts(items: list[dict]) -> np.ndarray:
    """Per item: number of correlation statements, of marginal and of conditional independence statements, log of
    the premise's word count. Surface statistics of the premise; nothing about the queried pair."""
    return np.column_stack([_premise_features(items)[:, 1:], np.log([len(i["context"].split()) for i in items])])


def premise_counts_within_string(fit_items, test_items, boosting: bool = False) -> np.ndarray:
    """A per-hypothesis-string classifier on `premise_counts`, fitted on fit_items, scored on test_items (0.5 for a
    string with fewer than 8 training items or one label). Logistic regression, or gradient boosting."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    Xf, Xt = premise_counts(fit_items), premise_counts(test_items)
    yf = np.array([i["label"] for i in fit_items]); qf = np.array([i["query"] for i in fit_items]); qt = np.array([i["query"] for i in test_items])
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
    out = np.full(len(test_items), 0.5)
    for s in np.unique(qt):
        a, b = np.where(qf == s)[0], np.where(qt == s)[0]
        if len(a) < 8 or len(np.unique(yf[a])) < 2:
            continue
        m = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0) if boosting else LogisticRegression(C=1.0, max_iter=1000)
        out[b] = m.fit((Xf[a] - mu) / sd, yf[a]).predict_proba((Xt[b] - mu) / sd)[:, 1]
    return out


def nearest_premise_distance(train_items: list[dict], test_items: list[dict]) -> dict[str, int]:
    """test premise -> smallest symmetric difference of statement sets to any training premise of the same size."""
    st = {}
    for it in train_items:
        if it["group"] not in st:
            st[it["group"]] = parse_premise(it["context"])
    out = {}
    for it in test_items:
        if it["group"] not in out:
            n, s = parse_premise(it["context"])
            out[it["group"]] = min((len(s ^ t) for m, t in st.values() if m == n), default=-1)
    return out


def swap_premise(item: dict, donor: dict) -> dict:
    """Condition C: the item's hypothesis and label on another item's premise."""
    assert donor["num_variables"] == item["num_variables"] and donor["group"] != item["group"]
    return dict(item, uid=item["uid"] + "-swap", context=donor["context"], group=donor["group"], own_group=item["group"])


def load_authors(kind: str, distinct: bool = True) -> list[dict]:
    """The authors' perturbed test sets ("paraphrasing" | "refactorization"), as items. valid = label 1
    (the paraphrase file is three-way: 1 entailment, 0 neutral, 2 contradiction)."""
    out, seen = [], set()
    with open(D.DATA_RAW / "corr2cause" / f"perturbation_by_{kind}_test.csv") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if distinct and r["input"] in seen:          # the files repeat rows (2,246 rows; 1,522 and 1,884 distinct inputs)
                continue
            seen.add(r["input"])
            prem, hyp = r["input"][len("Premise: "):].split(D.HYP_SEP)
            out.append(dict(uid=f"c2c-authors-{kind}-{i}", dataset="c2c", split="authors_" + kind, context=prem, query=hyp,
                            label=int(r["label"] == "1"), group=f"prem-{D._h(prem)}", template="", num_variables=-1))
    return out


# ---------------------------------------------------------------------------- stored sets
def digest(items: list[dict]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(f"{it['context']}\t{it['query']}\t{it['label']}\n".encode())
    return h.hexdigest()


def save_set(name: str, items: list[dict], transform: str = "none", donors: list[dict] | None = None, note: str = "", **extra):
    base = [it["uid"].removesuffix({"rename": "-ren", "rename_keep_order": "-rko", "paraphrase": "-par", "premise_swap": "-swap"}.get(transform, "\0"))
            for it in items]
    spec = {"name": name, "n": len(items), "transform": transform, "note": note, **extra, "sha256": digest(items), "uids": base}
    if donors is not None:
        spec["donor_uids"] = [d["uid"] for d in donors]
    SETS.mkdir(parents=True, exist_ok=True)
    (SETS / f"{name}.json").write_text(json.dumps(spec) + "\n")
    return spec


_UNIVERSE: dict[str, dict] = {}


def universe() -> dict[str, dict]:
    """uid -> official item, for train and test (about 1 GB of Python objects: loaded once)."""
    if not _UNIVERSE:
        for split in ("train", "test"):
            for it in D.load_c2c(split):
                _UNIVERSE[it["uid"]] = it
    return _UNIVERSE


def materialise(name: str) -> list[dict]:
    spec = json.loads((SETS / f"{name}.json").read_text())
    if spec["transform"] == "authors":
        items = load_authors(spec["kind"])
    else:
        U = universe()
        items = [U[u] for u in spec["uids"]]
        if spec["transform"] == "rename":
            items = [D.rename_variables(i) for i in items]
        elif spec["transform"] == "rename_keep_order":
            items = [rename_keep_order(i) for i in items]
        elif spec["transform"] == "paraphrase":
            items = [paraphrase(i) for i in items]
        elif spec["transform"] == "premise_swap":
            items = [swap_premise(i, U[d]) for i, d in zip(items, spec["donor_uids"])]
        elif spec["transform"] != "none":
            raise KeyError(spec["transform"])
    if digest(items) != spec["sha256"] or len(items) != spec["n"]:
        raise RuntimeError(f"set {name}: the rebuilt items do not match the stored hash; raw data or code differ from the build")
    return items


# ---------------------------------------------------------------------------- shortcut checks (PREREG, B (iv))
def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _premise_features(items):
    feat = {}
    for it in items:
        if it["group"] not in feat:
            n, st = parse_premise(it["context"])
            feat[it["group"]] = [n, sum(k == "c" for k, _, _ in st), sum(k == "i" and not c for k, _, c in st),
                                 sum(k == "i" and bool(c) for k, _, c in st)]
    return np.array([feat[it["group"]] for it in items], float)


def shortcut_transfer(fit_items, fit_tokens, test_items, test_tokens, n_boot: int = 500, seed: int = 0) -> dict:
    """Can a classifier that sees only (1) the hypothesis string, (2) the prompt length within a string, (3) the
    premise's surface statistics, (4) string and length jointly, learn the label on `fit_items` and predict it on
    `test_items` (other premises)? AUROC with a premise-bootstrap interval for each.

    Why fit and test are two separately built sets and not folds of one: in a set that is exactly half valid inside
    every string, a string's label rate in the training folds is the mirror image of its rate in the held-out fold,
    and every such classifier scores BELOW one half (0.37 to 0.43 on condition B, 16 Nov; v1.0 decision D21 met the
    same effect). That number says nothing about a leak in either direction."""
    from sklearn.linear_model import LogisticRegression
    yf = np.array([i["label"] for i in fit_items]); yt = np.array([i["label"] for i in test_items])
    qf = np.array([i["query"] for i in fit_items]); qt = np.array([i["query"] for i in test_items])
    lf, lt = np.log(np.asarray(fit_tokens, float)), np.log(np.asarray(test_tokens, float))
    Pf, Pt = (np.column_stack([_premise_features(x), l]) for x, l in ((fit_items, lf), (test_items, lt)))
    mu, sd = Pf.mean(0), Pf.std(0) + 1e-9
    sc = {"premise_only": LogisticRegression(C=1.0, max_iter=1000).fit((Pf - mu) / sd, yf).predict_proba((Pt - mu) / sd)[:, 1]}
    rate = {s: (yf[qf == s].sum() + 1) / ((qf == s).sum() + 2) for s in np.unique(qf)}
    sc["hypothesis_only"] = np.array([rate.get(s, 0.5) for s in qt])
    sc["length_within_string"] = np.zeros(len(yt)); sc["string_and_length"] = sc["hypothesis_only"].copy()
    for s in np.unique(qt):
        a, b = np.where(qf == s)[0], np.where(qt == s)[0]
        if len(a) < 8 or len(np.unique(yf[a])) < 2:
            continue
        c = lf[a].mean()
        m = LogisticRegression(C=10.0, max_iter=1000).fit(np.column_stack([lf[a] - c, (lf[a] - c) ** 2]), yf[a])
        pr = m.predict_proba(np.column_stack([lt[b] - c, (lt[b] - c) ** 2]))[:, 1]
        sc["string_and_length"][b] = pr
        sc["length_within_string"][b] = pr - pr.mean()          # centred inside the string: only length is left
    rng = np.random.default_rng(seed)
    g = np.array([i["group"] for i in test_items])
    ug, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(ug))]
    draws = [np.concatenate([members[j] for j in rng.integers(0, len(ug), len(ug))]) for _ in range(n_boot)]
    out = {}
    for k, v in sc.items():
        if np.ptp(v) == 0:                                     # every score equal: exactly uninformative
            out[k] = {"auroc": 0.5, "lo": 0.5, "hi": 0.5, "passes": True, "note": "constant score"}
            continue
        bs = [_auc(yt[ix], v[ix]) for ix in draws if 0 < yt[ix].mean() < 1]
        lo, hi = np.quantile(bs, [0.025, 0.975]); est = _auc(yt, v)
        out[k] = {"auroc": est, "lo": float(lo), "hi": float(hi), "passes": bool(abs(est - 0.5) <= 0.02 and lo <= 0.5 <= hi)}
    out["all_pass"] = all(v["passes"] for v in out.values())
    return out


def in_sample_balance(items, n_tokens) -> dict:
    """What holds by construction, stated on the built set: the label rate inside every string and every premise, and
    (no fitting) the AUROC of the token count for the label inside each string."""
    y = np.array([i["label"] for i in items]); n = np.asarray(n_tokens, float)
    out = {}
    for key in ("query", "group", "template"):
        k = np.array([i[key] for i in items])
        out[f"cells_not_half_valid_by_{key}"] = int(sum(y[k == u].mean() != 0.5 for u in np.unique(k)))
    q = np.array([i["query"] for i in items]); t = np.array([i["template"] for i in items])
    per = np.array([_auc(y[q == s], n[q == s]) for s in np.unique(q)])
    out["token_count_within_string"] = {"mean_auroc": float(per.mean()), "largest_deviation_of_a_string": float(np.abs(per - 0.5).max())}
    out["token_count_within_template"] = {u: _auc(y[t == u], n[t == u]) for u in np.unique(t)}
    out["passes"] = bool(out["cells_not_half_valid_by_query"] == 0 and out["cells_not_half_valid_by_group"] == 0
                         and abs(per.mean() - 0.5) <= 0.02 and max(abs(v - 0.5) for v in out["token_count_within_template"].values()) <= 0.02)
    return out
