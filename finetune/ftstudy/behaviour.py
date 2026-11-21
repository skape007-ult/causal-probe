"""Behaviour analysis E1 to E5 of PREREG.md (version 2 and Amendment 1), as functions of score vectors.

Nothing here reads files: finetune/scripts/behaviour.py loads the sets, the references and the models' scores, and
calls `analyse`. That keeps the same code testable on synthetic scores (finetune/tests/test_behaviour.py).

Blinding (owner's decision, 17 Nov; CLAUDE.md item 7): no A-vs-B comparison of any kind on the evaluation sets
until all five seeds of A and of B exist. `check_blinding` is called by the loader before a single A or B score is
read, and again by `analyse`.

Speed. The registered statistic S (stat.S, which calls the owner's part (b) or its reference) is used for every
point estimate. The 2,000 premise resamples use `Plan`, an equivalent closed form (the within-string least-squares
fit as a projection matrix computed once per resample and shared by every model). `analyse` checks on every point
estimate and on the first resamples that the two agree to TOL (1e-4; see TOL), records the largest difference,
and stops if they disagree by more.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from . import seedtest, stat

REGISTERED_SEEDS = {"A": 5, "B": 5, "A48": 3, "C": 3, "Bswap": 3}
FAMILY = ["E2a", "E2b", "E3", "E4a", "E4b", "E5a_A", "E5a_B", "E5b_A", "E5b_B", "E5c_A", "E5c_B"]
GATE_S_ID48, GATE_AUROC_ID = 0.80, 0.922
SEED_SD_POWERED = 0.03
E5B_A_PREMISE_SE_POWERED = 0.014
E2C_BAND = 0.05


class Blinded(RuntimeError):
    pass


def check_blinding(models) -> None:
    """models: names like 'A_s0'. Refuses if any A or B model is among them while A_s0-4 and B_s0-4 are not all
    present."""
    models = set(models)
    uses_ab = any(m.split("_s")[0] in ("A", "B") for m in models)
    need = {f"{c}_s{s}" for c in ("A", "B") for s in range(5)}
    if uses_ab and not need <= models:
        missing = sorted(need - models)
        raise Blinded("blinded: no A or B evaluation scores are opened until all ten A and B seeds are ingested "
                      f"(missing: {', '.join(missing)}). Owner's decision, 17 Nov; CLAUDE.md item 7.")


# --------------------------------------------------------------------------- fast, equivalent statistic
class Plan:
    """S and S_cells on one resample of one set, for any score vector. Built from labels, strings, token counts (and
    adjacency); the scores only enter in `S` / `S_cells`."""

    def __init__(self, y, q, n, adj=None, min_class: int = stat.MIN_CLASS):
        y, q, n = np.asarray(y).astype(int), np.asarray(q), np.asarray(n, float)
        self.parts = []                    # (idx, fit, label mask, n1, n0) for eligible strings
        self.all_parts = []                # (idx, fit) for every string (S_cells residualises within every string)
        for u in np.unique(q):
            c = np.where(q == u)[0]
            lx = np.log(n[c]); lx = lx - lx.mean()
            X = np.column_stack([np.ones(len(c)), lx, lx ** 2])
            fit = (np.linalg.pinv(X), lx, lx ** 2)
            self.all_parts.append((c, fit))
            n1 = int(y[c].sum())
            if n1 >= min_class and len(c) - n1 >= min_class:
                self.parts.append((c, fit, y[c] == 1, n1, len(c) - n1))
        self.cells = []
        if adj is not None:
            adj = np.asarray(adj)
            key = np.array([f"{a}|{b}" for a, b in zip(q, adj)])
            for u in np.unique(key):
                c = np.where(key == u)[0]
                n1 = int(y[c].sum())
                if min(n1, len(c) - n1) >= min_class:
                    self.cells.append((c, y[c] == 1, n1, len(c) - n1))

    @staticmethod
    def _resid(sc, fit):
        """Score minus its least-squares fit on (1, lx, lx^2). The coefficients come from a matrix product; the fitted
        values are formed element by element, so that items with identical token counts get bit-identical fitted
        values and exact ties in the scores stay exact ties (a matrix product may round identical rows differently)."""
        P, lx, lx2 = fit
        b = P @ sc
        if sc.ndim == 1:
            return sc - (b[0] + b[1] * lx + b[2] * lx2)
        return sc - (b[0][None, :] + b[1][None, :] * lx[:, None] + b[2][None, :] * lx2[:, None])

    @staticmethod
    def _auc(r, pos, n1, n0):
        """r: (m,) or (m, K); AUROC of each column for the label, ties one half."""
        rk = rankdata(r, axis=0)
        return (rk[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)

    def S(self, s):
        """s: one score vector (m,) -> float, or a matrix (m, K) -> one S per column."""
        s = np.asarray(s, float)
        if not self.parts:
            return float("nan") if s.ndim == 1 else np.full(s.shape[1], np.nan)
        v = np.mean([self._auc(self._resid(s[c], fit), pos, n1, n0) for c, fit, pos, n1, n0 in self.parts], 0)
        return float(v) if s.ndim == 1 else v

    def S_cells(self, s):
        s = np.asarray(s, float)
        r = np.empty(s.shape)
        for c, fit in self.all_parts:
            r[c] = self._resid(s[c], fit)
        if not self.cells:
            return float("nan") if s.ndim == 1 else np.full(s.shape[1], np.nan)
        au = np.array([self._auc(r[c], pos, n1, n0) for c, pos, n1, n0 in self.cells])
        w = np.array([n1 * n0 for _, _, n1, n0 in self.cells], float)
        v = (w[:, None] * au.reshape(len(w), -1)).sum(0) / w.sum()
        return float(v[0]) if s.ndim == 1 else v


# --------------------------------------------------------------------------- data containers
@dataclass
class EvalSet:
    name: str
    y: np.ndarray
    q: np.ndarray            # hypothesis string identity (for renamed / paraphrased sets: the ORIGINAL string)
    n: np.ndarray            # token count of the prompt as evaluated
    groups: np.ndarray       # premise id (the resampling unit)
    adj: np.ndarray | None = None
    boot: list = field(default_factory=list)


def pooled_auroc(y, s) -> float:
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    rk = rankdata(np.asarray(s, float))
    return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def f1_valid(y, pred) -> float:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def classification(y, s) -> dict:
    pred = (np.asarray(s) > 0).astype(int)
    f1v, f1i = f1_valid(y, pred), f1_valid(1 - np.asarray(y), 1 - pred)
    return {"pooled_auroc": pooled_auroc(y, s), "f1_valid_at_0": f1v, "macro_f1_at_0": (f1v + f1i) / 2,
            "accuracy_at_0": float((pred == np.asarray(y)).mean())}


TOL = 1e-4      # agreement required between the closed form and the registered function. Not 1e-9: the registered
                # function forms fitted values with a matrix product, which can round identical rows differently and so
                # split an exact tie in the scores (seen on the base model: 2e-6). The largest difference is recorded.


class Engine:
    """Point estimates (registered stat.S) and resample vectors (Plan) for every (set, score vector, statistic).
    `precompute` evaluates all vectors of a set in one pass over the resamples (a resample's plan is built once and
    discarded: keeping 2,000 plans would take gigabytes); `boot` then reads the result."""

    def __init__(self, sets: dict[str, EvalSet], check_first: int = 3):
        self.sets, self.check_first = sets, check_first
        self.cache: dict = {}
        self.max_discrepancy = 0.0

    def point(self, k, key, s, kind="S") -> float:
        e = self.sets[k]
        if kind == "S":
            v, fast = stat.S(e.y, s, e.q, e.n), Plan(e.y, e.q, e.n).S(s)
        else:
            v, fast = stat.S_cells(e.y, s, e.q, e.adj, e.n), Plan(e.y, e.q, e.n, e.adj).S_cells(s)
        if not (np.isnan(v) and np.isnan(fast)):
            self.max_discrepancy = max(self.max_discrepancy, abs(v - fast))
        if not (np.isnan(v) and np.isnan(fast)) and abs(v - fast) > TOL:
            raise RuntimeError(f"fast statistic differs from the registered one on {k}/{key}/{kind}: {fast} vs {v}")
        return v

    def precompute(self, k, wanted: dict) -> None:
        """wanted: {(key, kind): score vector}. Fills the cache for set k."""
        todo = {kk: np.asarray(v, float) for kk, v in wanted.items() if (k,) + kk not in self.cache}
        if not todo:
            return
        e = self.sets[k]
        kinds = sorted({kind for _, kind in todo})
        keys = {kind: [kk for kk in todo if kk[1] == kind] for kind in kinds}
        mats = {kind: np.column_stack([todo[kk] for kk in keys[kind]]) for kind in kinds}
        res = {kind: np.empty((len(e.boot), len(keys[kind]))) for kind in kinds}
        for b, idx in enumerate(e.boot):
            pl = Plan(e.y[idx], e.q[idx], e.n[idx], None if e.adj is None else e.adj[idx])
            for kind in kinds:
                res[kind][b] = pl.S(mats[kind][idx]) if kind == "S" else pl.S_cells(mats[kind][idx])
            if b < self.check_first:              # the closed form against the registered function, on a resample
                for kind in kinds:
                    for j, kk in enumerate(keys[kind]):
                        s = todo[kk]
                        ref = (stat.S(e.y[idx], s[idx], e.q[idx], e.n[idx]) if kind == "S"
                               else stat.S_cells(e.y[idx], s[idx], e.q[idx], e.adj[idx], e.n[idx]))
                        if not (np.isnan(ref) and np.isnan(res[kind][b, j])):
                            self.max_discrepancy = max(self.max_discrepancy, abs(ref - res[kind][b, j]))
                        if not (np.isnan(ref) and np.isnan(res[kind][b, j])) and abs(ref - res[kind][b, j]) > TOL:
                            raise RuntimeError(f"fast statistic differs on resample {b} of {k}/{kk}: {res[kind][b, j]} vs {ref}")
        for kind in kinds:
            for j, kk in enumerate(keys[kind]):
                self.cache[(k,) + kk] = res[kind][:, j]

    def boot(self, k, key, s, kind="S") -> np.ndarray:
        if (k, key, kind) not in self.cache:
            self.precompute(k, {(key, kind): s})
        return self.cache[(k, key, kind)]


# --------------------------------------------------------------------------- one registered test, with the outlier rule
def _drop(v, out):
    return None if v is None else [x for i, x in enumerate(v) if i not in out]


def run_test(px, bx, py=None, by=None, constant=None, ref_point=None, ref_boot=None, paired=False) -> dict:
    """px: per-seed statistic of X; bx: list of their resample vectors. Exactly one comparison:
       py/by          two conditions (D = mean X - mean Y)
       ref_point/_boot one fixed reference score (D = mean X - S(reference); reference rescored on every resample)
       constant        a constant
       paired          px, bx are already within-seed differences; compared with 0.
    Applies PREREG's outlying-seed rule: if a seed of X (or Y) lies more than 0.10 from its condition's median, the
    test is also run without it; the result carries both."""
    def one(px_, bx_, py_, by_):
        mx = np.mean(np.vstack(bx_), 0)
        if py_ is not None:
            D_boot = mx - np.mean(np.vstack(by_), 0)
            return seedtest.test(px_, py_, boot_D=D_boot)
        if ref_point is not None:
            return {**seedtest.test(px_, constant=ref_point, boot_D=mx - ref_boot), "reference_S": ref_point}
        return seedtest.test(px_, constant=constant or 0.0, boot_D=mx - (constant or 0.0))
    res = one(px, bx, py, by)
    ox, oy = seedtest.outlying_seeds(px), (seedtest.outlying_seeds(py) if py is not None else [])
    res["outlying_seeds_x"], res["outlying_seeds_y"] = ox, oy
    if ox or oy:
        if len(px) - len(ox) >= 2 and (py is None or len(py) - len(oy) >= 2):
            res["without_outliers"] = one(_drop(px, ox), _drop(bx, ox), _drop(py, oy), _drop(by, oy))
        else:
            res["without_outliers"] = {"p_two_sided": 1.0, "note": "fewer than two seeds left"}
    return res


def holm_with_outlier_rule(tests: dict) -> None:
    """Adds holm_p and claimed to every test of the family, in place. A test with an outlying seed is claimed only if
    the Holm-adjusted p is below 0.05 both with the seed and without it (the family recomputed with the 'without' p)."""
    # The family is always the eleven registered tests: one that could not be run counts with p = 1, so that Holm is
    # never less strict than registered.
    p = {k: tests[k]["p_two_sided"] if k in tests else 1.0 for k in FAMILY}
    adj = seedtest.holm(p)
    p_wo = {k: tests[k].get("without_outliers", tests[k])["p_two_sided"] if k in tests else 1.0 for k in FAMILY}
    adj_wo = seedtest.holm(p_wo)
    tests.setdefault("_not_run", [k for k in FAMILY if k not in tests])
    for k in [k for k in FAMILY if k in tests]:
        t = tests[k]
        t["holm_p"] = adj[k]
        t["holm_p_without_outliers"] = adj_wo[k] if "without_outliers" in t else None
        t["claimed"] = bool(adj[k] < 0.05 and ("without_outliers" not in t or adj_wo[k] < 0.05))
        t["direction"] = ("above" if t["D"] > 0 else "below") if t["claimed"] else "not shown"
        notes = []
        for side in ("seed_sd_x", "seed_sd_y"):
            if t.get(side) is not None and t[side] > SEED_SD_POWERED:
                notes.append(f"observed {side.replace('_', ' ')} {t[side]:.3f} > {SEED_SD_POWERED}: underpowered for its registered effect")
        t["power_notes"] = notes


def rung(e5a: dict, e5b: dict, e5c: dict) -> str:
    up = lambda t: t["claimed"] and t["D"] > 0                                       # noqa: E731
    if not up(e5a):
        return "L0" + (" (E5b claimed with E5a not shown: reported as found, no rung above L0)" if up(e5b) else "")
    if not up(e5b):
        return "L1"
    return "L3" if up(e5c) else "L2"


def e2c_reading(t: dict) -> str:
    if t["below_alpha"]:
        return "claimed different from 0.5: B and POOL-LB share an artefact; E2a, E2b and E5 are not interpreted until it is explained"
    if t["lo"] >= -E2C_BAND and t["hi"] <= E2C_BAND:
        return "pass: interval inside 0.5 +/- 0.05"
    return "inconclusive: interval not inside 0.5 +/- 0.05 and not excluding 0.5 (expected with 3 seeds; Amendment 1, item 2)"


def share_interval(e: EvalSet, a_scores: list, c_scores: list) -> dict:
    """E3, descriptive: (AUROC_C - 0.5) / (AUROC_A - 0.5) on ID, mean over seeds, with a premise-bootstrap interval."""
    def sh(idx):
        aa = np.mean([pooled_auroc(e.y[idx], s[idx]) for s in a_scores]); cc = np.mean([pooled_auroc(e.y[idx], s[idx]) for s in c_scores])
        return (cc - 0.5) / (aa - 0.5)
    full = sh(np.arange(len(e.y)))
    bs = np.array([sh(i) for i in e.boot])
    return {"est": float(full), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975))}


# --------------------------------------------------------------------------- the whole analysis
def analyse(sets: dict[str, EvalSet], scores: dict[str, dict[str, np.ndarray]], refs: dict, registered_only=True) -> dict:
    """sets: pool_lb, id48, id, id48_renamed, id48_renamed_keep_order (EvalSet, with .boot filled).
    scores: model ('A_s0', ...) -> set name -> score vector, aligned with the set.
    refs: {'mem': {cond: {set: vector}}, 'mem_variant': {cond: {set: vector}}, 'text_best': {cond: (name, vector)},
           'mem_registered_S': {cond: {set: float}}}
    Returns everything the page and the audit need, per E-number."""
    check_blinding(scores)
    eng = Engine(sets)
    seeds = {c: [m for m in sorted(scores) if m.split("_s")[0] == c][:REGISTERED_SEEDS[c]] for c in REGISTERED_SEEDS}
    extra_seeds = {c: [m for m in sorted(scores) if m.split("_s")[0] == c][REGISTERED_SEEDS[c]:] for c in REGISTERED_SEEDS}
    for c in ("A", "B"):
        if len(seeds[c]) != 5:
            raise Blinded(f"{c} has {len(seeds[c])} seeds; the registered tests need 5")

    def P(c, k, kind="S"):
        return [eng.point(k, m, scores[m][k], kind) for m in seeds[c]]

    def Bv(c, k, kind="S"):
        return [eng.boot(k, m, scores[m][k], kind) for m in seeds[c]]

    out = {"seeds": seeds, "extra_seeds_not_in_registered_tests": extra_seeds, "tests": {}, "descriptive": {}}
    want = {k: {} for k in sets}
    for c, ms in seeds.items():
        for m in ms:
            want["pool_lb"][(m, "S")] = scores[m]["pool_lb"]
            if c in ("A", "B", "A48"):
                want["pool_lb"][(m, "cells")] = scores[m]["pool_lb"]
            if c in ("A", "C"):
                want["id48"][(m, "S")] = scores[m]["id48"]
            if c == "A":
                for k in ("id48_renamed", "id48_renamed_keep_order"):
                    want[k][(m, "S")] = scores[m][k]
    for c in ("A", "B", "A48"):
        if c in refs["mem"]:
            want["pool_lb"][(f"mem_{c}", "S")] = refs["mem"][c]["pool_lb"]
    for c, v in refs.get("mem_variant", {}).items():
        want["pool_lb"][(f"memv_{c}", "S")] = v["pool_lb"]
    for c, (_, v) in refs["text_best"].items():
        want["pool_lb"][(f"text_{c}", "S")] = v                       # A, B and A48
    for k, w in want.items():
        if w:
            eng.precompute(k, w)
    T, Dd = out["tests"], out["descriptive"]

    # E2c first (read first)
    if seeds["Bswap"]:
        T["E2c"] = run_test(P("Bswap", "pool_lb"), Bv("Bswap", "pool_lb"), constant=0.5)
        T["E2c"]["reading"] = e2c_reading(T["E2c"])

    # E1: gate (A), no test
    sA_id48 = P("A", "id48")
    cls = [classification(sets["id"].y, scores[m]["id"]) for m in seeds["A"]]
    mean_auc = float(np.mean([c["pooled_auroc"] for c in cls]))
    Dd["E1"] = {"S_id48_per_seed": sA_id48, "S_id48_mean": float(np.mean(sA_id48)),
                "pooled_auroc_id_per_seed": [c["pooled_auroc"] for c in cls], "pooled_auroc_id_mean": mean_auc,
                "classification_id_per_seed": cls,
                "f1_at_threshold_maximising_training_f1": "not computed: training-set scores were not stored by ft_run.py",
                "memorisation_reference_A_id48": refs["mem_registered_S"]["A"]["id48"],
                "passed": bool(np.mean(sA_id48) >= GATE_S_ID48 and mean_auc >= GATE_AUROC_ID),
                "thresholds": {"S_id48": GATE_S_ID48, "pooled_auroc_id": GATE_AUROC_ID}}

    # E2a, E2b
    T["E2a"] = run_test(P("B", "pool_lb"), Bv("B", "pool_lb"), P("A", "pool_lb"), Bv("A", "pool_lb"))
    if seeds["A48"]:
        T["E2b"] = run_test(P("B", "pool_lb"), Bv("B", "pool_lb"), P("A48", "pool_lb"), Bv("A48", "pool_lb"))

    # E3
    if seeds["C"]:
        T["E3"] = run_test(P("A", "id48"), Bv("A", "id48"), P("C", "id48"), Bv("C", "id48"))
        Dd["E3"] = {"classification_id": {c: [classification(sets["id"].y, scores[m]["id"]) for m in seeds[c]] for c in ("A", "C")},
                    "share_of_A_pooled_auroc_available_without_premise": share_interval(
                        sets["id"], [scores[m]["id"] for m in seeds["A"]], [scores[m]["id"] for m in seeds["C"]])}

    # E4: within seed, A, on ID48 (string identity = the original string)
    def diff(k1, k2):
        px = [eng.point(k1, m, scores[m][k1]) - eng.point(k2, m, scores[m][k2]) for m in seeds["A"]]
        bx = [eng.boot(k1, m, scores[m][k1]) - eng.boot(k2, m, scores[m][k2]) for m in seeds["A"]]
        return run_test(px, bx, paired=True)
    T["E4a"] = diff("id48", "id48_renamed")
    T["E4b"] = diff("id48_renamed_keep_order", "id48_renamed")

    # E5 for A and B
    for c in ("A", "B"):
        T[f"E5a_{c}"] = run_test(P(c, "pool_lb", "cells"), Bv(c, "pool_lb", "cells"), constant=0.5)
        mem = refs["mem"][c]["pool_lb"]
        T[f"E5b_{c}"] = run_test(P(c, "pool_lb"), Bv(c, "pool_lb"), ref_point=eng.point("pool_lb", f"mem_{c}", mem),
                                 ref_boot=eng.boot("pool_lb", f"mem_{c}", mem))
        T[f"E5b_{c}"]["registered_reference_S"] = refs["mem_registered_S"][c]["pool_lb"]
        name, tb = refs["text_best"][c]
        T[f"E5c_{c}"] = run_test(P(c, "pool_lb"), Bv(c, "pool_lb"), ref_point=eng.point("pool_lb", f"text_{c}", tb),
                                 ref_boot=eng.boot("pool_lb", f"text_{c}", tb))
        T[f"E5c_{c}"]["reference"] = name
        if c in refs.get("mem_variant", {}):                         # Amendment 1, item 1: beside E5b, not in the family
            mv = refs["mem_variant"][c]["pool_lb"]
            Dd[f"E5b_{c}_against_pair_only_variant"] = run_test(P(c, "pool_lb"), Bv(c, "pool_lb"), ref_point=eng.point("pool_lb", f"memv_{c}", mv),
                                                                 ref_boot=eng.boot("pool_lb", f"memv_{c}", mv))
    if seeds["A48"]:                                                  # E5 for A48: intervals, no test in the family
        Dd["E5_A48"] = {"S_cells_minus_half": run_test(P("A48", "pool_lb", "cells"), Bv("A48", "pool_lb", "cells"), constant=0.5),
                        "S_minus_memorisation": run_test(P("A48", "pool_lb"), Bv("A48", "pool_lb"),
                                                         ref_point=eng.point("pool_lb", "mem_A48", refs["mem"]["A48"]["pool_lb"]),
                                                         ref_boot=eng.boot("pool_lb", "mem_A48", refs["mem"]["A48"]["pool_lb"]))}
        if "A48" in refs["text_best"]:
            name, tb = refs["text_best"]["A48"]
            Dd["E5_A48"]["S_minus_best_text"] = {**run_test(P("A48", "pool_lb"), Bv("A48", "pool_lb"), ref_point=eng.point("pool_lb", "text_A48", tb),
                                                            ref_boot=eng.boot("pool_lb", "text_A48", tb)), "reference": name}
        for t in Dd["E5_A48"].values():                              # descriptive: intervals only (audit 21 Nov, finding 11)
            t["below_alpha"] = None
            t["descriptive_only"] = "PREREG E5: reported for A48 with intervals, no test; p is not a claim"
            sds = [x for x in (t.get("seed_sd_x"),) if x is not None and x > SEED_SD_POWERED]
            t["power_notes"] = [f"seed SD {sds[0]:.3f} > {SEED_SD_POWERED}" ] if sds else []
    if "text_id48" in refs and "A" in refs["text_id48"]:              # PREREG E3: text references fitted on A's training set, beside E3
        Dd.setdefault("E3", {})["text_references_fitted_on_A_on_id48"] = refs["text_id48"]["A"]
    for k in [k for k in Dd if k.startswith("E5b_") and k.endswith("_against_pair_only_variant")]:
        Dd[k]["below_alpha"] = None
        Dd[k]["descriptive_only"] = "Amendment 1: reported beside E5b, not in the family"

    # E5, descriptive: the half of POOL-LB nearer than the median distance to a training premise, and the farther half
    e = sets["pool_lb"]
    for c in ("A", "B"):
        if c in refs.get("near", {}):
            near = np.asarray(refs["near"][c], bool)
            Ssub = lambda s, m: stat.S(e.y[m], np.asarray(s)[m], e.q[m], e.n[m])      # noqa: E731
            Dd[f"E5_near_far_{c}"] = {half: {"n_items": int(mask.sum()),
                                             "S_per_seed": [Ssub(scores[mm]["pool_lb"], mask) for mm in seeds[c]],
                                             "memorisation_reference_S": Ssub(refs["mem"][c]["pool_lb"], mask)}
                                      for half, mask in (("near", near), ("far", ~near))}

    holm_with_outlier_rule(T)
    for c in ("A", "B"):
        Dd[f"rung_{c}"] = rung(T[f"E5a_{c}"], T[f"E5b_{c}"], T[f"E5c_{c}"])
        if c in refs.get("mem_variant", {}) and T[f"E5b_{c}"]["claimed"] and T[f"E5b_{c}"]["D"] > 0:
            v = Dd[f"E5b_{c}_against_pair_only_variant"]
            if not (v["p_two_sided"] < 0.05 and v["D"] > 0):          # unadjusted, in the reference's disfavour
                Dd[f"rung_{c}"] += " (claimed above the registered memorisation reference but not above the reviewer's pair-only variant; Amendment 1)"
    pse = T["E5b_A"]["premise_se"]
    if pse > E5B_A_PREMISE_SE_POWERED:
        T["E5b_A"]["power_notes"].append(f"observed premise SE {pse:.3f} > {E5B_A_PREMISE_SE_POWERED}: E5b(A) underpowered for 0.10 (Amendment 1, item 3)")
    out["max_discrepancy_fast_vs_registered_statistic"] = eng.max_discrepancy
    out["S_per_model"] = {m: {k: eng.point(k, m, scores[m][k]) for k in ("pool_lb", "id48") if k in scores[m]} for m in scores}
    return out
