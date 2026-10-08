"""Recursive partitioning: a Python port of R's ``rpart`` (version 4.1.27).

Radiant's ``crtree`` is a thin layer over ``rpart::rpart``. To give the same
trees as Radiant, this module ports rpart's C code (``partition.c``,
``bsplit.c``, ``gini.c``, ``anova.c``, ``graycode.c``, ``surrogate.c``,
``choose_surg.c``, ``nodesplit.c``, ``make_cp_list.c``, ``make_cp_table.c``,
``xval.c``, ``rundown.c``, ``branch.c``) and the R code that turns the result
into a fitted object (``rpart.R``, ``rpart.class.R``, ``importance.R``,
``prune.rpart.R``, ``snip.rpart.R``, ``pred.rpart.R``). Function and variable
names follow the C code so the two can be read side by side.

Only the "anova" (regression) and "class" (classification) methods are
ported. Training data must be complete (no missing values), which is what
Radiant passes in as well; missing values in data used for prediction are
handled through the surrogate splits, as in rpart.

Sums are accumulated sequentially, in the same order as the C loops, so split
improvements (and hence ties between candidate splits) match rpart to the last
bit wherever the observation order allows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LEFT = -1
RIGHT = 1

__all__ = ["RPartFit", "rpart_fit", "r_sample_xgroups", "r_sample_int_replace", "RRandom"]


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------


def _seqsum(a) -> float:
    """Left-to-right sum, as a C ``for`` loop computes it (np.sum is pairwise)."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return 0.0
    return float(np.cumsum(a)[-1])


def _running_minus(start: float, a: np.ndarray) -> np.ndarray:
    """``start - a[0] - a[1] - ...`` after each step, subtracting one at a time."""
    return np.subtract.accumulate(np.concatenate(([start], a)))[1:]


def _bincount_seq(codes: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    """Per-category sums, each accumulated in observation order (as in C)."""
    return np.bincount(codes, weights=weights, minlength=size).astype(float)


def _mysort(x: np.ndarray) -> np.ndarray:
    """mysort.c: rpart's quicksort, returning the observation order.

    Not stable. Tied values end up in the order this particular algorithm
    leaves them, and that order decides the rounding of the running sums in
    the split search (and so which of two equally good splits wins).
    """
    x = [float(v) for v in x]
    cvec = list(range(len(x)))

    def sort(start: int, stop: int) -> None:
        while start < stop:
            if (stop - start) < 11:
                for i in range(start + 1, stop + 1):
                    temp = x[i]
                    tempd = cvec[i]
                    j = i - 1
                    while j >= start and x[j] > temp:
                        x[j + 1] = x[j]
                        cvec[j + 1] = cvec[j]
                        j -= 1
                    x[j + 1] = temp
                    cvec[j + 1] = tempd
                return
            i = start
            j = stop
            k = (start + stop) // 2
            median = x[k]
            if x[i] >= x[k]:
                if x[j] > x[k]:
                    median = x[j] if x[i] > x[j] else x[i]
            elif x[j] < x[k]:
                median = x[i] if x[i] > x[j] else x[j]
            while i < j:
                while x[i] < median:
                    i += 1
                while x[j] > median:
                    j -= 1
                if i < j:
                    if x[i] > x[j]:
                        x[i], x[j] = x[j], x[i]
                        cvec[i], cvec[j] = cvec[j], cvec[i]
                    i += 1
                    j -= 1
            while x[i] >= median and i > start:
                i -= 1
            while x[j] <= median and j < stop:
                j += 1
            if (i - start) < (stop - j):
                if (i - start) > 0:
                    sort(start, i)
                start = j
            else:
                if (stop - j) > 0:
                    sort(j, stop)
                stop = i

    sort(0, len(x) - 1)
    return np.array(cvec, dtype=np.int64)


# ---------------------------------------------------------------------------
# Gray code walk through category subsets (graycode.c)
# ---------------------------------------------------------------------------


class _GrayCode:
    def init1(self, numcat: int, count) -> None:
        self.maxc = numcat
        self.gray = [1 if count[i] else 0 for i in range(numcat)]
        self.gsave = -2

    def init2(self, numcat: int, count, val) -> None:
        """Order the categories: empty ones first, then by ``val`` (stable)."""
        self.maxc = numcat
        val = list(val)
        gray = [0] * numcat
        k = 1 if count[0] == 0 else 0
        for i in range(1, numcat):
            if count[i] == 0:
                for j in range(i - 1, k - 1, -1):
                    gray[j + 1] = gray[j]
                    val[j + 1] = val[j]
                gray[k] = i
                k += 1
            else:
                temp = val[i]
                j = i - 1
                while j >= k and val[j] > temp:
                    gray[j + 1] = gray[j]
                    val[j + 1] = val[j]
                    j -= 1
                val[j + 1] = temp
                gray[j + 1] = i
        self.gray = gray
        self.gsave = k - 1

    def next(self) -> int:
        if self.gsave > -2:
            self.gsave += 1
            return self.gray[self.gsave] if self.gsave < self.maxc else self.maxc
        gray = self.gray
        for i in range(self.maxc - 1):
            if gray[i] == 1:
                gray[i] = 2
                return i
            elif gray[i] == 2:
                gray[i] = 1
        return self.maxc


# ---------------------------------------------------------------------------
# Splitting methods
# ---------------------------------------------------------------------------


class _Anova:
    """anova.c: regression trees (sum of squares)."""

    num_resp = 1

    def __init__(self, y: np.ndarray, wt: np.ndarray):
        self.gray = _GrayCode()

    def eval(self, y: np.ndarray, wt: np.ndarray):
        twt = _seqsum(wt)
        mean = _seqsum(y * wt) / twt
        ss = _seqsum((y - mean) ** 2 * wt)
        return np.array([mean]), ss

    def error(self, y: float, response: np.ndarray) -> float:
        temp = y - response[0]
        return temp * temp

    def choose(self, y, x, wt, nclass, edge, myrisk):
        n = y.size
        right_wt = _seqsum(wt)
        right_sum = _seqsum(y * wt)
        grandmean = right_sum / right_wt
        if nclass == 0:
            temp = (y - grandmean) * wt
            left_sum = np.cumsum(temp)
            right_sum = -left_sum
            left_wt = np.cumsum(wt)
            right_wt_v = _running_minus(right_wt, wt)
            i = np.arange(n - edge) if n - edge > 0 else np.arange(0)
            if i.size == 0:
                return 0.0, 0.0, None
            left_n = i + 1
            valid = (x[i + 1] != x[i]) & (left_n >= edge)
            with np.errstate(divide="ignore", invalid="ignore"):
                val = (
                    left_sum[i] * left_sum[i] / left_wt[i]
                    + right_sum[i] * right_sum[i] / right_wt_v[i]
                )
            val = np.where(valid, val, -np.inf)
            where = int(np.argmax(val))
            best = val[where] if val[where] > 0 else 0.0
            if best > 0:
                direction = LEFT if left_sum[where] < right_sum[where] else RIGHT
                return best / myrisk, (x[where] + x[where + 1]) / 2, direction
            return best / myrisk, 0.0, None

        # categorical predictor: rank the classes by their mean y value
        codes = x.astype(int) - 1
        sums = _bincount_seq(codes, (y - grandmean) * wt, nclass)
        countn = np.bincount(codes, minlength=nclass)
        wts = _bincount_seq(codes, wt, nclass)
        tsplit = [0] * nclass
        mean = [0.0] * nclass
        for i in range(nclass):
            if countn[i] > 0:
                tsplit[i] = RIGHT
                mean[i] = sums[i] / wts[i]
        self.gray.init2(nclass, countn, mean)
        left_wt = 0.0
        left_sum = 0.0
        right_sum = 0.0
        left_n = 0
        right_n = n
        best = 0.0
        csplit = None
        while (j := self.gray.next()) < nclass:
            tsplit[j] = LEFT
            left_n += countn[j]
            right_n -= countn[j]
            left_wt += wts[j]
            right_wt -= wts[j]
            left_sum += sums[j]
            right_sum -= sums[j]
            if left_n >= edge and right_n >= edge:
                temp = left_sum * left_sum / left_wt + right_sum * right_sum / right_wt
                if temp > best:
                    best = temp
                    if (left_sum / left_wt) > (right_sum / right_wt):
                        csplit = [-t for t in tsplit]
                    else:
                        csplit = list(tsplit)
        return best / myrisk, 0.0, csplit


class _Gini:
    """gini.c: classification trees with priors, a loss matrix, gini/information."""

    def __init__(self, y, wt, nclass, prior, loss, split):
        self.nclass = nclass
        self.num_resp = 2 + nclass
        self.information = split == 2
        self.gray = _GrayCode()
        freq = np.zeros(nclass)
        temp = 0.0
        for j, w in zip((y - 1).astype(int), wt):
            freq[j] += w
            temp += w
        freq = freq / temp
        # loss[i, j]: true class i, predicted class j (rows = true class)
        self.loss = np.asarray(loss, dtype=float)
        prior = [float(p) for p in prior]
        aprior = [0.0] * nclass
        temp = 0.0
        for i in range(nclass):
            for j in range(nclass):
                temp += self.loss[i, j] * prior[i]
                aprior[i] += self.loss[i, j] * prior[i]
        for i in range(nclass):
            if freq[i] > 0:
                prior[i] /= freq[i]
                aprior[i] /= temp * freq[i]
        self.prior = np.array(prior)
        self.aprior = np.array(aprior)

    def impurity(self, p):
        if self.information:
            p = np.asarray(p, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.where(p == 0, 0.0, -p * np.log(np.where(p == 0, 1.0, p)))
            return out if out.ndim else float(out)
        return p * (1.0 - p)

    def eval(self, y, wt):
        nc = self.nclass
        yi = (y - 1).astype(int)
        freq = _bincount_seq(yi, wt, nc)
        prob = _seqsum(wt * self.prior[yi])
        best = 0
        dev = 0.0
        for i in range(nc):
            temp = 0.0
            for j in range(nc):
                temp += freq[j] * self.loss[j, i] * self.prior[j]
            if i == 0 or temp < dev:
                best = i
                dev = temp
        value = np.concatenate(([best + 1], freq, [prob]))
        return value, dev

    def error(self, y, response):
        # ginipred.c indexes the loss matrix as loss[i * numclass + j] with i
        # the true and j the predicted class, which is L[j, i] -- the transpose
        # of what ginidev uses. Kept as is so cross-validated errors match rpart
        # (it only matters for an asymmetric loss, e.g. cost and margin).
        i = int(y) - 1
        j = int(response[0]) - 1
        return self.prior[i] * self.loss[j, i]

    def _split_ss(self, left, right, lwt, rwt):
        """Weighted impurity of a split and the class 'means' used for direction."""
        temp = 0.0
        lmean = 0.0
        rmean = 0.0
        for j in range(self.nclass):
            p = self.aprior[j] * left[j] / lwt
            temp = temp + lwt * self.impurity(p)
            lmean = lmean + p * j
            p = self.aprior[j] * right[j] / rwt
            temp = temp + rwt * self.impurity(p)
            rmean = rmean + p * j
        return temp, lmean, rmean

    def choose(self, y, x, wt, numcat, edge, my_risk):
        nc = self.nclass
        ap = self.aprior
        n = y.size
        yi = (y - 1).astype(int)
        right = _bincount_seq(yi, wt, nc)
        rwt = _seqsum(ap[yi] * wt)
        total_ss = 0.0
        for i in range(nc):
            temp = ap[i] * right[i] / rwt
            total_ss += rwt * self.impurity(temp)
        best = total_ss

        if numcat == 0:
            if n - edge <= 0:
                return 0.0, 0.0, None
            a = ap[yi] * wt
            rwt_v = _running_minus(rwt, a)
            lwt_v = np.cumsum(a)
            left_v = []
            right_v = []
            for j in range(nc):
                wj = np.where(yi == j, wt, 0.0)
                left_v.append(np.cumsum(wj))
                right_v.append(_running_minus(right[j], wj))
            i = np.arange(n - edge)
            ltot = i + 1
            valid = (ltot >= edge) & (x[i + 1] != x[i])
            with np.errstate(divide="ignore", invalid="ignore"):
                temp, lmean, rmean = self._split_ss(
                    [lv[i] for lv in left_v],
                    [rv[i] for rv in right_v],
                    lwt_v[i],
                    rwt_v[i],
                )
            temp = np.where(valid, temp, np.inf)
            where = int(np.argmin(temp))
            if temp[where] < best:
                best = temp[where]
                direction = LEFT if lmean[where] < rmean[where] else RIGHT
                improve = total_ss - best
                if improve > 0:
                    return improve, (x[where] + x[where + 1]) / 2, direction
                return improve, 0.0, None
            return total_ss - best, 0.0, None

        # categorical: collapse the data into a numclass x numcat table
        codes = x.astype(int) - 1
        awt = _bincount_seq(codes, ap[yi] * wt, numcat)
        countn = np.bincount(codes, minlength=numcat)
        ccnt = _bincount_seq(yi * numcat + codes, wt, nc * numcat).reshape(nc, numcat)
        tsplit = [0] * numcat
        rate = [0.0] * numcat
        for i in range(numcat):
            if awt[i] != 0:
                rate[i] = ccnt[0, i] / awt[i]
                tsplit[i] = RIGHT
        if nc == 2:
            self.gray.init2(numcat, countn, rate)
        else:
            self.gray.init1(numcat, countn)
        left = np.zeros(nc)
        right = right.copy()
        lwt = 0.0
        rtot = n
        ltot = 0
        csplit = None
        while (i := self.gray.next()) < numcat:
            if tsplit[i] == LEFT:
                tsplit[i] = RIGHT
                rwt += awt[i]
                lwt -= awt[i]
                rtot += countn[i]
                ltot -= countn[i]
                for j in range(nc):
                    right[j] += ccnt[j, i]
                    left[j] -= ccnt[j, i]
            else:
                tsplit[i] = LEFT
                rwt -= awt[i]
                lwt += awt[i]
                rtot -= countn[i]
                ltot += countn[i]
                for j in range(nc):
                    right[j] -= ccnt[j, i]
                    left[j] += ccnt[j, i]
            if ltot >= edge and rtot >= edge:
                temp, lmean, rmean = self._split_ss(left, right, lwt, rwt)
                if temp < best:
                    best = temp
                    if lmean < rmean:
                        csplit = list(tsplit)
                    else:
                        csplit = [-t for t in tsplit]
        return total_ss - best, 0.0, csplit


# ---------------------------------------------------------------------------
# Tree structures
# ---------------------------------------------------------------------------


@dataclass
class Split:
    var: int
    improve: float
    count: int
    ncat: int  # 0 for continuous
    spoint: float = 0.0
    direction: int = LEFT  # continuous: LEFT means "x < spoint goes left"
    csplit: list | None = None  # categorical: LEFT/RIGHT/0 per level
    adj: float = 0.0

    def go(self, xval: float) -> int:
        """Direction (LEFT/RIGHT, or 0 when undecided) for one value."""
        if not math.isfinite(xval):
            return 0
        if self.ncat == 0:
            return self.direction if xval < self.spoint else -self.direction
        return self.csplit[int(xval) - 1]


@dataclass
class Node:
    risk: float = 0.0
    complexity: float = 0.0
    sum_wt: float = 0.0
    num_obs: int = 0
    response: np.ndarray | None = None
    primary: list = field(default_factory=list)
    surrogate: list = field(default_factory=list)
    lastsurrogate: int = 0
    idx: np.ndarray | None = None  # observations, kept for lazy surrogates
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def _insert_split(lst: list, new: Split, maxn: int) -> None:
    """insert_split.c: keep the ``maxn`` best splits, sorted by ``improve``."""
    if not lst:
        lst.append(new)
        return
    if maxn < 2:
        if new.improve > lst[0].improve:
            lst[0] = new
        return
    pos = len(lst)
    for k, s in enumerate(lst):
        if new.improve > s.improve:
            pos = k
            break
    if len(lst) == maxn and pos == len(lst):
        return
    lst.insert(pos, new)
    del lst[maxn:]


# ---------------------------------------------------------------------------
# The fitter
# ---------------------------------------------------------------------------


class _Builder:
    def __init__(self, X, ncat, y, wt, method, control):
        self.X = X
        self.ncat = list(ncat)
        self.y = y
        self.wt = wt
        self.method = method
        self.n, self.nvar = X.shape
        self.min_split = int(control["minsplit"])
        self.min_node = max(1, int(control["minbucket"]))
        self.complexity = float(control["cp"])
        self.maxpri = max(1, int(control["maxcompete"]) + 1)
        self.maxsur = int(control["maxsurrogate"])
        self.usesurrogate = int(control["usesurrogate"])
        self.sur_agree = int(control["surrogatestyle"])
        self.maxnode = int(2 ** int(control["maxdepth"])) - 1
        self.iscale = 0.0
        self.alpha = 0.0
        self.lazy_surrogates = False
        self.which = np.ones(self.n, dtype=np.int64)
        # sort order of each continuous variable, once and for all, with the
        # same (unstable) quicksort as rpart so tied values line up the same way
        self.order = [
            _mysort(X[:, v]) if self.ncat[v] == 0 else None for v in range(self.nvar)
        ]
        self.rank = [
            np.argsort(o, kind="stable") if o is not None else None for o in self.order
        ]

    # -- helpers ------------------------------------------------------------
    def _sorted(self, idx: np.ndarray, v: int) -> np.ndarray:
        """Observations in ``idx`` in the (fixed) sort order of variable ``v``."""
        if self.ncat[v] > 0:
            return idx
        return idx[np.argsort(self.rank[v][idx], kind="stable")]

    def _eval(self, node: Node, idx: np.ndarray) -> None:
        o = self._sorted(idx, 0)
        node.response, node.risk = self.method.eval(self.y[o], self.wt[o])
        node.num_obs = idx.size
        node.sum_wt = _seqsum(self.wt[o])

    # -- partition.c --------------------------------------------------------
    def partition(self, nodenum: int, me: Node, idx: np.ndarray):
        """Returns (number of splits, risk of the subtree)."""
        if nodenum > 1:
            self._eval(me, idx)
            tempcp = me.risk
            if tempcp > me.complexity:
                tempcp = me.complexity
        else:
            tempcp = me.risk

        if me.num_obs < self.min_split or tempcp <= self.alpha or nodenum > self.maxnode:
            me.complexity = self.alpha
            me.left = me.right = None
            me.primary, me.surrogate = [], []
            self.which[idx] = nodenum
            return 0, me.risk

        self.bsplit(me, idx)
        if not me.primary:
            me.complexity = self.alpha
            me.left = me.right = None
            me.surrogate = []
            self.which[idx] = nodenum
            return 0, me.risk

        if self.maxsur > 0:
            if self.lazy_surrogates:
                # cross-validation trees only need surrogates when a held-out
                # row cannot follow the primary split; work them out then
                me.surrogate = None
                me.idx = idx
            else:
                self.surrogate(me, idx)
        else:
            me.surrogate = []
        left_idx, right_idx = self.nodesplit(me, nodenum, idx)

        me.left = Node(complexity=tempcp - self.alpha)
        left_split, left_risk = self.partition(2 * nodenum, me.left, left_idx)

        tempcp = (me.risk - left_risk) / (left_split + 1)
        tempcp2 = me.risk - me.left.risk
        if tempcp < tempcp2:
            tempcp = tempcp2
        if tempcp > me.complexity:
            tempcp = me.complexity
        me.right = Node(complexity=tempcp - self.alpha)
        right_split, right_risk = self.partition(1 + 2 * nodenum, me.right, right_idx)

        tempcp = (me.risk - (left_risk + right_risk)) / (left_split + right_split + 1)
        if me.right.complexity > me.left.complexity:
            if tempcp > me.left.complexity:
                left_risk = me.left.risk
                left_split = 0
                tempcp = (me.risk - (left_risk + right_risk)) / (
                    left_split + right_split + 1
                )
                if tempcp > me.right.complexity:
                    right_risk = me.right.risk
                    right_split = 0
        elif tempcp > me.right.complexity:
            right_split = 0
            right_risk = me.right.risk
            tempcp = (me.risk - (left_risk + right_risk)) / (left_split + right_split + 1)
            if tempcp > me.left.complexity:
                left_risk = me.left.risk
                left_split = 0

        me.complexity = (me.risk - (left_risk + right_risk)) / (
            left_split + right_split + 1
        )
        if me.complexity <= self.alpha:
            # all was in vain: this node doesn't split after all
            me.left = me.right = None
            me.primary, me.surrogate = [], []
            self.which[idx] = nodenum
            return 0, me.risk
        return left_split + right_split + 1, left_risk + right_risk

    # -- bsplit.c -----------------------------------------------------------
    def bsplit(self, me: Node, idx: np.ndarray) -> None:
        me.primary = []
        for i in range(self.nvar):
            nc = self.ncat[i]
            o = self._sorted(idx, i)
            o = o[self.wt[o] > 0]
            k = o.size
            if k == 0:
                continue
            xtemp = self.X[o, i]
            if nc == 0 and xtemp[0] == xtemp[k - 1]:
                continue
            improve, split, csplit = self.method.choose(
                self.y[o], xtemp, self.wt[o], nc, self.min_node, me.risk
            )
            if improve > self.iscale:
                self.iscale = improve
            if improve > self.iscale * 1e-10:
                if nc == 0:
                    s = Split(var=i, improve=improve, count=k, ncat=0, spoint=split,
                              direction=csplit)
                else:
                    s = Split(var=i, improve=improve, count=k, ncat=nc, csplit=csplit)
                _insert_split(me.primary, s, self.maxpri)

    # -- surrogate.c --------------------------------------------------------
    def surrogate(self, me: Node, idx: np.ndarray) -> None:
        prim = me.primary[0]
        var = prim.var
        xv = self.X[idx, var]
        if prim.ncat == 0:
            tempy = np.where(xv < prim.spoint, prim.direction, -prim.direction)
        else:
            cs = np.asarray(prim.csplit)
            tempy = cs[xv.astype(int) - 1]
        ty = np.zeros(self.n, dtype=np.int64)
        ty[idx] = tempy
        o = self._sorted(idx, var)
        lcount = _seqsum(np.where(ty[o] == LEFT, self.wt[o], 0.0))
        rcount = _seqsum(np.where(ty[o] == RIGHT, self.wt[o], 0.0))
        if lcount < rcount:
            me.lastsurrogate = RIGHT
        elif lcount > rcount:
            me.lastsurrogate = LEFT
        else:
            me.lastsurrogate = 0

        me.surrogate = []
        for i in range(self.nvar):
            if i == var:
                continue
            res = self.choose_surg(self._sorted(idx, i), ty, i, lcount, rcount)
            if res is None:
                continue
            agreement, adj, split, csplit = res
            if adj <= 1e-10:
                continue
            nc = self.ncat[i]
            if nc == 0:
                s = Split(var=i, improve=agreement, count=0, ncat=0, spoint=split,
                          direction=csplit, adj=adj)
            else:
                s = Split(var=i, improve=agreement, count=0, ncat=nc, csplit=csplit,
                          adj=adj)
            _insert_split(me.surrogate, s, self.maxsur)

    # -- choose_surg.c ------------------------------------------------------
    def choose_surg(self, order, ty, var, tleft, tright):
        ncat = self.ncat[var]
        y = ty[order]
        w = self.wt[order]
        x = self.X[order, var]
        pos = w > 0
        isL = y == LEFT
        isR = y == RIGHT
        if ncat == 0:
            n = order.size
            # backwards pass: me sending everyone to the left
            ll0 = int(np.sum(isL & pos))
            rl0 = int(np.sum(isR & pos))
            wl = np.where(isL, w, 0.0)
            wr = np.where(isR, w, 0.0)
            llwt0 = _seqsum(wl[::-1])
            rlwt0 = _seqsum(wr[::-1])
            agree = llwt0 if llwt0 > rlwt0 else rlwt0
            majority = agree
            total_wt = llwt0 + rlwt0
            csplit = LEFT
            split = x[0] if n else 0.0
            success = False
            # state *before* processing position i
            lr = np.concatenate(([0], np.cumsum(isL & pos)))[:-1]
            rr = np.concatenate(([0], np.cumsum(isR & pos)))[:-1]
            ll = ll0 - lr
            rl = rl0 - rr
            llwt = np.concatenate(([llwt0], _running_minus(llwt0, wl)))[:-1]
            rlwt = np.concatenate(([rlwt0], _running_minus(rlwt0, wr)))[:-1]
            lrwt = np.concatenate(([0.0], np.cumsum(wl)))[:-1]
            rrwt = np.concatenate(([0.0], np.cumsum(wr)))[:-1]
            stop = np.nonzero((ll + rl) < 2)[0]
            last = stop[0] if stop.size else n
            if last > 1:
                ii = np.arange(1, last)
                cand = ii[((lr[ii] + rr[ii]) >= 2) & (x[ii] != x[ii - 1])]
                A = llwt[cand] + rrwt[cand]
                B = lrwt[cand] + rlwt[cand]
                keep = (A > majority) | (B > majority)
                for c, a, b in zip(cand[keep], A[keep], B[keep]):
                    if a > agree:
                        success = True
                        agree = a
                        csplit = RIGHT
                        split = (x[c] + x[c - 1]) / 2
                    elif b > agree:
                        success = True
                        agree = b
                        csplit = LEFT
                        split = (x[c] + x[c - 1]) / 2
        else:
            codes = x.astype(int) - 1
            left = np.bincount(codes[isL & pos], minlength=ncat)
            right = np.bincount(codes[isR & pos], minlength=ncat)
            lwt = _bincount_seq(codes, np.where(isL, w, 0.0), ncat)
            rwt = _bincount_seq(codes, np.where(isR, w, 0.0), ncat)
            llwt = _seqsum(lwt)
            rrwt = _seqsum(rwt)
            if llwt > rrwt:
                defdir, majority = LEFT, llwt
            else:
                defdir, majority = RIGHT, rrwt
            total_wt = llwt + rrwt
            agree = 0.0
            lcnt = rcnt = 0
            csplit = [0] * ncat
            for i in range(ncat):
                if left[i] == 0 and right[i] == 0:
                    csplit[i] = 0
                elif lwt[i] < rwt[i] or (lwt[i] == rwt[i] and defdir == RIGHT):
                    agree += rwt[i]
                    csplit[i] = RIGHT
                    lcnt += left[i]
                    rcnt += right[i]
                else:
                    agree += lwt[i]
                    csplit[i] = LEFT
                    lcnt += right[i]
                    rcnt += left[i]
            success = lcnt > 1 and rcnt > 1
            split = 0.0

        if not success:
            return None
        if self.sur_agree == 0:
            total_wt = tleft + tright
            majority = tleft if tleft > tright else tright
        agreement = agree / total_wt
        majority /= total_wt
        adj = (agreement - majority) / (1.0 - majority)
        return agreement, adj, split, csplit

    # -- nodesplit.c --------------------------------------------------------
    def nodesplit(self, me: Node, nodenum: int, idx: np.ndarray):
        """Send the node's observations left or right by the primary split.

        Training data has no missing values, so surrogates are never needed
        here. As in the C code, an observation whose category has no direction
        (a level only seen with zero weight) goes nowhere and stays behind.
        """
        prim = me.primary[0]
        xv = self.X[idx, prim.var]
        if prim.ncat == 0:
            d = np.where(xv < prim.spoint, prim.direction, -prim.direction)
        else:
            d = np.asarray(prim.csplit)[xv.astype(int) - 1]
        left_idx = idx[d == LEFT]
        right_idx = idx[d == RIGHT]
        self.which[left_idx] = 2 * nodenum
        self.which[right_idx] = 2 * nodenum + 1
        return left_idx, right_idx

    # -- branch.c (used by cross-validation) ---------------------------------
    def branch(self, tree: Node, obs: int):
        if tree.left is None:
            return None
        prim = tree.primary[0]
        dirn = prim.go(self.X[obs, prim.var])
        if dirn == 0:
            if self.usesurrogate == 0:
                return None
            if tree.surrogate is None:
                self.surrogate(tree, tree.idx)
            for s in tree.surrogate:
                dirn = s.go(self.X[obs, s.var])
                if dirn:
                    break
            if dirn == 0:
                if self.usesurrogate < 2:
                    return None
                dirn = tree.lastsurrogate
        return tree.left if dirn == LEFT else tree.right


def _fix_cp(me: Node, parent_cp: float) -> None:
    if me.complexity > parent_cp:
        me.complexity = parent_cp
    if me.left is not None:
        _fix_cp(me.left, me.complexity)
        _fix_cp(me.right, me.complexity)


class _CpTable:
    """The linked cp list of make_cp_list.c / make_cp_table.c, as a Python list
    sorted from the largest cp (the root) to the smallest."""

    def __init__(self, root: Node):
        self.rows = [dict(cp=root.complexity, risk=root.risk, nsplit=0, xrisk=0.0,
                          xstd=0.0)]

    def make_cp_list(self, me: Node, parent: float, alpha: float) -> None:
        if me.complexity > parent:
            me.complexity = parent
        me_cp = me.complexity
        if me_cp < alpha:
            me_cp = alpha
        if me.left is not None:
            self.make_cp_list(me.left, me_cp, alpha)
            self.make_cp_list(me.right, me_cp, alpha)
        if me_cp < parent:
            pos = len(self.rows)
            for k, row in enumerate(self.rows):
                if me_cp == row["cp"]:
                    return
                if me_cp > row["cp"]:
                    pos = k
                    break
            self.rows.insert(pos, dict(cp=me_cp, risk=0.0, nsplit=0, xrisk=0.0,
                                       xstd=0.0))

    def make_cp_table(self, me: Node, parent: float, nsplit: int) -> int:
        if me.left is not None:
            self.make_cp_table(me.left, me.complexity, 0)
            k = self.make_cp_table(me.right, me.complexity, nsplit + 1)
        else:
            k = len(self.rows) - 1
        while self.rows[k]["cp"] < parent:
            self.rows[k]["risk"] += me.risk
            self.rows[k]["nsplit"] += nsplit
            k -= 1
        return k


# ---------------------------------------------------------------------------
# R's random number generator (for the cross-validation groups)
# ---------------------------------------------------------------------------


class RRandom:
    """Mersenne-Twister ``unif_rand`` as seeded by R's ``set.seed`` (R >= 3.6)."""

    def __init__(self, seed: int):
        s = seed & 0xFFFFFFFF
        for _ in range(50):
            s = (69069 * s + 1) & 0xFFFFFFFF
        state = []
        for _ in range(625):
            s = (69069 * s + 1) & 0xFFFFFFFF
            state.append(s)
        key = np.array(state[1:], dtype=np.uint32)
        self.bitgen = np.random.MT19937()
        self.bitgen.state = {
            "bit_generator": "MT19937",
            "state": {"key": key, "pos": 624},
        }

    def unif_rand(self) -> float:
        value = int(self.bitgen.random_raw()) * 2.3283064365386963e-10
        if value <= 0.0:
            return 0.5 * 2.328306437080797e-10
        if 1.0 - value <= 0.0:
            return 1.0 - 0.5 * 2.328306437080797e-10
        return value

    def rbits(self, bits: int) -> float:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return float(v & ((1 << bits) - 1))

    def unif_index(self, dn: float) -> float:
        if dn <= 0:
            return 0.0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self.rbits(bits)
            if dn > dv:
                return dv


def r_sample_xgroups(nobs: int, xval: int, seed: int | None = None,
                     rng: "RRandom | None" = None) -> np.ndarray:
    """``set.seed(seed); sample(rep(1:xval, length.out = nobs), nobs)``.

    Pass ``rng`` instead of ``seed`` to continue an existing R random stream.
    """
    if rng is None:
        rng = RRandom(seed)
    pool = np.resize(np.arange(1, xval + 1), nobs)
    x = list(range(nobs))
    out = np.empty(nobs, dtype=np.int64)
    n = nobs
    for i in range(nobs):
        j = int(rng.unif_index(n))
        out[i] = pool[x[j]]
        n -= 1
        x[j] = x[n]
    return out


def r_sample_int_replace(k: int, n: int, rng: "RRandom") -> np.ndarray:
    """``sample(k, n, replace = TRUE)`` drawing from ``rng`` (values 1..k)."""
    return np.array([int(rng.unif_index(k)) + 1 for _ in range(n)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Fitted object
# ---------------------------------------------------------------------------


@dataclass
class RPartFit:
    """A fitted tree in the shape of R's ``rpart`` object.

    ``frame`` rows are the nodes in pre-order (as in R); ``splits`` rows are,
    per internal node, the primary split, its competitors, then surrogates.
    """

    root: Node
    method: str  # "anova" or "class"
    var_names: list
    ncat: list
    xlevels: dict
    ylevels: list | None
    cptable: np.ndarray  # columns: CP, nsplit, rel error[, xerror, xstd]
    cp_scale: float
    alpha: float  # absolute (unscaled) complexity threshold
    prior: np.ndarray | None = None  # user prior (R's parms$prior)
    counts: np.ndarray | None = None  # overall class weights (R's init$counts)
    loss: np.ndarray | None = None
    usesurrogate: int = 2
    frame: list = field(default_factory=list)
    where: np.ndarray | None = None
    yprob_adjust: tuple | None = None  # (prior, bp) for adjusted probabilities

    # -- frame -------------------------------------------------------------
    def build_frame(self) -> None:
        rows = []

        def walk(node: Node, nid: int) -> None:
            leaf = node.complexity <= self.alpha or node.left is None
            row = dict(
                id=nid,
                node=node,
                var="<leaf>" if leaf else self.var_names[node.primary[0].var],
                n=node.num_obs,
                wt=node.sum_wt,
                dev=node.risk,
                yval=float(node.response[0]),
                complexity=node.complexity * self.cp_scale,
                ncompete=0 if leaf else max(0, len(node.primary) - 1),
                nsurrogate=0 if leaf else len(node.surrogate),
                leaf=leaf,
            )
            if self.method == "class":
                row["yval2"] = self._yval2(node)
            rows.append(row)
            if not leaf:
                walk(node.left, 2 * nid)
                walk(node.right, 2 * nid + 1)

        walk(self.root, 1)
        self.frame = rows

    def _yval2(self, node: Node) -> np.ndarray:
        nclass = len(self.ylevels)
        resp = node.response
        counts = resp[1 : 1 + nclass]
        temp = counts * (self.prior / np.maximum(1, self.counts))
        yprob = temp / _seqsum(temp)
        if self.yprob_adjust is not None:
            prior, bp = self.yprob_adjust
            p = yprob[0]
            p1 = bp * (p - p * prior) / (prior - p * prior + bp * p - prior * bp)
            yprob = np.array([p1, 1 - p1])
        nodeprob = resp[nclass + 1] / self.total_wt
        return np.concatenate(([resp[0]], counts, yprob, [nodeprob]))

    total_wt: float = 1.0

    # -- splits table ------------------------------------------------------
    def splits_table(self) -> list[dict]:
        out = []
        for row in self.frame:
            if row["leaf"]:
                continue
            node = row["node"]
            for kind, lst in (("primary", node.primary), ("surrogate", node.surrogate)):
                for s in lst:
                    out.append(
                        dict(
                            node=row["id"],
                            type=kind,
                            var=self.var_names[s.var],
                            count=s.count,
                            ncat=s.ncat if s.ncat else s.direction,
                            improve=s.improve,
                            index=s.spoint if s.ncat == 0 else None,
                            adj=s.adj if kind == "surrogate" else 0.0,
                            split=s,
                        )
                    )
        return out

    # -- importance.R --------------------------------------------------------
    def variable_importance(self) -> dict:
        imp: dict[str, float] = {}
        for row in self.frame:
            if row["leaf"]:
                continue
            node = row["node"]
            prim = node.primary[0]
            scaled = prim.improve * row["dev"] if self.method == "anova" else prim.improve
            name = self.var_names[prim.var]
            vals = [(name, scaled)]
            for s in node.surrogate:
                vals.append((self.var_names[s.var], scaled * s.adj))
            for k, v in vals:
                imp[k] = imp.get(k, 0.0) + v
        # R: sort(tapply(...), decreasing = TRUE); tapply orders names first
        items = sorted(imp.items(), key=lambda kv: kv[0])
        items = sorted(items, key=lambda kv: -kv[1])
        return dict(items)

    # -- prune.rpart.R / snip.rpart.R -----------------------------------------
    def prune(self, cp: float) -> "RPartFit":
        """Snip every internal node with complexity <= cp (relative scale)."""
        import copy

        new = copy.copy(self)
        new.root = copy.deepcopy(self.root)
        toss = [
            r["id"] for r in self.frame if (not r["leaf"]) and r["complexity"] <= cp
        ]
        if not toss:
            return self

        def snip(node: Node, nid: int) -> None:
            if node.left is None:
                return
            if node.complexity * self.cp_scale <= cp:
                node.left = node.right = None
                node.primary, node.surrogate = [], []
                return
            snip(node.left, 2 * nid)
            snip(node.right, 2 * nid + 1)

        snip(new.root, 1)
        cpt = self.cptable
        temp = np.maximum(cpt[:, 0], cp)
        _, first = np.unique(temp, return_index=True)
        keep = np.sort(first)
        new.cptable = cpt[keep].copy()
        new.cptable[-1, 0] = cp
        new.build_frame()
        ids = {r["id"] for r in new.frame}
        where_ids = np.array([self.frame[k]["id"] for k in self.where])
        for i in range(where_ids.size):
            while where_ids[i] not in ids:
                where_ids[i] //= 2
        pos = {r["id"]: k for k, r in enumerate(new.frame)}
        new.where = np.array([pos[i] for i in where_ids])
        return new

    # -- pred.rpart.R / pred_rpart.c ------------------------------------------
    def predict_where(self, X: np.ndarray) -> np.ndarray:
        """Frame row of the node each row of ``X`` ends up in."""
        pos = {r["id"]: k for k, r in enumerate(self.frame)}
        nobs_child = {r["id"]: r["n"] for r in self.frame}
        out = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            nid = 1
            while True:
                row = self.frame[pos[nid]]
                if row["leaf"]:
                    break
                node = row["node"]
                dirn = node.primary[0].go(X[i, node.primary[0].var])
                if dirn == 0 and self.usesurrogate > 0:
                    for s in node.surrogate:
                        dirn = s.go(X[i, s.var])
                        if dirn:
                            break
                if dirn == 0 and self.usesurrogate > 1:
                    lc = nobs_child[2 * nid]
                    rc = nobs_child[2 * nid + 1]
                    if lc != rc:
                        dirn = LEFT if lc > rc else RIGHT
                if dirn == 0:
                    break
                nid = 2 * nid if dirn == LEFT else 2 * nid + 1
            out[i] = pos[nid]
        return out


def rpart_fit(
    X: np.ndarray,
    ncat: list,
    y: np.ndarray,
    wt: np.ndarray,
    method: str,
    var_names: list,
    xlevels: dict,
    ylevels: list | None = None,
    prior=None,
    loss=None,
    split: int = 1,
    minsplit: int = 20,
    minbucket: int | None = None,
    cp: float = 0.01,
    maxcompete: int = 4,
    maxsurrogate: int = 5,
    usesurrogate: int = 2,
    surrogatestyle: int = 0,
    maxdepth: int = 30,
    xval: int = 10,
    xgroups: np.ndarray | None = None,
) -> RPartFit:
    """Fit a tree like ``rpart::rpart``.

    ``X`` holds continuous values, or 1-based level codes for categorical
    columns (``ncat[j]`` = number of levels, 0 for continuous). For
    ``method="class"`` ``y`` holds 1-based class codes.
    """
    if minbucket is None:
        minbucket = int(round(minsplit / 3))
    if maxdepth > 30:
        raise ValueError("Maximum depth is 30")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    wt = np.asarray(wt, dtype=float)
    if np.any(wt < 0):
        raise ValueError("negative weights not allowed")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Training data must not contain missing values")
    n = X.shape[0]

    if method == "class":
        nclass = len(ylevels)
        yi = y.astype(int) - 1
        counts = _bincount_seq(yi, wt, nclass)
        if prior is None:
            prior = counts / _seqsum(counts)
        prior = np.asarray(prior, dtype=float)
        if loss is None:
            loss = 1.0 - np.eye(nclass)
        loss = np.asarray(loss, dtype=float)
        meth = _Gini(y, wt, nclass, prior, loss, split)
    elif method == "anova":
        counts = None
        meth = _Anova(y, wt)
    else:
        raise ValueError("method must be 'anova' or 'class'")

    control = dict(minsplit=minsplit, minbucket=minbucket, cp=cp,
                   maxcompete=maxcompete, maxsurrogate=maxsurrogate,
                   usesurrogate=usesurrogate, surrogatestyle=surrogatestyle,
                   maxdepth=maxdepth)
    b = _Builder(X, ncat, y, wt, meth, control)

    all_idx = np.arange(n)
    tree = Node()
    tree.response, tree.risk = meth.eval(y, wt)
    tree.num_obs = n
    tree.sum_wt = _seqsum(wt)
    tree.complexity = tree.risk
    b.alpha = b.complexity * tree.risk
    b.partition(1, tree, all_idx)

    cpt = _CpTable(tree)
    nx = int(xval) if xval else 0
    if tree.left is not None:
        cpt.make_cp_list(tree, tree.complexity, b.alpha)
        cpt.make_cp_table(tree, tree.complexity, 0)
        if nx > 1:
            _xval(b, nx, cpt, xgroups, tree)
    else:
        nx = 0

    scale = 1.0 / tree.risk
    cols = []
    for row in cpt.rows:
        r = [row["cp"] * scale, row["nsplit"], row["risk"] * scale]
        if nx > 1:
            r += [row["xrisk"] * scale, row["xstd"] * scale]
        cols.append(r)
    cptable = np.array(cols, dtype=float)

    fit = RPartFit(
        root=tree,
        method=method,
        var_names=list(var_names),
        ncat=list(ncat),
        xlevels=xlevels,
        ylevels=list(ylevels) if ylevels is not None else None,
        cptable=cptable,
        cp_scale=scale,
        alpha=b.alpha,
        prior=prior if method == "class" else None,
        counts=counts,
        loss=loss if method == "class" else None,
        usesurrogate=usesurrogate,
        total_wt=_seqsum(wt),
    )
    fit.build_frame()
    # 'where': nodes may have been trimmed after obs were sent down, so climb
    # up to the nearest node that exists
    pos = {r["id"]: k for k, r in enumerate(fit.frame)}
    where = np.empty(n, dtype=np.int64)
    for i in range(n):
        k = int(b.which[i])
        while k not in pos:
            k //= 2
        where[i] = pos[k]
    fit.where = where
    return fit


def _xval(b: _Builder, n_xval: int, cpt: _CpTable, x_grp: np.ndarray, tree: Node) -> None:
    """xval.c: fill in the cross-validated risk and its standard deviation."""
    alphasave = b.alpha
    rows = cpt.rows
    ncp = len(rows)
    cp = np.empty(ncp)
    cp[0] = 10 * rows[0]["cp"]
    for i in range(1, ncp):
        cp[i] = math.sqrt(rows[i - 1]["cp"] * rows[i]["cp"])
    total_wt = _seqsum(b.wt)
    old_wt = total_wt
    whichsave = b.which.copy()
    b.lazy_surrogates = True
    xrisk = [0.0] * ncp
    xstd = [0.0] * ncp
    for xgroup in range(1, n_xval + 1):
        train = np.nonzero(x_grp != xgroup)[0]
        test = np.nonzero(x_grp == xgroup)[0]
        temp = _seqsum(b.wt[train])
        cp *= temp / old_wt
        b.alpha *= temp / old_wt
        old_wt = temp
        xtree = Node()
        xtree.num_obs = train.size
        xtree.response, xtree.risk = b.method.eval(b.y[train], b.wt[train])
        xtree.complexity = xtree.risk
        b.partition(1, xtree, train)
        _fix_cp(xtree, xtree.complexity)
        for j in test:
            tnode = xtree
            for i in range(ncp):
                while cp[i] < tnode.complexity:
                    nxt = b.branch(tnode, j)
                    if nxt is None:
                        break
                    tnode = nxt
                xt = b.method.error(b.y[j], tnode.response)
                xrisk[i] += xt * b.wt[j]
                xstd[i] += xt * xt * b.wt[j]
    for i in range(ncp):
        rows[i]["xrisk"] = xrisk[i]
        rows[i]["xstd"] = math.sqrt(max(0.0, xstd[i] - xrisk[i] * xrisk[i] / total_wt))
    b.alpha = alphasave
    b.which = whichsave
    b.lazy_surrogates = False
