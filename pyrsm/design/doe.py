import itertools
import math

import numpy as np
import polars as pl

from pyrsm.multivariate._lm import r_qr_rank
from pyrsm.utils import expand_grid, polychoric_matrix


def _term_type(term: str) -> str:
    n = term.count(":") + 1
    return "main effect" if n == 1 else f"{n}-way interaction"


def _build_model_matrix(ff_coded, levels_list, interaction_pairs=None):
    """Build dummy-coded model matrix with intercept from integer-coded full factorial."""
    n = ff_coded.shape[0]
    cols = [np.ones(n)]

    factor_dummies = {}
    for i, k in enumerate(levels_list):
        dummies = []
        for j in range(1, k):
            col = (ff_coded[:, i] == j).astype(float)
            dummies.append(col)
            cols.append(col)
        factor_dummies[i] = dummies

    if interaction_pairs:
        for f1, f2 in interaction_pairs:
            for d1 in factor_dummies[f1]:
                for d2 in factor_dummies[f2]:
                    cols.append(d1 * d2)

    return np.column_stack(cols)


def _parse_interactions(int_str, df_names):
    """Parse interaction strings like 'price:food' into index pairs."""
    if not int_str:
        return []
    if isinstance(int_str, str):
        int_str = [s.strip() for s in int_str.split(",") if s.strip()]

    pairs = []
    for term in int_str:
        parts = [p.strip() for p in term.split(":")]
        if len(parts) == 2 and parts[0] in df_names and parts[1] in df_names:
            i1 = df_names.index(parts[0])
            i2 = df_names.index(parts[1])
            pairs.append((i1, i2))
    return pairs


def _integer_code(full_pl, df_dict, df_names):
    """Derive integer-coded array from expand_grid DataFrame.

    Maps each column's values to 0, 1, 2, ... based on the level order
    in df_dict, producing the same coding that a fullfact() call would.
    """
    n = full_pl.height
    n_factors = len(df_names)
    coded = np.zeros((n, n_factors), dtype=float)
    for j, name in enumerate(df_names):
        levels = df_dict[name]
        col = full_pl[name].to_list()
        level_map = {lev: idx for idx, lev in enumerate(levels)}
        coded[:, j] = [level_map[v] for v in col]
    return coded


def _is_balanced(selected, levels_list, ff_coded):
    """Check if the selected row indices form a balanced design."""
    n_trials = len(selected)
    for j, k in enumerate(levels_list):
        if n_trials % k != 0:
            return False
        target = n_trials // k
        for level in range(k):
            count = sum(1 for i in selected if int(ff_coded[i, j]) == level)
            if count != target:
                return False
    return True


def _can_balance(n_trials, levels_list):
    """Check if balance is theoretically possible for this trial count."""
    return all(n_trials % k == 0 for k in levels_list)


def _balanced_random_start(n_trials, levels_list, ff_coded, rng):
    """Generate a random balanced starting design via rejection sampling."""
    N = ff_coded.shape[0]
    for _ in range(10000):
        selected = sorted(rng.choice(N, size=n_trials, replace=False).tolist())
        if _is_balanced(selected, levels_list, ff_coded):
            return selected
    # Fallback: return any random selection
    return sorted(rng.choice(N, size=n_trials, replace=False).tolist())


def _federov_exchange(
    Xf, selected, N, levels_list, ff_coded, enforce_balance, max_iter=1000
):
    """Run Federov exchange algorithm with optional balance constraint.

    Uses the variance-exchange criterion:
    delta = (1 + d_j)(1 - d_i) + d_ij^2
    where d_j = xj' M^{-1} xj, d_i = xi' M^{-1} xi, d_ij = xi' M^{-1} xj.
    A swap improves D-criterion when delta > 1.
    """
    n_trials = len(selected)

    # Pre-compute all leverage values for candidates (vectorized)
    # d_j = Xf[j] @ M_inv @ Xf[j] for all j
    for _ in range(max_iter):
        Xd = Xf[selected]
        M = Xd.T @ Xd
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            break

        # Vectorized: H = Xf @ M_inv @ Xf.T, diagonal = leverage values
        XM = Xf @ M_inv  # (N, p)
        d_all = np.sum(XM * Xf, axis=1)  # (N,) leverage for every candidate

        not_selected = np.array(sorted(set(range(N)) - set(selected)))
        d_cand = d_all[not_selected]  # leverage for candidates

        best_delta = 1.0
        best_swap = None

        for i_pos in range(n_trials):
            i = selected[i_pos]
            d_i = d_all[i]

            if enforce_balance:
                # A swap preserves balance iff for every factor, the outgoing
                # and incoming rows have the same level
                same_levels = np.all(ff_coded[not_selected] == ff_coded[i], axis=1)
                eligible = np.where(same_levels)[0]
                if len(eligible) == 0:
                    continue
                eligible_global = not_selected[eligible]
                d_j_vec = d_cand[eligible]
            else:
                eligible_global = not_selected
                d_j_vec = d_cand

            # Vectorized cross-leverage: d_ij = xi @ M_inv @ xj for all j
            xi_M = XM[i]  # (p,)
            d_ij_vec = Xf[eligible_global] @ xi_M  # (k,)

            # delta = (1 + d_j)(1 - d_i) + d_ij^2
            delta_vec = (1 + d_j_vec) * (1 - d_i) + d_ij_vec**2

            max_idx = np.argmax(delta_vec)
            if delta_vec[max_idx] > best_delta:
                best_delta = delta_vec[max_idx]
                best_swap = (i_pos, eligible_global[max_idx])

        if best_swap is None:
            break

        i_pos, j = best_swap
        selected[i_pos] = j

    return selected


def _find_optimal_design(Xf, ff_coded, levels_list, n_trials, n_repeats=50, seed=None):
    """Find D-optimal design using Federov exchange (no balance enforcement).

    Matches R's AlgDesign::optFederov behavior: optimizes D-criterion freely
    without constraining to balanced designs. Reports actual balance status.
    """
    N = Xf.shape[0]

    best_det = -1
    best_selected = None

    for rep in range(n_repeats):
        rep_seed = seed + rep
        rng = np.random.default_rng(rep_seed)

        # Random starting design
        selected = sorted(rng.choice(N, size=n_trials, replace=False).tolist())

        # Run Federov exchange without balance enforcement
        selected = _federov_exchange(
            Xf, selected, N, levels_list, ff_coded, enforce_balance=False
        )

        # Evaluate design
        Xd = Xf[selected]
        det_val = np.linalg.det(Xd.T @ Xd)
        if det_val > best_det:
            best_det = det_val
            best_selected = sorted(selected)

    # Compute R-compatible D-efficiency: Dea = exp(1 - 1/Ge)
    # where Ge = p / max_i(xi' (Xd'Xd/n)^{-1} xi) over all candidates
    if best_selected is not None:
        Dea = _compute_dea(Xf, best_selected)
    else:
        Dea = 0.0

    # Check actual balance of the found design
    actually_balanced = (
        _is_balanced(best_selected, levels_list, ff_coded)
        if best_selected is not None
        else False
    )

    return best_selected, Dea, actually_balanced


def _compute_dea(Xf, selected):
    """Compute R-compatible D-efficiency (Dea = exp(1 - 1/Ge)).

    Ge (G-efficiency) = p / max_i(xi' (Xd'Xd/n)^{-1} xi)
    where the max is over all candidate points in the full factorial.
    """
    N, p = Xf.shape
    n = len(selected)
    Xd = Xf[selected]
    M = Xd.T @ Xd / n
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0

    max_d = max(Xf[i] @ M_inv @ Xf[i] for i in range(N))
    if max_d == 0:
        return 0.0
    Ge = p / max_d
    return math.exp(1 - 1 / Ge)


class doe:
    """
    Create (partial) factorial experimental design.

    Uses Federov exchange algorithm to find D-optimal designs. Iterates from
    the minimum number of trials upward until a design with perfect
    D-efficiency (1.0) is found. Reports actual balance status (whether each
    level appears equally often) for each design found.

    Parameters
    ----------
    factors : dict[str, list[str]]
        Factor names mapped to their levels.
        e.g., {"price": ["$10", "$13", "$16"], "food": ["popcorn", "gourmet"]}
    interactions : str or list[str]
        Interaction terms like "price:food"
    trials : int or None
        Number of trials. If None, search from minimum to full factorial.
    seed : int or None
        Random seed for reproducibility. None draws a fresh starting point,
        so repeated calls can return different, equally good designs.

    Attributes
    ----------
    df : dict
        Factor names and their levels
    full : pl.DataFrame
        Full factorial design with trial numbers
    part : pl.DataFrame
        Partial (or full) factorial design with trial numbers
    eff : pl.DataFrame
        Design efficiency table
    cor_mat : np.ndarray
        Correlation matrix of partial factorial design
    Dea : float
        D-efficiency of the selected design

    Examples
    --------
    >>> d = doe({"price": ["$10", "$13", "$16"], "food": ["popcorn", "gourmet", "no food"]})
    >>> d.Dea
    1.0
    """

    def __init__(
        self,
        factors: dict[str, list[str]] | None = None,
        interactions: str | list[str] = "",
        trials: int | None = None,
        seed: int | None = None,
    ):
        self.error = None
        if factors is None:
            factors = {}

        df_dict = {k: list(v) for k, v in factors.items()}
        df_names = list(factors.keys())

        if len(df_dict) < 2:
            self.error = "DOE requires at least two factors"
            return

        self.df = df_dict
        self.df_names = df_names
        self.seed = seed
        # Without a seed every search starts somewhere new (as R's set.seed is
        # skipped), so running again can show an alternative design.
        search_seed = (
            seed if seed is not None else int(np.random.default_rng().integers(0, 2**31))
        )

        # Parse interactions
        int_pairs = _parse_interactions(interactions, df_names)

        # Build full factorial
        full_pl = expand_grid(df_dict)
        levels_list = [len(df_dict[name]) for name in df_names]
        n_full = full_pl.height

        # Integer-coded full factorial derived from expand_grid output
        ff_coded = _integer_code(full_pl, df_dict, df_names)

        # Full model matrix with intercept
        Xf = _build_model_matrix(ff_coded, levels_list, int_pairs)

        # Minimum trials based on main-effects formula (matches R's AlgDesign)
        # optFederov will skip trials where the model is rank-deficient
        min_trials = sum(levels_list) - len(df_names) + 1
        max_trials = n_full

        if trials is not None:
            trials = max(min(trials, max_trials), min_trials)
            min_trials = max_trials = trials

        # Design efficiency table
        eff_trials = []
        eff_d = []
        eff_balanced = []

        best_design_rows = None
        best_Dea = 0

        # Scale repeats: fewer for large factorials (many trial counts to try)
        n_repeats = max(10, min(50, 500 // max(1, max_trials - min_trials + 1)))

        for n_trials in range(min_trials, max_trials + 1):
            if n_trials < Xf.shape[1] and n_trials < n_full:
                # Fewer trials than model coefficients cannot estimate the
                # model (R's optFederov refuses, so the row never shows).
                continue
            if n_trials == n_full:
                # Full factorial — use all rows
                design_rows = list(range(n_full))
                Dea = 1.0
                balanced = True
            else:
                design_rows, Dea, balanced = _find_optimal_design(
                    Xf, ff_coded, levels_list, n_trials, n_repeats=n_repeats, seed=search_seed
                )

                if design_rows is None:
                    continue

            if Dea > 0:
                eff_trials.append(n_trials)
                eff_d.append(round(Dea, 3))
                eff_balanced.append(balanced)

            if Dea > best_Dea:
                best_Dea = Dea
                best_design_rows = design_rows

            if Dea == 1.0:
                break

        self.eff = pl.DataFrame(
            {
                "Trials": eff_trials,
                "D-efficiency": eff_d,
                "Balanced": eff_balanced,
            }
        )

        # Cast factor columns to pl.Enum with level ordering from input dict
        enum_casts = [pl.col(name).cast(pl.Enum(df_dict[name])) for name in df_names]
        full_pl = full_pl.with_columns(enum_casts)

        # Build full factorial with trial numbers
        self.full = pl.DataFrame({"trial": list(range(1, n_full + 1))}).hstack(full_pl)

        # Build partial factorial
        if best_design_rows is not None:
            sorted_rows = sorted(best_design_rows)
            part_pl = full_pl[sorted_rows]
            trial_nums = [r + 1 for r in sorted_rows]
            self.part = pl.DataFrame({"trial": trial_nums}).hstack(part_pl)
            self.Dea = best_Dea

            # Correlation matrix using polychoric correlation
            numeric_cols = []
            for name in df_names:
                col = part_pl[name]
                numeric = col.cast(pl.Categorical).to_physical().cast(pl.Float64)
                numeric_cols.append(numeric.to_numpy())

            if len(numeric_cols) >= 2:
                data_matrix = np.column_stack(numeric_cols)
                corr = polychoric_matrix(data_matrix)
                self.cor_mat = corr
                self.detcm = np.linalg.det(corr)
            else:
                self.cor_mat = np.array([[1.0]])
                self.detcm = 1.0
        else:
            self.part = self.full
            self.Dea = 1.0
            self.cor_mat = np.eye(len(df_names))
            self.detcm = 1.0

        # Store for estimable
        self._Xf = Xf
        self._ff_coded = ff_coded
        self._levels_list = levels_list
        self._int_pairs = int_pairs

    def _effect_columns(self):
        """Columns of the saturated model R's ``estimable`` fits.

        ``trial ~ f1 * f2 * ... * fk`` with treatment (dummy) coding, in R's
        ``model.matrix`` order: the intercept, then terms by order (main
        effects, two-way, three-way, ...), terms of one order in the order R
        creates them (by the bit pattern of the factors they contain), and
        within an interaction the first factor's levels varying fastest.

        Returns ``(X, effects, terms)``: the model matrix of the full factorial
        and, for every column after the intercept, the effect name
        (``price|13:food|gourmet``) and the term it belongs to (``price:food``).
        """
        k = len(self.df_names)
        dummies = []
        for i, name in enumerate(self.df_names):
            levels = self.df[name]
            dummies.append(
                [
                    ((self._ff_coded[:, i] == j).astype(float), f"{name}|{levels[j]}")
                    for j in range(1, len(levels))
                ]
            )
        cols = [np.ones(self._ff_coded.shape[0])]
        effects, terms = [], []
        masks = sorted(range(1, 2**k), key=lambda m: (bin(m).count("1"), m))
        for mask in masks:
            members = [i for i in range(k) if mask >> i & 1]
            # itertools.product varies the LAST factor fastest; R varies the
            # first, so build the product over the reversed factor list.
            for combo in itertools.product(*[dummies[i] for i in reversed(members)]):
                combo = combo[::-1]
                col = np.prod([c for c, _ in combo], axis=0)
                cols.append(col)
                effects.append(":".join(label for _, label in combo))
                terms.append(":".join(self.df_names[i] for i in members))
        return np.column_stack(cols), effects, terms

    def _estimable_mask(self):
        """Which effects of the saturated model the partial design identifies.

        Fits the saturated model to the partial factorial the way R's ``lm()``
        does (LINPACK ``dqrdc2``, tolerance 1e-7): columns are taken left to
        right and one that is a linear combination of the columns kept so far
        is aliased (``NA`` in R) and dropped. Main effects come first, so a
        main effect is only dropped when it cannot be estimated at all.
        """
        X, effects, terms = self._effect_columns()
        part_rows = [t - 1 for t in self.part["trial"].to_list()]
        rank, pivot = r_qr_rank(X[part_rows])
        kept = set(int(i) for i in pivot[:rank])
        return [i + 1 in kept for i in range(len(effects))], effects, terms

    def estimable(self) -> list[str]:
        """Effects that can be estimated from the partial factorial design.

        Matches R's ``radiant.design::estimable``: the coefficients of the
        saturated model (all main effects and interactions of every order)
        that ``lm()`` does not drop as aliased when it is fit to the partial
        design.

        Returns
        -------
        list[str]
            Names of estimable effects (excluding the intercept), e.g.
            ``price|13`` or ``price|13:food|gourmet``
        """
        if self.error:
            return []
        mask, effects, _ = self._estimable_mask()
        return [e for e, ok in zip(effects, mask) if ok]

    def estimable_table(self, by: str = "term") -> pl.DataFrame:
        """Estimable effects of the partial factorial design as a table.

        Parameters
        ----------
        by : str
            ``"term"`` (default): one row per main effect or interaction with
            the number of coefficients (dummy columns) it has and how many of
            them the partial design can estimate. ``"effect"``: one row per
            coefficient with its term and whether it is estimable.

        Returns
        -------
        pl.DataFrame
        """
        if by not in ("term", "effect"):
            raise ValueError("by must be 'term' or 'effect'")
        if self.error:
            return pl.DataFrame()
        mask, effects, terms = self._estimable_mask()
        detail = pl.DataFrame(
            {"Effect": effects, "Term": terms, "Estimable": mask},
            schema={"Effect": pl.String, "Term": pl.String, "Estimable": pl.Boolean},
        )
        if by == "effect":
            return detail
        order = list(dict.fromkeys(terms))
        out = (
            detail.group_by("Term", maintain_order=True)
            .agg(
                pl.len().alias("Coefficients"),
                pl.col("Estimable").sum().alias("Estimable"),
            )
            .with_columns(
                pl.col("Term")
                .map_elements(_term_type, return_dtype=pl.String)
                .alias("Type"),
                pl.when(pl.col("Estimable") == pl.col("Coefficients"))
                .then(pl.lit("all"))
                .when(pl.col("Estimable") == 0)
                .then(pl.lit("none"))
                .otherwise(pl.lit("some"))
                .alias("Status"),
            )
            .with_columns(pl.col("Estimable").cast(pl.Int64), pl.col("Coefficients").cast(pl.Int64))
            .select("Term", "Type", "Coefficients", "Estimable", "Status")
        )
        assert out["Term"].to_list() == order
        return out

    def summary(
        self,
        eff: bool = True,
        part: bool = True,
        full: bool = True,
        est: bool = False,
        dec: int = 3,
    ) -> None:
        """Print summary of experimental design.

        Parameters
        ----------
        eff : bool
            Print efficiency output
        part : bool
            Print partial factorial design
        full : bool
            Print full factorial design
        est : bool
            Print estimable effects
        dec : int
            Number of decimal places
        """
        if self.error:
            print(self.error)
            return

        print("Experimental design")
        print(f"# trials for partial factorial: {self.part.height}")
        print(f"# trials for full factorial   : {self.full.height}")
        if self.seed is not None:
            print(f"Random seed                   : {self.seed}")

        print("\nAttributes and levels:")
        for name in self.df_names:
            print(f"{name}: {', '.join(self.df[name])}")

        with pl.Config(
            tbl_rows=-1,
            tbl_cols=-1,
            tbl_hide_dataframe_shape=True,
            tbl_hide_column_data_types=True,
            fmt_str_lengths=100,
            float_precision=dec,
        ):
            if eff:
                print("\nDesign efficiency:")
                print(self.eff)

                print("\nPartial factorial design correlations (polychoric):")
                cor_rounded = np.round(self.cor_mat, dec)
                cor_df = pl.DataFrame(
                    {name: cor_rounded[:, i] for i, name in enumerate(self.df_names)}
                ).with_columns(pl.Series("", self.df_names).alias(""))
                cor_df = cor_df.select([""] + self.df_names)
                print(cor_df)

            if part:
                print("\nPartial factorial design:")
                print(self.part)

            if est:
                print("\nEstimable effects from partial factorial design:")
                print(self.estimable_table())
                print("\nEstimable coefficients (saturated model, aliased columns dropped as in a regression):")
                print(self.estimable_table(by="effect"))

            if full:
                print("\nFull factorial design:")
                print(self.full)
