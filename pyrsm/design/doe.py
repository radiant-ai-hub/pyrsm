import math

import numpy as np
import polars as pl

from pyrsm.utils import expand_grid, polychoric_matrix


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


def _federov_exchange(Xf, selected, N, levels_list, ff_coded, enforce_balance, max_iter=1000):
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
        rep_seed = (seed if seed is not None else 172110) + rep
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
        _is_balanced(best_selected, levels_list, ff_coded) if best_selected is not None else False
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
        Random seed for reproducibility

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
            if n_trials == n_full:
                # Full factorial — use all rows
                design_rows = list(range(n_full))
                Dea = 1.0
                balanced = True
            else:
                design_rows, Dea, balanced = _find_optimal_design(
                    Xf, ff_coded, levels_list, n_trials, n_repeats=n_repeats, seed=seed
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
        enum_casts = [
            pl.col(name).cast(pl.Enum(df_dict[name])) for name in df_names
        ]
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

    def estimable(self) -> list[str]:
        """Determine which effects are estimable from the partial factorial design.

        Returns
        -------
        list[str]
            Names of estimable effects (excluding intercept)
        """
        if self.error:
            return []

        # Build full interaction model matrix (all possible interactions)
        all_pairs = []
        for i in range(len(self.df_names)):
            for j in range(i + 1, len(self.df_names)):
                all_pairs.append((i, j))

        Xf_full = _build_model_matrix(self._ff_coded, self._levels_list, all_pairs)

        # Build full coefficient names
        full_coef_names = ["intercept"]
        for i, name in enumerate(self.df_names):
            levels = self.df[name]
            for j in range(1, len(levels)):
                full_coef_names.append(f"{name}|{levels[j]}")

        for f1, f2 in all_pairs:
            name1 = self.df_names[f1]
            name2 = self.df_names[f2]
            levels1 = self.df[name1]
            levels2 = self.df[name2]
            for j1 in range(1, len(levels1)):
                for j2 in range(1, len(levels2)):
                    full_coef_names.append(f"{name1}|{levels1[j1]}:{name2}|{levels2[j2]}")

        # Get partial factorial rows
        part_rows = [t - 1 for t in self.part["trial"].to_list()]

        # Fit on partial design using pivoted QR decomposition
        # Columns that are NOT aliased (i.e., in the first 'rank' pivot positions)
        # correspond to estimable effects. This matches R's lm() behavior.
        Xp = Xf_full[part_rows]

        try:
            from scipy.linalg import qr

            _, R, P = qr(Xp, pivoting=True)
            tol = 1e-10
            rank = np.sum(np.abs(np.diag(R)) > tol)
            estimable_indices = set(int(P[i]) for i in range(rank))

            estimable_names = []
            for idx in range(1, len(full_coef_names)):
                if idx in estimable_indices:
                    estimable_names.append(full_coef_names[idx])
        except Exception:
            estimable_names = []

        return estimable_names

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
                estimable_effects = self.estimable()
                print("\nEstimable effects from partial factorial design:\n")
                for e in estimable_effects:
                    print(f"  {e}")

            if full:
                print("\nFull factorial design:")
                print(self.full)
