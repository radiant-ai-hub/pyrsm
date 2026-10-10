import numpy as np
from scipy import stats

from .utils import check, iround, plot_discrete


def prob_pois(lamb, lb=None, ub=None, plb=None, pub=None):
    if lamb <= 0:
        raise ValueError("Lambda must be positive")
    if lb is not None and lb < 0:
        lb = 0
    if ub is not None and ub < 0:
        ub = 0
    if lb is None:
        p_lb = None
        p_elb = None
    else:
        p_elb = stats.poisson.pmf(lb, lamb)
        p_lelb = stats.poisson.cdf(lb, lamb)
        if lb > 0:
            p_lb = p_lelb - p_elb
        else:
            p_lb = 0
    if ub is None:
        p_ub = None
        p_eub = None
    else:
        p_eub = stats.poisson.pmf(ub, lamb)
        p_leub = stats.poisson.cdf(ub, lamb)
        if ub > 0:
            p_ub = p_leub - p_eub
        else:
            p_ub = 0
    p_int = None
    if lb is not None and ub is not None:
        # P(lb <= X <= ub); works for float bounds such as 1.0 from a form
        p_int = max(stats.poisson.cdf(ub, lamb) - p_lb, 0)
    if pub is not None:
        if pub < 0:
            pub = 0
        elif pub > 1:
            pub = 1
    if plb is not None:
        if plb < 0:
            plb = 0
        elif plb > 1:
            plb = 1
    # Probability bounds map to the smallest value whose cumulative
    # probability reaches the bound (R's qpois), then get the same
    # discrete table as a value bound.
    def at_value(prob):
        if prob is None:
            return None, None, None, None
        v = stats.poisson.ppf(prob, lamb)
        if not np.isfinite(v):
            raise ValueError("Probability bounds must be smaller than 1")
        v = int(v)
        p_e = stats.poisson.pmf(v, lamb)
        p_le = stats.poisson.cdf(v, lamb)
        return v, p_e, p_le - p_e, p_le

    v_lb, vp_elb, vp_lb, vp_lelb = at_value(plb)
    v_ub, vp_eub, vp_ub, vp_leub = at_value(pub)
    vp_int = vp_leub - vp_lb if plb is not None and pub is not None else None
    check(lb, ub, plb, pub)
    return {
        "p_lb": p_lb,
        "p_ub": p_ub,
        "p_int": p_int,
        "v_lb": v_lb,
        "v_ub": v_ub,
        "lamb": lamb,
        "lb": lb,
        "ub": ub,
        "plb": plb,
        "pub": pub,
        "p_elb": p_elb,
        "p_eub": p_eub,
        "vp_elb": vp_elb,
        "vp_lb": vp_lb,
        "vp_lelb": vp_lelb,
        "vp_eub": vp_eub,
        "vp_ub": vp_ub,
        "vp_leub": vp_leub,
        "vp_int": vp_int,
    }


def summary_prob_pois(dct, type="values", dec=3, ret=False):
    def add(d, k, v):
        if v is not None:
            d[k] = v

    dct = {k: iround(v, dec) for k, v in dct.items()}
    summary_dict = {}
    add(summary_dict, "Distribution", "Poisson")
    lamb = dct["lamb"]
    add(summary_dict, "Lambda", lamb)
    add(summary_dict, "Mean", lamb)
    add(summary_dict, "Variance", lamb)
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
        p_lb, p_ub, p_int = dct["p_lb"], dct["p_ub"], dct["p_int"]
        p_elb, p_eub = dct["p_elb"], dct["p_eub"]
        add(summary_dict, "Lower bound", "" if lb is None else lb)
        add(summary_dict, "Upper bound", "" if ub is None else ub)
        if ub is not None or lb is not None:
            if lb is not None:
                add(summary_dict, f"P(X  = {lb})", p_elb)
                if lb > 0:
                    add(summary_dict, f"P(X  < {lb})", p_lb)
                    add(summary_dict, f"P(X <= {lb})", round(p_lb + p_elb, dec))
                add(summary_dict, f"P(X  > {lb})", round(1 - (p_lb + p_elb), dec))
                add(summary_dict, f"P(X >= {lb})", round(1 - p_lb, dec))
            if ub is not None:
                add(summary_dict, f"P(X  = {ub})", p_eub)
                if ub > 0:
                    add(summary_dict, f"P(X  < {ub})", p_ub)
                    add(summary_dict, f"P(X <= {ub})", round(p_ub + p_eub, dec))
                add(summary_dict, f"P(X  > {ub})", round(1 - (p_ub + p_eub), dec))
                add(summary_dict, f"P(X >= {ub})", round(1 - p_ub, dec))
            if lb is not None and ub is not None:
                add(summary_dict, f"P({lb} <= X <= {ub})", p_int)
                add(summary_dict, f"1 - P({lb} <= X <= {ub})", round(1 - p_int, dec))
    else:
        plb, pub = dct["plb"], dct["pub"]
        v_lb, v_ub = dct["v_lb"], dct["v_ub"]
        add(summary_dict, "Lower bound", "" if plb is None else f"{plb} ({v_lb})")
        add(summary_dict, "Upper bound", "" if pub is None else f"{pub} ({v_ub})")
        for prob, v, p_e, p_l in (
            (plb, v_lb, dct["vp_elb"], dct["vp_lb"]),
            (pub, v_ub, dct["vp_eub"], dct["vp_ub"]),
        ):
            if prob is None:
                continue
            add(summary_dict, f"P(X  = {v})", p_e)
            if v > 0:
                add(summary_dict, f"P(X  < {v})", p_l)
                add(summary_dict, f"P(X <= {v})", round(p_l + p_e, dec))
            add(summary_dict, f"P(X  > {v})", round(1 - (p_l + p_e), dec))
            add(summary_dict, f"P(X >= {v})", round(1 - p_l, dec))
        if plb is not None and pub is not None:
            vp_int = dct["vp_int"]
            add(summary_dict, f"P({v_lb} <= X <= {v_ub})", vp_int)
            add(summary_dict, f"1 - P({v_lb} <= X <= {v_ub})", round(1 - vp_int, dec))
    if ret:
        return summary_dict
    else:
        from .utils import pretty_print_summary

        pretty_print_summary(summary_dict)


def plot_prob_pois(dct, type="values"):
    if type == "values":
        lb, ub = dct["lb"], dct["ub"]
    else:
        lb, ub = dct["v_lb"], dct["v_ub"]
    lmb = dct["lamb"]
    x_range = list(range(int(stats.poisson.ppf(1 - 0.00001, lmb)) + 1))
    y_range = [stats.poisson.pmf(k, lmb) for k in x_range]
    x_array = np.array(x_range)
    y_array = np.array(y_range)
    mask = y_array > 0.00001
    x_filtered = x_array[mask]
    y_filtered = y_array[mask]
    return plot_discrete(x_filtered, y_filtered, lb, ub, title="Poisson Distribution")
