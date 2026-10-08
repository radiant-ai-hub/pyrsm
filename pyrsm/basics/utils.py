def _pct(x, dec):
    # 5.0 -> "5%", 2.5 -> "2.5%", as radiant.basics labels interval bounds
    return f"{round(x, dec):g}%"


def ci_label(alt="two-sided", conf=0.95, dec=3):
    if alt == "less":
        return ["0%", _pct(100 * conf, dec)]
    elif alt == "greater":
        return [_pct(100 * (1 - conf), dec), "100%"]
    elif alt == "two-sided":
        val = 100 * (1 - conf) / 2
        return [_pct(v, dec) for v in [val, 100 - val]]
    else:
        return "Error: alt_hyp must be one of 'less', 'greater', or 'two-sided'"
