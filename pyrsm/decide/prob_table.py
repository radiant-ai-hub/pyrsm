"""Probability tables for a test or signal that is not perfect.

The table way to apply Bayes' rule in a decision tree: start from what the
problem gives -- the prior probability of each state (a store succeeds 60%
of the time) and how often the test or signal comes back each way in each
state (a popup store does well at 80% of locations that succeed) -- and fill in

1. the **joint** table, P(signal and state) = P(signal | state) x P(state),
   whose column totals are the probability of each signal, P(signal);
2. the **posterior** table, P(state | signal) = P(signal and state) / P(signal),
   the probabilities to use on the branches after the signal is seen.

>>> import pyrsm as rsm
>>> pt = rsm.decide.prob_table(
...     {"S": 0.6, "F": 0.4},
...     {"S": {"Pos": 0.8, "Neg": 0.2}, "F": {"Pos": 0.1, "Neg": 0.9}},
... )
>>> round(pt.values["P(Pos)"], 2), round(pt.values["P(S|Pos)"], 4)
(0.52, 0.9231)

``pt.variables()`` writes the ``variables:`` lines for a decision tree, with
the derived probabilities as formulas, so a sensitivity analysis on a given
probability flows through to the posteriors.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

__all__ = ["prob_table"]

_TOL = 1e-6
_BAD = ("|", "(", ")", ":", "#", "\n")


def _check_name(kind: str, name: str) -> str:
    name = str(name).strip()
    if not name:
        raise ValueError(f"A {kind} name is empty.")
    if any(c in name for c in _BAD) or " and " in f" {name} ":
        raise ValueError(
            f"The {kind} name '{name}' cannot contain '|', '(', ')', ':', '#' or the word 'and' "
            "(they are part of the probability names, such as P(Pos|S))."
        )
    return name


def _number(value: Any, what: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} is not a number: {value!r}.") from None
    if not math.isfinite(x) or x < 0 or x > 1:
        raise ValueError(f"{what} must be a probability between 0 and 1, not {value!r}.")
    return x


def _fmt(x: float) -> str:
    """A given probability as typed in a tree: 0.8, 0.125 (no float noise)."""
    return f"{round(x, 10):g}"


class prob_table:
    """Prior, conditional, joint and posterior probability tables.

    Parameters
    ----------
    prior : dict
        The probability of each state, e.g. ``{"S": 0.6, "F": 0.4}``. They must
        add up to 1.
    likelihood : dict
        For each state, the probability of each signal in that state
        (P(signal | state)), e.g. ``{"S": {"Pos": 0.8, "Neg": 0.2}, "F":
        {"Pos": 0.1, "Neg": 0.9}}``. Each state's probabilities must add up to
        1 and every state lists the same signals.
    state, signal : str
        What the states and signals are, for the table titles (``"store"``,
        ``"popup"``).
    dec : int
        Decimals shown in the tables (the values are not rounded).

    Attributes
    ----------
    states, signals : list[str]
    values : dict[str, float]
        Every probability by its tree name: ``P(S)``, ``P(Pos|S)``,
        ``P(Pos and S)``, ``P(Pos)``, ``P(S|Pos)``.
    conditional, joint, posterior : polars.DataFrame
        The three tables, one row per state (the joint and posterior tables end
        with a ``Total`` row).
    """

    def __init__(
        self,
        prior: Mapping[str, Any],
        likelihood: Mapping[str, Mapping[str, Any]],
        *,
        state: str = "state",
        signal: str = "signal",
        dec: int = 4,
    ) -> None:
        if not prior:
            raise ValueError("Give the prior probability of at least two states.")
        self.state, self.signal, self.dec = str(state), str(signal), int(dec)
        self.states = [_check_name("state", s) for s in prior]
        if len(set(self.states)) != len(self.states):
            raise ValueError("Two states have the same name.")
        p_state = {s: _number(prior[k], f"P({s})") for s, k in zip(self.states, prior)}
        total = sum(p_state.values())
        if abs(total - 1) > _TOL:
            raise ValueError(
                f"The prior probabilities add up to {total:.4g}, not 1: "
                + ", ".join(f"P({s}) = {_fmt(p)}" for s, p in p_state.items())
                + "."
            )

        lik = {str(k).strip(): v for k, v in likelihood.items()}
        missing = [s for s in self.states if s not in lik]
        extra = [s for s in lik if s not in self.states]
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"no P(signal | state) row for {', '.join(missing)}")
            if extra:
                parts.append(f"rows for {', '.join(extra)}, which has no prior probability")
            raise ValueError("The conditional table has " + " and ".join(parts) + ".")
        first = lik[self.states[0]]
        self.signals = [_check_name("signal", g) for g in first]
        if len(self.signals) < 2:
            raise ValueError("Give at least two signals (for example Pos and Neg).")
        clash = set(self.signals) & set(self.states)
        if clash:
            raise ValueError(f"A state and a signal share a name: {', '.join(sorted(clash))}.")

        cond: dict[str, dict[str, float]] = {}
        for s in self.states:
            row = {str(k).strip(): v for k, v in lik[s].items()}
            if sorted(row) != sorted(self.signals):
                raise ValueError(
                    f"State {s} lists signals {', '.join(row)}; every state needs "
                    f"{', '.join(self.signals)}."
                )
            cond[s] = {g: _number(row[g], f"P({g}|{s})") for g in self.signals}
            rtot = sum(cond[s].values())
            if abs(rtot - 1) > _TOL:
                raise ValueError(
                    f"P(signal | {s}) adds up to {rtot:.4g}, not 1: "
                    + ", ".join(f"P({g}|{s}) = {_fmt(p)}" for g, p in cond[s].items())
                    + "."
                )

        joint = {s: {g: cond[s][g] * p_state[s] for g in self.signals} for s in self.states}
        p_signal = {g: sum(joint[s][g] for s in self.states) for g in self.signals}
        zero = [g for g, p in p_signal.items() if p <= 0]
        if zero:
            raise ValueError(
                f"Signal {', '.join(zero)} never happens (probability 0), so there is no "
                "posterior to calculate after it. Remove it."
            )
        post = {s: {g: joint[s][g] / p_signal[g] for g in self.signals} for s in self.states}

        self._prior, self._cond, self._joint = p_state, cond, joint
        self._p_signal, self._post = p_signal, post
        values: dict[str, float] = {}
        for s in self.states:
            values[f"P({s})"] = p_state[s]
        for s in self.states:
            for g in self.signals:
                values[f"P({g}|{s})"] = cond[s][g]
        for s in self.states:
            for g in self.signals:
                values[f"P({g} and {s})"] = joint[s][g]
        for g in self.signals:
            values[f"P({g})"] = p_signal[g]
        for g in self.signals:
            for s in self.states:
                values[f"P({s}|{g})"] = post[s][g]
        self.values = values

    # -- tables --------------------------------------------------------------

    @property
    def conditional(self):
        import polars as pl

        return pl.DataFrame(
            [{self.state: s, "prior": self._prior[s], **self._cond[s]} for s in self.states]
        )

    @property
    def joint(self):
        import polars as pl

        rows = [
            {self.state: s, **self._joint[s], "Total": self._prior[s]} for s in self.states
        ]
        rows.append({self.state: "Total", **self._p_signal, "Total": 1.0})
        return pl.DataFrame(rows)

    @property
    def posterior(self):
        import polars as pl

        rows = [{self.state: s, **self._post[s]} for s in self.states]
        rows.append({self.state: "Total", **{g: 1.0 for g in self.signals}})
        return pl.DataFrame(rows)

    # -- tree variables ------------------------------------------------------

    def variables(self, header: bool = True) -> str:
        """The ``variables:`` lines for a decision tree: the given probabilities
        as numbers, every other one as a formula of those (the last state's
        prior and each state's last signal as ``1 - ...``), so changing a given
        probability in a sensitivity analysis updates the posteriors.

        ``header=False`` leaves out the ``variables:`` line, to add the lines
        to a tree that already has one.
        """
        ind = "    "
        lines = ["variables:"] if header else []
        lines.append(f"{ind}## Given probabilities")
        *first, last = self.states
        for s in first:
            lines.append(f"{ind}P({s}): {_fmt(self._prior[s])}")
        lines.append(f"{ind}P({last}): 1 - " + " - ".join(f"P({s})" for s in first))
        for s in self.states:
            *gs, gl = self.signals
            for g in gs:
                lines.append(f"{ind}P({g}|{s}): {_fmt(self._cond[s][g])}")
            lines.append(f"{ind}P({gl}|{s}): 1 - " + " - ".join(f"P({g}|{s})" for g in gs))
        lines.append(f"{ind}## Joint probabilities: P(signal and state) = P(signal|state) * P(state)")
        for s in self.states:
            for g in self.signals:
                lines.append(f"{ind}P({g} and {s}): P({g}|{s}) * P({s})")
        lines.append(f"{ind}## Signal probabilities: the column totals of the joint table")
        for g in self.signals:
            lines.append(f"{ind}P({g}): " + " + ".join(f"P({g} and {s})" for s in self.states))
        lines.append(f"{ind}## Posterior probabilities: P(state|signal) = P(signal and state) / P(signal)")
        for g in self.signals:
            for s in self.states:
                lines.append(f"{ind}P({s}|{g}): P({g} and {s}) / P({g})")
        return "\n".join(lines) + "\n"

    # -- output --------------------------------------------------------------

    def report(self) -> dict[str, Any]:
        """The report Radiant shows (pyrsm.report layout)."""
        from pyrsm.report import build_report, column, table

        def tbl(id: str, frame, title: str, subtitle: str = "") -> dict[str, Any]:
            cols = [column(self.state, self.state, "term")] + [column(c) for c in frame.columns[1:]]
            return table(
                id, frame.iter_rows(named=True), title=title, subtitle=subtitle,
                columns=cols, terms=[self.state], dec=self.dec,
            )

        cond = self.conditional.rename({"prior": f"P({self.state})"})
        cond_cols = {g: f"P({g}|{self.state})" for g in self.signals}
        cond = cond.rename(cond_cols)
        header = {
            "title": "Probability tables",
            "meta": [
                f"States ({self.state}): {', '.join(self.states)}",
                f"Signals ({self.signal}): {', '.join(self.signals)}",
            ],
        }
        return build_report(
            "prob_table",
            header=header,
            tables=[
                tbl(
                    "conditional", cond, "Prior and conditional probabilities",
                    f"Given: P({self.state}) and P({self.signal} | {self.state}); each row adds up to 1",
                ),
                tbl(
                    "joint", self.joint, "Joint probabilities",
                    f"P({self.signal} and {self.state}) = P({self.signal} | {self.state}) x P({self.state}); "
                    f"the Total row is P({self.signal})",
                ),
                tbl(
                    "posterior", self.posterior, "Posterior probabilities",
                    f"P({self.state} | {self.signal}) = P({self.signal} and {self.state}) / P({self.signal}); "
                    "each column adds up to 1",
                ),
            ],
            downloads=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Everything Radiant stores: the inputs, every value, the report and
        the tree variables."""
        return {
            "state": self.state,
            "signal": self.signal,
            "states": list(self.states),
            "signals": list(self.signals),
            "prior": dict(self._prior),
            "likelihood": {s: dict(r) for s, r in self._cond.items()},
            "values": dict(self.values),
            "report": self.report(),
            "variables": self.variables(),
        }

    def summary(self, plain: bool = True, dec: int | None = None) -> None:
        """Print the three tables (``plain=False`` shows Radiant's tables in a
        notebook)."""
        if dec is not None:
            self.dec = int(dec)
        if not plain:
            from pyrsm.report import display

            display(self.report(), self._text())
            return
        print(self._text())

    def _text(self) -> str:
        import polars as pl

        def show(frame) -> str:
            num = [c for c in frame.columns if c != self.state]
            frame = frame.with_columns(pl.col(num).round(self.dec))
            with pl.Config(
                tbl_formatting="NOTHING", tbl_hide_dataframe_shape=True,
                tbl_hide_column_data_types=True, tbl_hide_dtype_separator=True,
                float_precision=self.dec,
            ):
                return str(frame)

        cond = self.conditional.rename({"prior": f"P({self.state})"}).rename(
            {g: f"P({g}|{self.state})" for g in self.signals}
        )
        return "\n".join(
            [
                "Probability tables",
                f"States ({self.state}) : {', '.join(self.states)}",
                f"Signals ({self.signal}): {', '.join(self.signals)}",
                "",
                "Prior and conditional probabilities (given):",
                show(cond),
                "",
                f"Joint probabilities P({self.signal} and {self.state}) (Total row = P({self.signal})):",
                show(self.joint),
                "",
                f"Posterior probabilities P({self.state} | {self.signal}):",
                show(self.posterior),
            ]
        )

    def __repr__(self) -> str:
        return self._text()
