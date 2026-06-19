"""Lightweight checks against a model transcript for the three pyrsm-skill failure modes.

Usage:
    python check_transcript.py <transcript.txt> [--skill pyrsm-regress]
    cat transcript.txt | python check_transcript.py - [--skill pyrsm-regress]

What it checks:
    1. NO_BAD_SUMMARY_FLAGS — the transcript should not be calling
       `reg.summary(rmse=True, ssq=True)` or similar UNLESS the user prompt
       explicitly asked for sum-of-squares / RMSE / VIF / CIs. The default is
       plain `summary()`.

    2. FULL_OUTPUT_SHOWN — when `.summary()` is called, the response should
       include the verbatim printed output (header + coefficient table +
       significance codes + model-fit block). We detect this by looking for
       the polars box-drawing characters `┌`, `│`, `└` AND keywords like
       "p.value", "Adjusted R-squared", "Nr obs", or "Chi-squared". Failing
       these is a strong sign the table was omitted.

    3. NOT_LAME_BULLETS — the interpretation should be detailed (with units,
       holding-constant qualifiers, statistical-test verdicts), not a generic
       4-bullet "main takeaways" list. We detect lame bullets by counting
       short bullet items and looking for the bullet-only failure pattern.

The script exits 0 if all checks pass, 1 if any check fails. Designed to be
backend-agnostic — works against transcripts from Pi (Inflection), Claude,
GPT, or any other model.

Examples
--------

    # Basic check
    python check_transcript.py /path/to/transcript.txt

    # Check with a specific skill in mind
    python check_transcript.py /path/to/transcript.txt --skill pyrsm-regress

    # Pipe a transcript in from another tool
    pi-cli run-skill pyrsm-regress --prompt "..." | \
        python check_transcript.py - --skill pyrsm-regress

    # JSON output for CI integration
    python check_transcript.py /path/to/transcript.txt --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# -------- patterns ---------------------------------------------------------

# Calls like reg.summary(rmse=True, ssq=True, vif=True, ci=True) when not asked.
BAD_SUMMARY_FLAG_RE = re.compile(
    r"\.summary\s*\(\s*[^)]*"
    r"(?:rmse\s*=\s*True|ssq\s*=\s*True|vif\s*=\s*True|ci\s*=\s*True)"
    r"[^)]*\)"
)

# Box-drawing chars used by polars / pyrsm in printed tables.
TABLE_CHARS_RE = re.compile(r"[┌┐└┘├┤┬┴┼│─━╪╞╡]")

# Specific summary "evidence" markers — strings that appear in the verbatim
# printed output of pyrsm regress / logistic / etc.
SUMMARY_EVIDENCE_PATTERNS = [
    r"\bp\.value\b",
    r"R-squared",
    r"Adjusted R-squared",
    r"F-statistic",
    r"Chi-squared",
    r"AUC",
    r"Nr obs",
    r"Signif\. codes",
    r"Intercept",
    r"OOB",
    r"OR\b",  # logistic odds ratios
    r"std\.error",
    r"t\.value",
    r"z\.value",
    r"coefficient",
    r"variance",
    r"RMSE",
    # PIP / generic polars DataFrame headers (so PIP & predict outputs count too)
    r"\bimportance\b",
    r"\bvariable\b",
    r"\bprediction\b",
    r"shape:\s*\(\d+,\s*\d+\)",
]
SUMMARY_EVIDENCE_RE = re.compile("|".join(SUMMARY_EVIDENCE_PATTERNS))

# Common abbreviation phrases that indicate the model is hiding output.
ABBREVIATION_PHRASES = [
    r"\bomitted for brevity\b",
    r"\b\[.*?table.*?(?:omitted|truncated)\]",
    r"\bhigh-level summary\b",
    r"\bmain take[- ]?aways?\b",
    r"\bI'?ll skip the full output\b",
    r"\bI'?ll show the highlights\b",
    r"\bsummariz(?:ed|ing) (?:the |key )?results\b",
    r"\binstead of (?:the |a )?full (?:coefficient )?(?:table|output)",
]
ABBREVIATION_RE = re.compile("|".join(ABBREVIATION_PHRASES), re.IGNORECASE)

# Strong "lame bullet" signature: a short list of 3-6 bullets, each starting
# with a predictor name and ending in a short phrase like "is significant" or
# "is positively associated". This matches the user's reported failure.
LAME_BULLET_BULLET = re.compile(
    r"^[\s]*[-*]\s+(?:`?[A-Za-z_][A-Za-z_0-9]*`?\s+)?"
    r".{0,80}?(?:is (?:positively|negatively|statistically )?\s*"
    r"(?:significant|associated|not significant|related)|"
    r"has (?:a )?(?:positive|negative|significant) effect|"
    r"are (?:not )?statistically significant|"
    r"is the most important predictor)\s*\.?\s*$",
    re.MULTILINE,
)

# Detailed-interpretation evidence: language we EXPECT in a proper walk-through.
DETAILED_INTERPRETATION_PATTERNS = [
    r"\bholding (?:all )?(?:other )?variables (?:in the model )?constant\b",
    r"\bcompared to\b.+\breference\b",
    r"\bH[0₀].*\bH[a]?[1₁ₐ]?\b",  # H0 and Ha mentioned somewhere
    r"\bone[- ]unit\b",
    r"\bone[- ](?:dollar|year|carat|percentage[- ]point)\b",
    r"\bsignificant at (?:the )?5% level\b",
    r"\bp[- ]?value\b.+\b(?:<|less than|smaller than)\b\s*0\.05",
    r"\breject the null\b",
    r"\bfail to reject\b",
    r"\bin (?:dollars|units of)\b",
]

# ---- Importance-disclaimer check -----------------------------------------
# Language that strongly implies the model is claiming relative importance.
IMPORTANCE_CLAIM_PATTERNS = [
    r"\bmost (?:important|influential|strongly associated|predictive)\b",
    r"\b(?:biggest|largest|strongest|smallest|weakest) (?:effect|influence|impact|contributor|predictor)\b",
    r"\b(?:strongest|weakest) (?:predictor|driver)\b",
    r"\bmatters? (?:the )?most\b",
    r"\b(?:bottom[- ]?line|key takeaway|main finding)s?\b.{0,200}?(?:most|strongest|biggest)",
    r"\bwas most strongly associated with\b",
    r"\bdrives? the (?:largest|biggest) share\b",
    r"\branked? by (?:importance|influence|impact)\b",
    r"\b(?:more|less) (?:important|influential) than\b",
]
IMPORTANCE_CLAIM_RE = re.compile("|".join(IMPORTANCE_CLAIM_PATTERNS), re.IGNORECASE)

# Language that satisfies the disclaimer requirement: must mention permutation
# importance (PIP) by name OR show `plot("pip")`, AND mention the scale-
# comparability caveat (PDP / coefficients are not directly comparable).
DISCLAIMER_PIP_PATTERNS = [
    r"\bpermutation importance\b",
    r"\.plot\(\s*['\"]pip['\"]",
    r"\bPIP\b(?!_)",  # the abbreviation PIP, but not PIP_sklearn etc.
]
DISCLAIMER_PIP_RE = re.compile("|".join(DISCLAIMER_PIP_PATTERNS), re.IGNORECASE)

DISCLAIMER_CAVEAT_PATTERNS = [
    r"\bnot directly comparable\b",
    r"\bdifferent scales?\b",
    r"\b(?:without|unless) standardization\b",
    r"\bstandardize(?:d)? (?:the )?(?:predictors|coefficients|features)\b",
    r"\bnot (?:valid|suitable|appropriate) for (?:ranking|comparing) (?:importance|predictors|relative)\b",
    r"\b(?:PDP|partial dependence) (?:show|shows|reflects?) (?:the )?(?:shape|effect shape)\b",
    r"\bshape of (?:the |each )?(?:predictor'?s )?effect\b.+\bnot\b.+\b(?:importance|magnitude|contribution)\b",
    r"\bare on different scales\b",
    r"\bbiased toward (?:continuous|high-cardinality)\b",
]
DISCLAIMER_CAVEAT_RE = re.compile("|".join(DISCLAIMER_CAVEAT_PATTERNS), re.IGNORECASE)
DETAILED_INTERPRETATION_RE = re.compile("|".join(DETAILED_INTERPRETATION_PATTERNS), re.IGNORECASE)


# -------- check engine -----------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    evidence: str
    severity: str = "fail"  # "fail" | "warn"


def check_no_bad_summary_flags(transcript: str, user_prompt: str | None) -> CheckResult:
    """Pass if no .summary(rmse=True/ssq=True/vif=True/ci=True) appears unless the
    user prompt explicitly asked for these.
    """
    matches = list(BAD_SUMMARY_FLAG_RE.finditer(transcript))
    if not matches:
        return CheckResult(
            name="NO_BAD_SUMMARY_FLAGS",
            passed=True,
            evidence="No summary(rmse=True/ssq=True/vif=True/ci=True) found in transcript.",
        )
    if user_prompt:
        asked_for = re.search(
            r"\b(?:rmse|ssq|sum.of.squares|vif|multicollin|"
            r"confidence interval(?:s)? for (?:each |the )?coefficient)\b",
            user_prompt,
            re.IGNORECASE,
        )
        if asked_for:
            return CheckResult(
                name="NO_BAD_SUMMARY_FLAGS",
                passed=True,
                evidence=(
                    f"Found {len(matches)} summary call(s) with extra flags, "
                    f"but the user prompt asked for it: '{asked_for.group(0)}'."
                ),
            )
    examples = [m.group(0) for m in matches[:3]]
    return CheckResult(
        name="NO_BAD_SUMMARY_FLAGS",
        passed=False,
        evidence=(
            f"Found {len(matches)} summary call(s) with extra flags not requested "
            f"by the user. Example(s): {examples}. Default should be plain "
            "`.summary()` — only add flags when the user asks."
        ),
    )


_PLOT_SAVED_PATTERNS = [
    r"\.save\(\s*['\"][^'\"]+\.(?:png|pdf|svg)['\"]",
    r"\bsaved\s+(?:to\s+)?\S+\.(?:png|pdf|svg)\b",
    r"\b\w+\.(?:png|pdf|svg)\b",
]
_PLOT_SAVED_RE = re.compile("|".join(_PLOT_SAVED_PATTERNS), re.IGNORECASE)


def check_full_output_shown(transcript: str) -> CheckResult:
    """Pass if the transcript shows real pyrsm output and has not been
    abbreviated.

    Three acceptable signals:
      1. A polars-style table with box-drawing chars AND summary-evidence
         markers (regression / classification summaries, PIP DataFrames,
         predictions). The common case.
      2. A reference to a saved plot file (`.save("name.png")`, `saved foo.png`).
         Used for PDP / dashboard / correlation plots where the "output" is
         a plot file, not a printed table.

    Always fail on abbreviation phrases like "omitted for brevity",
    "high-level summary", "main takeaways".
    """
    table_chars = len(TABLE_CHARS_RE.findall(transcript))
    evidence_markers = set(m.group(0).lower() for m in SUMMARY_EVIDENCE_RE.finditer(transcript))
    abbreviations = [m.group(0) for m in ABBREVIATION_RE.finditer(transcript)]
    saved_plot_refs = [m.group(0) for m in _PLOT_SAVED_RE.finditer(transcript)]

    if abbreviations:
        return CheckResult(
            name="FULL_OUTPUT_SHOWN",
            passed=False,
            evidence=(
                f"Transcript contains abbreviation phrases that indicate hidden "
                f"output: {abbreviations[:3]}. The skill requires the FULL output "
                "to be shown verbatim, not summarized."
            ),
        )

    has_table = table_chars >= 4
    has_evidence = len(evidence_markers) >= 2
    has_saved_plot = bool(saved_plot_refs)

    if has_table and has_evidence:
        return CheckResult(
            name="FULL_OUTPUT_SHOWN",
            passed=True,
            evidence=(
                f"Found {table_chars} box-drawing chars and "
                f"{len(evidence_markers)} summary-evidence markers "
                f"({sorted(evidence_markers)[:5]}). Likely full output included."
            ),
        )

    if has_saved_plot:
        return CheckResult(
            name="FULL_OUTPUT_SHOWN",
            passed=True,
            evidence=(
                f"Transcript references {len(saved_plot_refs)} saved plot "
                f"file(s) (examples: {saved_plot_refs[:3]}). The 'output' for "
                "this turn is a plot file rather than a printed table; the "
                "shape interpretation is checked separately by the plot-"
                "specific check (e.g., PDP_RESULTS_SHOWN_IF_ASKED)."
            ),
        )

    return CheckResult(
        name="FULL_OUTPUT_SHOWN",
        passed=False,
        evidence=(
            f"Transcript has only {table_chars} box-drawing chars and "
            f"{len(evidence_markers)} summary-evidence markers, and references "
            "no saved plot files — looks like the verbatim output was "
            "abbreviated or omitted. Required either: 4+ table chars + 2+ "
            "markers, or evidence of a saved plot (.save('foo.png'))."
        ),
    )


def check_not_lame_bullets(transcript: str) -> CheckResult:
    """Pass if the interpretation is detailed, not a lame 4-bullet list.

    Fail conditions:
      - 3+ lame bullet matches AND fewer than 2 detailed-interpretation markers.
      - The "main takeaways" / "high-level summary" phrase appears.
    """
    lame_bullets = list(LAME_BULLET_BULLET.finditer(transcript))
    detailed_markers = list(DETAILED_INTERPRETATION_RE.finditer(transcript))

    has_takeaway_phrase = bool(re.search(r"\bmain take[- ]?aways?\s*:", transcript, re.IGNORECASE))

    if has_takeaway_phrase and len(lame_bullets) >= 3 and len(detailed_markers) < 3:
        bullet_examples = [m.group(0).strip() for m in lame_bullets[:3]]
        return CheckResult(
            name="NOT_LAME_BULLETS",
            passed=False,
            evidence=(
                f"Transcript has a 'Main takeaways:'-style section followed by "
                f"{len(lame_bullets)} terse bullets like {bullet_examples}, with "
                f"only {len(detailed_markers)} detailed-interpretation markers "
                "(units, holding-constant, H0/Ha, etc.). This is the exact "
                "'lame bullet' failure mode the skill is designed to prevent."
            ),
        )

    if len(lame_bullets) >= 4 and len(detailed_markers) < 2:
        bullet_examples = [m.group(0).strip() for m in lame_bullets[:3]]
        return CheckResult(
            name="NOT_LAME_BULLETS",
            passed=False,
            evidence=(
                f"Transcript has {len(lame_bullets)} short 'X is significant'-"
                f"style bullets and only {len(detailed_markers)} detailed-"
                f"interpretation markers. Examples: {bullet_examples}. "
                "Detailed walkthrough required, not a bullet list."
            ),
        )

    return CheckResult(
        name="NOT_LAME_BULLETS",
        passed=True,
        evidence=(
            f"Found {len(detailed_markers)} detailed-interpretation markers "
            f"(units, holding-constant, H0/Ha, etc.) vs {len(lame_bullets)} "
            f"terse bullets — looks like a proper walkthrough."
        ),
    )


def check_importance_disclaimer(transcript: str) -> CheckResult:
    """Pass if either (a) the transcript makes no relative-importance claim,
    or (b) it makes such a claim AND closes with a proper disclaimer pointing
    to permutation importance (PIP) along with the scale-comparability caveat.

    Fails when the transcript says something like "X was most strongly
    associated with Y" or "the bottom line is that A matters most" but
    doesn't reference permutation importance OR the caveat that PDPs /
    coefficients are not directly comparable across predictors on different
    scales.
    """
    importance_claims = list(IMPORTANCE_CLAIM_RE.finditer(transcript))
    if not importance_claims:
        return CheckResult(
            name="IMPORTANCE_DISCLAIMER_PRESENT",
            passed=True,
            evidence=(
                "No relative-importance claim detected in the transcript. "
                "(The disclaimer is only required when such a claim is made.)"
            ),
        )

    has_pip_pointer = bool(DISCLAIMER_PIP_RE.search(transcript))
    has_caveat = bool(DISCLAIMER_CAVEAT_RE.search(transcript))
    claim_examples = [m.group(0) for m in importance_claims[:3]]

    if has_pip_pointer and has_caveat:
        return CheckResult(
            name="IMPORTANCE_DISCLAIMER_PRESENT",
            passed=True,
            evidence=(
                f"Found {len(importance_claims)} importance claim(s) "
                f"(examples: {claim_examples}) AND a proper disclaimer "
                f"(PIP / permutation importance referenced AND scale-"
                f"comparability caveat present)."
            ),
        )

    missing = []
    if not has_pip_pointer:
        missing.append("no reference to permutation importance / PIP / .plot('pip')")
    if not has_caveat:
        missing.append(
            "no caveat about coefficients / PDPs not being directly "
            "comparable across predictors on different scales"
        )
    return CheckResult(
        name="IMPORTANCE_DISCLAIMER_PRESENT",
        passed=False,
        evidence=(
            f"Transcript makes {len(importance_claims)} relative-importance "
            f"claim(s) (examples: {claim_examples}) but is missing the "
            f"required closing disclaimer: {' AND '.join(missing)}. "
            "The skill requires this disclaimer at the end of any "
            "interpretation that implies one predictor matters more than "
            "another. Pi-style 'PDPs suggest survival was most strongly "
            "associated with sex...' without the PIP redirect is exactly "
            "the failure mode this check is designed to catch."
        ),
    )


_PIP_REQUEST_RE = re.compile(
    r"\b(?:pip|permutation\s+importance|feature\s+importance|"
    r"variable\s+importance|which\s+(?:predictor|variable|feature)s?\s+"
    r"(?:matters?|is\s+(?:the\s+)?most\s+(?:important|influential)))\b",
    re.IGNORECASE,
)

# Evidence that the actual PIP DataFrame appears in the response.
_PIP_OUTPUT_PATTERNS = [
    r"\bvariable\b\s*[│|]\s*importance\b",  # column headers
    r"\.plot\(\s*['\"]pip['\"][^)]*ret\s*=\s*True",  # the canonical call
    r"shape:\s*\(\d+,\s*2\)",  # polars shape header right above (variable, importance)
    r"\bimportance\b\s*\n[\s│┼├┤─]+\nf64",  # polars table header with f64 dtype line
]
_PIP_OUTPUT_RE = re.compile("|".join(_PIP_OUTPUT_PATTERNS), re.IGNORECASE)


def check_pip_results_shown_if_asked(transcript: str, user_prompt: str | None) -> CheckResult:
    """Pass if the user did not ask for PIP, OR they did and the response
    contains the actual PIP DataFrame (variable + importance scores).

    The Pi failure mode this catches: user asks "generate and interpret pip",
    Pi announces it'll generate PIP, then quietly reverts to interpreting
    regression coefficients without ever showing the PIP scores.
    """
    if not user_prompt:
        return CheckResult(
            name="PIP_RESULTS_SHOWN_IF_ASKED",
            passed=True,
            evidence=(
                "No user prompt supplied; cannot tell whether PIP was "
                "requested. Pass --prompt to enable this check."
            ),
            severity="warn",
        )

    asked_for_pip = bool(_PIP_REQUEST_RE.search(user_prompt))
    if not asked_for_pip:
        return CheckResult(
            name="PIP_RESULTS_SHOWN_IF_ASKED",
            passed=True,
            evidence=(
                "User prompt did not request permutation importance / PIP. "
                "(Check only fires when the prompt mentions it explicitly.)"
            ),
        )

    pip_output_present = bool(_PIP_OUTPUT_RE.search(transcript))
    if pip_output_present:
        return CheckResult(
            name="PIP_RESULTS_SHOWN_IF_ASKED",
            passed=True,
            evidence=(
                "User asked for PIP and the response contains the PIP "
                "DataFrame (variable + importance scores)."
            ),
        )

    return CheckResult(
        name="PIP_RESULTS_SHOWN_IF_ASKED",
        passed=False,
        evidence=(
            "User explicitly asked for PIP / permutation importance, but the "
            "response is missing the PIP DataFrame (variable name + importance "
            "score table). The expected pattern is: "
            "`pip_plot, pip_df = MODEL.plot('pip', ret=True); print(pip_df)` "
            "with the printed scores in the response. The Pi failure mode is "
            "to announce 'I'll generate the PIP plot' but then silently revert "
            "to interpreting regression coefficients without showing any PIP "
            "numbers — this check is designed to catch that."
        ),
    )


_PDP_REQUEST_RE = re.compile(
    r"\b(?:pdp|partial\s+dependence(?:\s+plot)?s?|"
    r"partial[- ]?dependency|ice\s+plot)\b",
    re.IGNORECASE,
)

# Evidence that a PDP plot was actually saved AND/OR that shape language is used.
_PDP_SAVED_PATTERNS = [
    r"\.plot\(\s*['\"]pdp['\"]",
    r"\.plot\(\s*['\"]pdp_sklearn['\"]",
    r"\bsaved\s+pdp",
    r"\bpdp\.png\b",
    r"\bpdp\.pdf\b",
    r"\bpdp_.{0,30}\.(?:png|pdf|svg)\b",
]
_PDP_SAVED_RE = re.compile("|".join(_PDP_SAVED_PATTERNS), re.IGNORECASE)

# Shape-describing language we EXPECT in a real PDP interpretation.
_PDP_SHAPE_PATTERNS = [
    r"\b(?:monotonic(?:ally)?|monotone)\b",
    r"\bnon[- ]monotonic\b",
    r"\bU[- ]shape(?:d)?\b",
    r"\binverted[- ]U\b",
    r"\bcurv(?:ed|ature|ing|e)\b",
    r"\bsaturate(?:s|d|ing)?\b",
    r"\bplateau(?:s|ed|ing)?\b",
    r"\bthreshold\b",
    r"\bflat (?:line|curve|trend)\b",
    r"\b(?:rises?|rising|falling|falls?|increases?|decreases?)\b\s+"
    r"(?:roughly\s+|approximately\s+)?(?:linearly|sharply|steadily|gradually|with)",
    r"\bslope\s+(?:is|changes|changes\s+sign)\b",
    r"\bsharp(?:ly)?\s+(?:rises?|falls?|jump(?:s)?)\b",
    # Direct shape verbs that wouldn't appear in a coefficient walk-through
    r"\b(?:rises|falls|climbs|drops|increases|decreases)\b\s+(?:from|across|"
    r"between|near|until|up to|toward|towards|to)\s+",
    # Reading-the-image markers
    r"\bopen(ed)?\s+pdp\.png\b",
    r"\bin the (?:pdp )?plot,\s+",
    r"\bcurves?\b.+\b(?:show|shows)\b",
]
_PDP_SHAPE_RE = re.compile("|".join(_PDP_SHAPE_PATTERNS), re.IGNORECASE)


def check_pdp_results_shown_if_asked(transcript: str, user_prompt: str | None) -> CheckResult:
    """Pass if (a) PDP was not requested, OR (b) PDP was requested AND
    the response shows evidence of saving the plot AND uses shape-describing
    language (not just coefficient interpretation).

    The Pi failure mode this catches: user asks for PDP, Pi says "I'll
    generate PDPs and then interpret the fitted results in business terms"
    and then produces a coefficient walk-through with no plot save and no
    PDP shape description.
    """
    if not user_prompt:
        return CheckResult(
            name="PDP_RESULTS_SHOWN_IF_ASKED",
            passed=True,
            evidence=(
                "No user prompt supplied; cannot tell whether PDP was "
                "requested. Pass --prompt to enable this check."
            ),
            severity="warn",
        )

    asked_for_pdp = bool(_PDP_REQUEST_RE.search(user_prompt))
    if not asked_for_pdp:
        return CheckResult(
            name="PDP_RESULTS_SHOWN_IF_ASKED",
            passed=True,
            evidence=(
                "User prompt did not request PDP / partial dependence. "
                "(Check only fires when the prompt mentions it explicitly.)"
            ),
        )

    plot_saved = bool(_PDP_SAVED_RE.search(transcript))
    shape_described = bool(_PDP_SHAPE_RE.search(transcript))

    if plot_saved and shape_described:
        return CheckResult(
            name="PDP_RESULTS_SHOWN_IF_ASKED",
            passed=True,
            evidence=(
                "User asked for PDP and the response (a) saves/references "
                "the PDP plot file AND (b) uses PDP shape-describing "
                "language (monotonic / curved / threshold / saturating / "
                "increases-across-range / etc.)."
            ),
        )

    missing = []
    if not plot_saved:
        missing.append("no call to .plot('pdp') and no saved pdp.png path")
    if not shape_described:
        missing.append(
            "no shape-describing language (monotonic / curved / threshold / "
            "U-shape / increases-across-range / etc.)"
        )
    return CheckResult(
        name="PDP_RESULTS_SHOWN_IF_ASKED",
        passed=False,
        evidence=(
            "User explicitly asked for PDP / partial dependence, but the "
            f"response is missing required content: {' AND '.join(missing)}. "
            "Expected pattern: call `.plot('pdp')`, save to disk with "
            "`p.save('pdp.png', ...)`, read the image, and describe the "
            "shape of each predictor's curve. The Pi failure mode is to "
            "announce 'I'll generate PDPs and interpret the fitted results "
            "in business terms' but then produce a coefficient walk-through "
            "with no plot save and no shape description — this check is "
            "designed to catch that."
        ),
    )


def run_checks(transcript: str, user_prompt: str | None = None) -> list[CheckResult]:
    return [
        check_no_bad_summary_flags(transcript, user_prompt),
        check_full_output_shown(transcript),
        check_not_lame_bullets(transcript),
        check_importance_disclaimer(transcript),
        check_pip_results_shown_if_asked(transcript, user_prompt),
        check_pdp_results_shown_if_asked(transcript, user_prompt),
    ]


# -------- CLI --------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "transcript",
        help="Path to a transcript file (use '-' to read from stdin).",
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Optional path to the user prompt that produced the transcript. "
            "Used to allow summary(rmse=True, ssq=True) when the user asked. "
            "Pass '-' to read prompt from stdin (mutually exclusive with "
            "transcript stdin)."
        ),
    )
    parser.add_argument(
        "--skill",
        help="Optional skill name for context (e.g. pyrsm-regress). Not "
        "currently used for branching but printed in the output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    if args.transcript == "-":
        transcript = sys.stdin.read()
    else:
        transcript_path = Path(args.transcript)
        if not transcript_path.exists():
            print(f"error: transcript file not found: {transcript_path}", file=sys.stderr)
            return 2
        transcript = transcript_path.read_text(encoding="utf-8", errors="replace")

    prompt_text: str | None = None
    if args.prompt:
        if args.prompt == "-":
            if args.transcript == "-":
                print("error: cannot read both transcript and prompt from stdin.", file=sys.stderr)
                return 2
            prompt_text = sys.stdin.read()
        else:
            prompt_path = Path(args.prompt)
            if prompt_path.exists():
                prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace")

    results = run_checks(transcript, prompt_text)

    if args.json:
        payload = {
            "skill": args.skill,
            "transcript_chars": len(transcript),
            "checks": [
                {"name": r.name, "passed": r.passed, "evidence": r.evidence} for r in results
            ],
            "pass_count": sum(1 for r in results if r.passed),
            "fail_count": sum(1 for r in results if not r.passed),
            "overall": "pass" if all(r.passed for r in results) else "fail",
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"=== check_transcript: {args.skill or '(skill unspecified)'} ===")
        print(f"  transcript: {len(transcript)} chars")
        if prompt_text:
            print(f"  prompt provided: {len(prompt_text)} chars")
        print()
        for r in results:
            mark = "✓" if r.passed else "✗"
            print(f"  [{mark}] {r.name}")
            print(f"      {r.evidence}")
            print()
        overall = "PASS" if all(r.passed for r in results) else "FAIL"
        print(f"  Overall: {overall}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
