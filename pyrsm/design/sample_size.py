import math

from scipy.stats import norm

from pyrsm.utils import format_nr
from pyrsm.basics import display_utils as du


class sample_size:
    """
    Sample size calculation for means or proportions.

    Parameters
    ----------
    type : str
        Choose "mean" or "proportion"
    err_mean : float
        Acceptable error for mean
    sd_mean : float
        Standard deviation for mean
    err_prop : float
        Acceptable error for proportion
    p_prop : float
        Initial proportion estimate for proportion
    conf : float
        Confidence level
    incidence : float
        Incidence rate (fraction of valid respondents)
    response : float
        Response rate
    pop_correction : bool
        Apply correction for population size
    pop_size : int
        Population size

    Attributes
    ----------
    n : int
        Required sample size (ceiling)
    contact_attempts : int
        Required contact attempts adjusted for incidence and response rates

    Examples
    --------
    >>> ss = sample_size("mean", err_mean=2, sd_mean=10)
    >>> ss.n
    97
    >>> ss = sample_size("proportion", err_prop=0.1, p_prop=0.5)
    >>> ss.n
    97
    """

    def __init__(
        self,
        type: str = "mean",
        err_mean: float = 2,
        sd_mean: float = 10,
        err_prop: float = 0.1,
        p_prop: float = 0.5,
        conf: float = 0.95,
        incidence: float = 1,
        response: float = 1,
        pop_correction: bool = False,
        pop_size: int = 1_000_000,
    ):
        self.type = type
        self.err_mean = err_mean
        self.sd_mean = sd_mean
        self.err_prop = err_prop
        self.p_prop = p_prop
        self.conf = conf
        self.incidence = incidence
        self.response = response
        self.pop_correction = pop_correction
        self.pop_size = pop_size

        if conf is None or conf < 0 or conf > 1:
            conf = 0.95
            self.conf = conf

        zval = -norm.ppf((1 - conf) / 2)

        if type == "mean":
            n = (zval**2 * sd_mean**2) / (err_mean**2)
        else:
            n = (zval**2 * p_prop * (1 - p_prop)) / (err_prop**2)

        if pop_correction:
            n = n * pop_size / ((n - 1) + pop_size)

        self.n = math.ceil(n)
        self.contact_attempts = math.ceil(self.n / incidence / response)

    def summary(self, dec: int = 3, plain: bool = True) -> None:
        """Print summary of sample size calculation.

        ``plain=False`` in a notebook (or a rendered Quarto report) shows the formatted report Radiant shows.
        """
        if not plain and du.is_notebook():
            # Radiant's formatted report (pyrsm.report), as its Summary tab shows it
            return du.show_text_report(self, "sample_size", dec=dec)

        print("Sample size calculation")

        def num(x):
            # 10.0 -> "10", 60.95 -> "60.95"
            x = round(float(x), dec)
            return str(int(x)) if x.is_integer() else str(x)

        if self.type == "mean":
            print("Calculation type     : Mean")
            print(f"Acceptable error     : {num(self.err_mean)}")
            print(f"Standard deviation   : {num(self.sd_mean)}")
        else:
            print("Calculation type     : Proportion")
            print(f"Acceptable error     : {num(self.err_prop)}")
            print(f"Proportion           : {num(self.p_prop)}")

        print(f"Confidence level     : {num(self.conf)}")
        print(f"Incidence rate       : {num(self.incidence)}")
        print(f"Response rate        : {num(self.response)}")

        if not self.pop_correction:
            print("Population correction: None")
        else:
            print("Population correction: Yes")
            print(f"Population size      : {format_nr(self.pop_size, dec=0)}")

        print(f"\nRequired sample size     : {format_nr(self.n, dec=0)}")
        print(f"Required contact attempts: {format_nr(self.contact_attempts, dec=0)}")
