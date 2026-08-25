import numpy as np

from wetlandbirds.statistics.significance import holm_bonferroni, mcnemar


def test_mcnemar_no_discordance():
    y = np.array([0, 1, 1])
    assert mcnemar(y, y, y)["p_value"] == 1.0


def test_holm_is_bounded():
    adjusted = holm_bonferroni([0.001, 0.02, 0.5])
    assert np.all((adjusted >= 0) & (adjusted <= 1))
