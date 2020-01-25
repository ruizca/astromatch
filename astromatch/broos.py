"""
Functions for statistics calculations following the Broos et al. 2006 method.

@author: A.Ruiz
"""
import numpy as np
from astropy.table import Table


def set_stats_table(ncutoff=101, mincutoff=0.0, maxcutoff=10.0):
    stats = Table()
    stats["cutoff"] = np.linspace(mincutoff, maxcutoff, num=ncutoff)
    stats["false_positives"] = [0] * ncutoff    # FP
    stats["true_negatives"] = [0] * ncutoff     # TN
    stats["correct_matches"] = [0] * ncutoff    # CM
    stats["incorrect_matches"] = [0] * ncutoff  # IM
    stats["false_negatives"] = [0] * ncutoff    # FN

    return stats


def _count_true_negatives_base(match):
    # Number of primary sources with no match
    # This base number have to be corrected including
    # the number of rejected matches after applying a cutoff
    mask_true_negatives_base = np.logical_and(
        match["ncat"] == 1, match["match_flag"] == 1
    )

    return len(match[mask_true_negatives_base])


def _count_true_negatives_correction(match, cutoff, cutoff_column):
    # Number of primary sources with a match but below the cutoff limit
    # This correction have to be added to the true negatives base number
    mask_below_cutoff = match[cutoff_column] <= cutoff

    mask_true_negatives_correction = np.logical_and(
        match["ncat"] == 2, match["match_flag"] == 1
    )
    mask_true_negatives_correction = np.logical_and(
        mask_below_cutoff, mask_true_negatives_correction
    )

    return len(match[mask_true_negatives_correction])


def _count_false_positives(match, cutoff, cutoff_column):
    # Number of primary sources with a match above the cutoff limit
    mask_above_cutoff = match[cutoff_column] > cutoff

    mask_false_positives = np.logical_and(match["ncat"] == 2, match["match_flag"] == 1)
    mask_false_positives = np.logical_and(mask_above_cutoff, mask_false_positives)

    return len(match[mask_false_positives])


def stats_isolated_pop(match, stats, cutoff_column):
    """
    Statistics for the isolated population:
    true negatives and false positives.
    """
    stats["true_negatives"] = _count_true_negatives_base(match)

    for i, lrlim in enumerate(stats['cutoff']):
        stats["false_positives"][i] = _count_false_positives(
            match, lrlim, cutoff_column
        )
        stats["true_negatives"][i] += _count_true_negatives_correction(
            match, lrlim, cutoff_column
        )

    return stats


def _count_false_negatives_base(match):
    # Number of primary sources without a match in the fake counterparts catalogue
    # This number has to be corrected by the number of rejected correct matches
    # when a cutoff is applied to the match.
    mask_false_negatives_base = np.logical_and(
        match["match_flag"] == 1, match["ncat"] == 1
    )

    return len(match[mask_false_negatives_base])


def _count_false_negatives_correction(match, cutoff, cutoff_column, pcat_id, scat_id):
    # Number of correct matches in the fake counterparts catalogue rejected
    # when a cutoff is applied to the match. This has to be added to the
    # base number of false negatives.
    mask_below_cutoff = match[cutoff_column] <= cutoff

    mask_false_negatives_correction = np.logical_and(
        match["match_flag"] == 1, match[pcat_id] == match[scat_id]
    )
    mask_false_negatives_correction = np.logical_and(
        mask_below_cutoff, mask_false_negatives_correction
    )

    return len(match[mask_false_negatives_correction])


def _count_correct_matches(match, cutoff, cutoff_column, pcat_id, scat_id):
    # Number of correct matches in the fake counterparts catalogue
    # above a given cutoff
    mask_above_cutoff = match[cutoff_column] > cutoff

    mask_correct_matches = np.logical_and(
        match["match_flag"] == 1,
        match[pcat_id] == match[scat_id]
    )
    mask_correct_matches = np.logical_and(mask_above_cutoff, mask_correct_matches)

    return len(match[mask_correct_matches])


def _count_incorrect_matches(match, cutoff, cutoff_column, pcat_id, scat_id):
    # Number of incorrect matches in the fake counterparts
    # catalogue above a given cutoff
    mask_above_cutoff = match[cutoff_column] > cutoff

    mask_incorrect_matches = np.logical_and(
        match["match_flag"] == 1, match[pcat_id] != match[scat_id]
    )
    mask_incorrect_matches = np.logical_and(
        mask_incorrect_matches, match["ncat"] == 2
    )
    mask_incorrect_matches = np.logical_and(mask_above_cutoff, mask_incorrect_matches)

    return len(match[mask_incorrect_matches])


def stats_associated_pop(match, stats, cutoff_column, pcat_id, scat_id):
    """
    Statistics for the associated population:
    correct matches, incorrect matches and false negatives.
    """
    stats["false_negatives"] = _count_false_negatives_base(match)

    for i, cutoff in enumerate(stats['cutoff']):
        stats["correct_matches"][i] = _count_correct_matches(
            match, cutoff, cutoff_column, pcat_id, scat_id
        )
        stats["incorrect_matches"][i] = _count_incorrect_matches(
            match, cutoff, cutoff_column, pcat_id, scat_id
        )
        stats["false_negatives"][i] += _count_false_negatives_correction(
            match, cutoff, cutoff_column, pcat_id, scat_id
        )

    return stats


def _count_negative_matches_base(match):
    # Number of sources with no match in the actual crossmatch results.
    # This has to be corrected by the number of rejected matches
    # when a cutoff is applied.
    mask_negative_matches_base = np.logical_and(
        match["match_flag"] == 1, match["ncat"] == 1
    )

    return len(match[mask_negative_matches_base])


def _count_negative_matches_correction(match, cutoff, cutoff_column):
    mask_below_cutoff = match[cutoff_column] <= cutoff

    mask_negative_matches_correction = np.logical_and(
        match["ncat"] == 2, match["match_flag"] == 1
    )
    mask_negative_matches_correction = np.logical_and(
        mask_below_cutoff, mask_negative_matches_correction
    )

    return len(match[mask_negative_matches_correction])


def _count_positive_matches(match, cutoff, cutoff_column):
    mask_above_cutoff = match[cutoff_column] > cutoff

    mask_positive_matches = np.logical_and(match["ncat"] == 2, match["match_flag"] == 1)
    mask_positive_matches = np.logical_and(mask_above_cutoff, mask_positive_matches)

    return len(match[mask_positive_matches])


def _completeness(fstats):
    P = fstats["positive_matches"]
    N = fstats["negative_matches"]

    return P / (P + N)


def _frac_associated_pop(fstats):
    # Associated fraction (FA): N = FN*FA + TN*(1 - FA),
    # where N is the number of negative matches (primary sources with no match)
    N = fstats["negative_matches"]
    TN = fstats["true_negatives"]
    FN = fstats["false_negatives"]

    return (N - TN) / (FN - TN)


def _error_rate(fstats):
    # If we know FA, we can calculate the number of false matches (F):
    # F = IM*FA + FP*(1 - FA), and the error rate would be F/M,
    # where M is the number of matches (primary sources with a match)
    FA = fstats["frac_assoc_pop"]
    IM = fstats["incorrect_matches"]
    FP = fstats["false_positives"]
    P = fstats["positive_matches"]

    return (IM*FA + FP*(1 - FA)) / P


def stats_global_pop(match, stats, cutoff_column, ntest):
    for col in stats.colnames[1:]:
        stats[col] = stats[col] / ntest

    stats["negative_matches"] = _count_negative_matches_base(match)
    stats["positive_matches"] = 0

    for i, cutoff in enumerate(stats['cutoff']):
        stats["positive_matches"][i] = _count_positive_matches(
            match, cutoff, cutoff_column
        )
        stats["negative_matches"][i] += _count_negative_matches_correction(
            match, cutoff, cutoff_column
        )

    stats["frac_assoc_pop"] = _frac_associated_pop(stats)
    stats["completeness"] = _completeness(stats)
    stats["error_rate"] = _error_rate(stats)
    stats["reliability"] = 1 - stats["error_rate"]
    stats["CR"] = stats["completeness"] + stats["error_rate"]

    return stats
