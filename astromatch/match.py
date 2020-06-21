"""
astromatch module with the base Match class.

@author: A.Ruiz
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

from inspect import signature

from astropy import log
from astropy import units as u
from astropy.table import Table, join
import numpy as np

from . import broos
from .priors import Prior


class BaseMatch(object):
    """
    Base class for Match classes.
    """
    def __init__(self, *args):
        self.catalogues = list(args)

        self.pcat = args[0]
        self.scats = list(args[1:])
        self._priors = None

    ### Class Properties
    @property
    def priors(self):
        if self._priors is None:
            raise AttributeError('No magnitude priors defined yet!')
        else:
            try:
                if len(self.scats) == 1:
                    return self._priors[self.scats[0].name]
                else:
                    return self._priors

            except TypeError:
                return self._priors

    ### Public methods
    def stats(
        self,
        match,
        ncutoff=101,
        mincutoff=0.0,
        maxcutoff=1.0,
        plot_to_file=None
    ):
        """
        Calculates and store match statistics (completness and reliability)
        for a range of thresholds. This can be used later to select the
        optimal threshold.
        """
        # parametrise common statistics/definitions that
        # quantify the reliability and completeness of
        # the cross-match.
        # Produce table with summary of results

        # We use only primary matches
        mask = match['match_flag'] == 1

        pdata = match[mask]
        reliability = self._reliability(pdata)

        stats = Table()
        stats['cutoff'] = np.linspace(mincutoff, maxcutoff, num=ncutoff)
        stats['completeness'] = np.nan
        stats['reliability'] = np.nan

        for i, plim in enumerate(stats['cutoff']):
            rel_good = reliability[pdata[self._cutoff_column] > plim]
            stats['completeness'][i] = float(rel_good.size)/len(pdata) # AGE
            stats['reliability'][i] = np.mean(rel_good)

        stats['error_rate'] = 1.0 - stats['reliability']
        stats['CR'] = stats['completeness'] + stats['reliability']

        if plot_to_file is not None:
            self._plot_stats(stats, plot_to_file)

        return stats

    def stats_rndmatch(
        self,
        match,
        match_rnd,
        ncutoff=101,
        mincutoff=0.0,
        maxcutoff=1.0,
        plot_to_file=None,
        **kwargs
    ):
        """
        Calculates match statistics (completness and reliability), using a
        random match, for a range of thresholds. This can be used later to
        select the optimal threshold.
        """
        mask = match['ncat'] == 1
        p_any0 = match['prob_has_match'][mask]

        mask = match_rnd['ncat'] == 1
        p_any0_offset = match_rnd['prob_has_match'][mask]

        cutoffs = np.linspace(mincutoff, maxcutoff, num=ncutoff)

        stats = Table()
        stats['cutoff'] = cutoffs
        stats['completeness'] = [(p_any0 > c).mean() for c in cutoffs]
        stats['error_rate'] = [(p_any0_offset > c).mean() for c in cutoffs]
        stats['reliability'] = 1 - stats['error_rate']
        stats['CR'] = stats['completeness'] + stats['reliability']

        if plot_to_file is not None:
            self._plot_stats(stats, plot_to_file)

        return stats

    def stats_broos(
        self,
        match,
        ntest=1,
        ncutoff=101,
        mincutoff=0.0,
        maxcutoff=1.0,
        plot_to_file=None,
        **kwargs
    ):
        """
        Calculates and store match statistics (completness and reliability)
        for a range of thresholds. We use here the Monte Carlo method
        presented in Broos et al. 2006, where isolated and associated populations
        of the primary catalogue are treated independently.
        """
        # TODO: incomplete implementation, only tested with LR method (two catalogues).
        if len(self.catalogues) > 2:
            raise NotImplementedError(
                'Broos method only implemented for two-catalogue cross-matchs'
            )

        # TODO: estimate uncertainties in the final statistics using e.g. bootstraping
        fstats = None
        for _ in range(ntest):
            stats = broos.set_stats_table(
                ncutoff=ncutoff, mincutoff=mincutoff, maxcutoff=maxcutoff
            )

            ## Isolated population
            # We crossmatch with a randomized pcat (nway randomization),
            # using priors calculated for the real match
            match_rnd = self._match_rndcat(**kwargs)
            stats = broos.stats_isolated_pop(match_rnd, stats, self._cutoff_column)

            ## Associated population
            # We crossmatch with a randomized pcat (nway randomization)
            # and a scat with fake counterparts for the pcat,
            # using priors calculated for the real match

            # Identify candidates for fake counterparts
            mask_candidates = np.logical_and(
                match["match_flag"] == 1, match["prob_has_match"] > 0.9
            )
            ids_candidates = match.columns[1][mask_candidates]
            candidates = self.scats[0].select_by_id(ids_candidates)

            # Run match with fake counterparts and calculate stats.
            # We use again the priors calculated for the real match
            match_rnd = self._match_fake(candidates, **kwargs)
            stats = broos.stats_associated_pop(
                match_rnd,
                stats,
                self._cutoff_column,
                match_rnd.colnames[0],
                match_rnd.colnames[1],
            )

            if fstats is None:
                fstats = stats.copy()
            else:
                for col in fstats.colnames[1:]:
                    fstats[col] += stats[col]

        fstats = broos.stats_global_pop(match, fstats, self._cutoff_column, ntest)

        if plot_to_file is not None:
            self._plot_stats(fstats, plot_to_file)

        return fstats

    def flag_best_match(self, match, cutoff=None, match_rnd=None):
        if cutoff is None:
            if match_rnd is None:
                stats = self.stats(match)
            else:
                stats = self.stats_rndmatch(match, match_rnd)

            cutoff = self._calc_cutoff(stats)

        mask = np.logical_and(match[self._cutoff_column] > cutoff,
                              match['match_flag'] == 1)

        match['best_match_flag'] = 0
        match['best_match_flag'][mask] = 1

        cutoff_str = '{} > {}'.format(self._cutoff_column, cutoff)
        match.meta['best_match_cutoff'] = cutoff_str

        return cutoff

    def parse_args(self, kwargs):
        sig = signature(Prior)
        kwargs_prior_default = [p for p in sig.parameters]

        kwargs_prior, kwargs_run = {}, {}
        for key, value in kwargs.items():
            if key in kwargs_prior_default:
                kwargs_prior[key] = value
            else:
                kwargs_run[key] = value

        kwargs_prior["radius"] = kwargs_run.pop('mag_include_radius', 6*u.arcsec)
        kwargs_prior["priors"] = kwargs_run.pop('priors', None)

        return kwargs_prior, kwargs_run

    ### Internal methods
    def _calc_cutoff(self, stats, false_rate=None):
        if false_rate is None:
            i = np.nanargmax(stats['CR'])
            p_cutoff = stats['cutoff'][i]

            print('Optimal threshold is {}:'.format(p_cutoff))
            message = ('Selects ~{:.0f}% of matches with '
                       'a false detection rate of < {:.0f}%')
            print(message.format(stats['completeness'][i]*100,
                                 stats['error_rate'][i]*100))
        else:
            mask = stats['error_rate'] < false_rate
            if not mask.any():
                message = 'A false detection rate of {}% is not possible.'
                print(message.format(false_rate*100))

                return None
            else:
                i = np.min(np.where(mask)[0])
                p_cutoff = stats['cutoff'][i]

                print('For a false detection rate of < {}%'.format(false_rate*100))
                message = ('--> use only counterparts with cutoff > '
                           '{:.2f} (~{:.0f}% of matches)')
                print(message.format(p_cutoff, stats['completeness'][i]*100))

        return p_cutoff

    def _sort_as_pcat(self, match, pcat_idcol):
        # Sort match table as in the primary catalogue,
        # as given by the pcat_idcol column.
        pos_table = Table()
        pos_table[pcat_idcol] = self.pcat.ids
        pos_table['PORDER'] = list(range(len(self.pcat)))

        match = join(match, pos_table, keys=pcat_idcol, join_type='left')
        match['temp'] = -match['prob_this_match']  # For sorting in decreasing order
        match.sort(['PORDER', 'ncat', 'temp'])
        match.remove_columns(['PORDER', 'temp'])

        # Change SRCIDs of secondary sources for sources
        #  with no match from '0.0' to ''
        for cat in self.scats:
            idcol = 'SRCID_{}'.format(cat.name)
            try:
                mask = np.char.strip(match[idcol]) == b'0.0'
                match[idcol][mask] = ''
            except ValueError:
                continue

        return match

    @staticmethod
    def _plot_stats(stats, plotfile):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6,5))
        plt.plot(stats['cutoff'], stats['completeness'], '-', color='k', lw=2,
                 label='completeness')
        plt.plot(stats['cutoff'], stats['error_rate'], '--', color='r',
                 label='false selection rate')
        plt.plot(stats['cutoff'], 0.5*stats['CR'], ':', color='g',
                 label='(C+R)/2')

        plt.ylabel('fraction(> cutoff)')
        plt.xlabel('cutoff')
        plt.legend(loc='lower left', prop=dict(size=10))

        plt.savefig(plotfile, bbox_inches='tight')
        plt.close()

        log.info('created plot "{}"'.format(plotfile))
