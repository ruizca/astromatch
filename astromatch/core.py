"""
Main module of astromatch.

author: A.Ruiz
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from astropy import log
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning

from .catalogues import Catalogue
from .lr import LRMatch
from .xmatch import XMatch


class Match(object):
    """
    Main class for crossmatching catalogues.
    """
    def __init__(self, *args, **kwargs):
        self.catalogues = []
        for i, cat_data in enumerate(args):
            if isinstance(cat_data, Catalogue):
                cat = cat_data
            else:
                try:
                    cat_params = {kw: kwargs[kw][i] for kw in kwargs}
                except IndexError:
                    message = 'Some parameters were not defined for catalogue {}!'
                    raise ValueError(message.format(i))

                cat = Catalogue(cat_data, **cat_params)

            self.catalogues.append(cat)

        self._match = None
        self._result = None
        self._priors = None


    @property
    def results(self):
        if self._result is None:
            raise AttributeError('Match has not been performed yet!')
        else:
            return self._result

    @property
    def priors(self):
        if self._match is None:
            raise AttributeError('Match has not been performed yet!')
        else:
            return self._match.priors

    @property
    def priorsND(self):
        if self._match is None:
            raise AttributeError('Match has not been performed yet!')
        else:
            return self._match.priorsND

    @property
    def lr(self):
        if isinstance(self._match, LRMatch):
            return self._match.lr
        else:
            raise AttributeError

    @property
    def bkg(self):
        if isinstance(self._match, LRMatch):
            return self._match.bkg
        else:
            raise AttributeError

    @property
    def xmatch_raw(self):
        if isinstance(self._match, XMatch):
            return self._match.results_raw
        else:
            raise AttributeError

    def run(self, method='lr', **kwargs):
        """
        Cross-matching between catalogues, using the method defined in `method`.

        If the catalogues MOCs are defined, the crossmatch is done in the common
        region defined by these MOCs. Otherwise the method assumes that all
        catalogues cover the same sky area, as defined by the `area` parameter).

        Parameters
        ----------
        method : 'lr', 'nway' or 'xmatch
            Cross-matching method. Defaults to 'lr'

        Other Parameters
        ----------------
        **kwargs : specific parameters for the selected cross-matching method.
            See the method documentation for more information.
        """
        total_moc = self.total_moc()

        # Crossmatch only for sources cointained in the intersection of mocs,
        # if no mocs were defined, we assume that all catalogues cover the same sky area
        if total_moc is not None:
            catalogues_inmoc = []
            for cat in self.catalogues:
                catalogues_inmoc.append(cat.apply_moc(total_moc))

            self.catalogues = catalogues_inmoc

        # Run the crossmatch with the defined method
        method_name = '_{}__{}'.format(self.__class__.__name__, method)
        match_method = getattr(self, method_name)
        self._result = match_method(**kwargs)

        return self._result

    def total_moc(self):
        mocs = [cat.moc for cat in self.catalogues if cat.moc is not None]

        if len(mocs) == 1:
            # Assumes that all catalogues cover the same
            # area as the catalogue with a defined moc
            moc = mocs[0]

        elif len(mocs) > 1:
            moc = mocs[0].intersection(*mocs[1:])

        else:
            warnings.warn('No MOCs defined for the catalogues!')
            moc = None

        return moc

    def get_matchs(self, match_type='all'):
        """
        Returns an Astropy Table with matches for the primary catalogue.

        Parameters
        ----------
        match_type : 'all', 'primary_all', 'primary' or 'best'.
            Set of matches to be returned. Defaults to 'all'.

            'all' : All possible matches.

            'primary' : Return only the matches with the most likely counterpart.

            'primary_all' : Return only the matches with the most likely counterpart,
            including primary sources with no counterpart.

            'best' : Return only the matches with the most likely counterpart and above
            a likelihood limit. The likelihood limit is set through the `set_best_matchs`
            method.
        """
        if self._result is None:
            raise AttributeError('Match has not been performed yet!')

        if match_type not in ['all', 'primary_all', 'primary', 'best']:
            raise ValueError('Unknown match type: {}'.format(match_type))

        if match_type == 'primary_all':
            mask = self._result['match_flag'] == 1
            matchs = self._result[mask]

        else:
            mask = self._result['ncat'] > 1
            matchs = self._result[mask]

            if match_type == 'primary':
                mask = matchs['match_flag'] == 1
                matchs = matchs[mask]

            elif match_type == 'best':
                try:
                    mask = matchs['best_match_flag'] == 1
                    matchs = matchs[mask]
                except KeyError:
                    raise ValueError('Best matchs not identified yet!')

        return matchs

    def offset(self, pcat_name, scat_name, only_best=False):
        cat_names = self._catalogue_names()
        if not (set([pcat_name, scat_name]) <= set(cat_names)):
            raise ValueError('Unknown catalogue name.')

        if only_best:
            match_type = 'best'
        else:
            match_type = 'all'

        matchs = self.get_matchs(match_type=match_type)

        pcat = self.catalogues[cat_names.index(pcat_name)]
        pcat_srcid = matchs['SRCID_' + pcat_name]
        pcat_coords = pcat.select_by_id(pcat_srcid).coords

        scat = self.catalogues[cat_names.index(scat_name)]
        scat_srcid = matchs['SRCID_' + scat_name]
        scat_coords = scat.select_by_id(scat_srcid).coords

        dra, ddec = pcat_coords.spherical_offsets_to(scat_coords)

        return dra.to(u.arcsec), ddec.to(u.arcsec)

    def set_best_matchs(
        self, cutoff=None, false_rate=None, calibrate_with_random_cat=False, **kwargs,
    ):

        if self._result is None:
            raise AttributeError('Match has not been performed yet!')

        match_rnd = None
        if cutoff is None:
            cutoff, match_rnd  = self._calibrate_best_match(
                false_rate, calibrate_with_random_cat, **kwargs
            )

        self._match.flag_best_match(self._result, cutoff=cutoff)

        return match_rnd

    def stats(self, match_rnd=False, use_broos=False, **kwargs):
        """
        Calculate statistics for the match: completeness, reliability and
        other associated parameters to characterize the quality of the
        cross-matching. There are several methods implemented:

        Probabilistic: uses the probabilities calculated by the matching method.

        Monte Carlo: simple estimation of statistics based on matching with
        a randomly shifted primary catalogue.

        Monte Carlo 2: follows the method described in Broos et al. 2006, where
        MC simulations are used to characterize the properties of the isolated and
        the associated populations of the primary catalogue.
        """
        if use_broos:
            stats = self._match.stats_broos(self.results, **kwargs)

        else:
            if match_rnd is True:
                match_rnd = self._match._match_rndcat(**kwargs)
                stats = self._match.stats_rndmatch(self.results, match_rnd, **kwargs)

            elif isinstance(match_rnd, Catalogue):
                stats = self._match.stats_rndmatch(self.results, match_rnd, **kwargs)

            else:
                stats = self._match.stats(self.results, **kwargs)

        return stats


    def __lr(self, **kwargs):
        log.info('Using LR method:')

        if len(self.catalogues) > 2:
            warnings.warn(
                'Only the first two catalogues are cross-matched using LR!!!',
                AstropyUserWarning
            )

        # Our current implementation of the LR method always assumes 1-sigma
        # circular positional errors. Hence, we transform non-circular errors
        # to circular errors of equal area. We pass only the firs two catalogues,
        # other catalogues are ignored while using this method.
        catalogues_circular = self._poserrs_to_circle(self.catalogues[:2])

        self._match = LRMatch(*catalogues_circular)

        return self._match.run(**kwargs)

    def __nway(self, **kwargs):
        from .nway import NWMatch

        log.info('Using NWAY method:')

        # The nway method always assumes 1-sigma circular positional errors.
        # Hence, we transform non-circular errors to circular errors of equal area.
        catalogues_circular = self._poserrs_to_circle(self.catalogues)

        self._match = NWMatch(*catalogues_circular)

        return self._match.run(**kwargs)

    def __xmatch(self, **kwargs):
        log.info('Using XMatch method:')

        self._match = XMatch(*self.catalogues)

        return self._match.run(**kwargs)

    def _catalogue_names(self):
        return [cat.name for cat in self.catalogues]

    def _calibrate_best_match(self, false_rate, calibrate_with_random_cat, **kwargs):

        if calibrate_with_random_cat:
            log.info('Calibrating probability cutoff using a random match...')
            match_rnd = self._match._match_rndcat(**kwargs)
            stats = self._match.stats_rndmatch(self._result, match_rnd)
        else:
            match_rnd = None
            stats = self._match.stats(self._result)

        cutoff = self._match._calc_cutoff(stats, false_rate)

        return cutoff, match_rnd

    @staticmethod
    def _poserrs_to_circle(catalogues):
        """
        Convert positional errors of the catalogues in args to circular errors.
        """
        newcats = []
        for cat in catalogues:
            if cat.poserr.errtype != 'circle':
                errs = cat.poserr.transform_to('circle')
                cat.poserr = errs

            newcats.append(cat)

        return newcats
