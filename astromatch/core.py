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


    def run(self, method='lr', **kwargs):
        """
        Cross-matching between catalogues, using the method defined in `method`.

        If the catalogues MOCs are defined, the crossmatch is done in the common
        region defined by these MOCs. Otherwise the method assumes that all 
        catalogues cover the same sky area, as defined by the `area` parameter).
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
        
        if len(mocs):            
            moc = mocs[0].intersection(*mocs[1:])
        else:
            warnings.warn('No MOCs defined for the catalogues!')
            moc = None

        return moc

    def get_matchs(self, match_type='all'):
        if self._result is None:
            raise AttributeError('Match has not been performed yet!')

        else:
            if match_type not in ['all', 'primary', 'best']:
                raise ValueError('Unknown match type: {}'.format(match_type))

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
            match_type = 'primary'

        matchs = self.get_matchs(match_type=match_type)

        pcat = self.catalogues[cat_names.index(pcat_name)]
        pcat_srcid = matchs['SRCID_' + pcat_name]
        pcat_coords = pcat.select_by_id(pcat_srcid).coords

        scat = self.catalogues[cat_names.index(scat_name)]
        scat_srcid = matchs['SRCID_' + scat_name]
        scat_coords = scat.select_by_id(scat_srcid).coords

        dra, ddec = pcat_coords.spherical_offsets_to(scat_coords)

        return dra.to(u.arcsec), ddec.to(u.arcsec)        

    def set_best_matchs(self, cutoff=None, false_rate=None, 
                        calibrate_with_random_cat=False, **kwargs):

        if self._result is None:
            raise AttributeError('Match has not been performed yet!')

        match_rnd = None
        if cutoff is None:
            cutoff, match_rnd  = self._calibrate_best_match(false_rate,
                                            calibrate_with_random_cat, **kwargs)

        self._match.flag_best_match(self._result, cutoff=cutoff)
        #self._flag_best_match(cutoff, calibrate_with_random_cat)

        return match_rnd
        
    def stats(self, match_rnd=None, ncutoff=101, plot_to_file=None):
        if match_rnd is None:
            stats = self._match.stats(self.results, ncutoff=ncutoff,
                                      plot_to_file=plot_to_file)
        else:
            stats = self._match.stats_rndmatch(self.results, match_rnd,
                                               ncutoff=ncutoff,
                                               plot_to_file=plot_to_file)
        return stats


    def __lr(self, **kwargs):
        log.info('Using LR method:')
        
        if len(self.catalogues) > 2:
            message = 'Only the first two catalogues are cross-matched using LR!!!'
            warnings.warn(message, AstropyUserWarning)

        # Our current implementation of the LR method always assumes 1-sigma
        # circular positional errors. Hence, we transform non-circular errors
        # to circular errors of equal area. We pass only the firs two catalogues,
        # other catalogues are ignored while using this method.
        catalogues_circular = self._poserrs_to_circle(self.catalogues[:2]) 

        self._match = LRMatch(*catalogues_circular)
        results = self._match.run(**kwargs)
        
        # Assign some attributes and methods from LRMatch to Match class
        # TODO: is this a good idea???
        self.lr = self._match.lr
        self.bkg = self._match.bkg

        return results

    def __nway(self, **kwargs):
        from .nway import NWMatch

        log.info('Using NWAY method:')

        # The nway method always assumes 1-sigma circular positional errors. 
        # Hence, we transform non-circular errors to circular errors of equal area. 
        catalogues_circular = self._poserrs_to_circle(self.catalogues) 

        self._match = NWMatch(*catalogues_circular)
        results = self._match.run(**kwargs)

        return results
    
    def __xmatch(self, **kwargs):
        log.info('Using XMatch method:')

        self._match = XMatch(*self.catalogues)
        results = self._match.run(**kwargs)
        self.results_raw = self._match.results_raw
        
        return results
    
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

#    def _flag_best_match(self, cutoff, calibrate_with_random_cat):
#        self._match.flag_best_match(self._result, cutoff=cutoff)

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
