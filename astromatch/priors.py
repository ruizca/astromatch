# -*- coding: utf-8 -*-
"""
astromatch module for calculation of magnitude priors.

@author: A.Ruiz
"""
import os
import warnings

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from scipy.interpolate import interp1d
from scipy.ndimage import convolve


from .catalogues import Catalogue


class Prior(object):
    """
    Class for probability priors.
    """

    def __init__(
        self,
        pcat=None,
        scat=None,
        rndcat=True,
        radius=5*u.arcsec,
        magmin='auto',
        magmax='auto',
        magbinsize='auto',
        match_mags=None,
        prior_dict=None
    ):
        """
        Estimates the prior probability distribution for a source in the
        primary catalogue `pcat` having a counterpart in the secondary
        catalogue `scat` with magnitude m.

        The a priori probability is determined as follows. First, we estimate
        the magnitude distribution of the spurious matches and it is scaled
        to the area within which we search for counterparts. This is then
        subtracted from the magnitude distribution of all counterparts in the
        secondary catalogue to determine the magnitude distribution of the
        true associations.

        Parameters
        ----------
        pcat, scat : ``Catalogue``
        rndcat : `boolean`, optional
            We implemented two methods for estimating the magnitude distribution
            of spurious matches: If 'rndcat' is ``False``, it removes all sources
            in the secondary catalogue within one arcmin of the positions of the
            primary sources. The magnitude distribution of the remaining sources,
            divided by the remaining catalogue area, corresponds to the
            probability distribution of a spurious match per magnitude and per
            square degree.
            If 'rndcat' is ``True``, it generates a catalogue of random
            positions away from the primary sources and searchs for all available
            counterparts in the secondary catalogue. The magnitude distribution
            of these sources corresponds to the probability distribution of a
            spurious match.
        radius : Astropy ``Quantity``, optional
            Distance limit used for searching counterparts in the secondary
            catalogue in angular units. Default to 5 arcsec.
        magmin : `float` or 'auto', optional
            Lower magnitude limit when estimating magnitude distributions.
            Default to 'auto'.
        magmax : `float` or 'auto', optional
            Upper magnitude limit when estimating magnitude distributions.
            Default to 'auto'.
        magbinsize : `float` or 'auto', optional
            Magnitude bin width when estimating magnitude distributions.
            Default to 'auto'.
        """
        if prior_dict is None:
            self._from_catalogues(pcat, scat, match_mags, rndcat,
                                  radius, magmin, magmax, magbinsize)
        else:
            self.prior_dict = prior_dict
            self.rndcat = None

    def _from_catalogues(self, pcat, scat, match_mags, rndcat,
                              radius, magmin, magmax, magbinsize):
        if None in [pcat, scat]:
            raise ValueError('Two Catalogues must be passed!')

        if rndcat is True:
            self.rndcat = pcat.randomise()

        elif isinstance(rndcat, Catalogue):
            self.rndcat = rndcat

        else:
            message = 'Using mask method for the prior calculation.'
            warnings.warn(message, AstropyUserWarning)
            self.rndcat = None

        if match_mags is None:
            match_mags = self._get_match_mags(pcat, scat, radius)

        self.prior_dict = self._calc_prior_dict(
            pcat, scat, radius, match_mags, magmin, magmax, magbinsize
        )

    @property
    def magnames(self):
        return list(self.prior_dict.keys())

    @classmethod
    def from_nway_hists(cls, cat, renorm_factors, path='.'):
        """
        Create a ``Prior`` object using nway histogram files.
        """
        prior_dict = {}
        for mag in cat.mags.colnames:
            filename = '{}_{}_fit.txt'.format(cat.name, mag)
            filename = os.path.join(path, filename)
            prior_dict[mag] = cls._from_nway_maghist(filename, renorm_factors[mag])

        return cls(prior_dict=prior_dict)

    @classmethod
    def from_table(cls, priors_table, magnames):
        """
        Create a ``Prior`` object using an Astropy Table.

        Parameters
        ----------
        priors_table : `str` or ``Table``
            Astropy table with the prior values or, alternatively,
            a file path containing a table in a format readable by Astropy.
            Note: If the table does not include priors for "field" sources,
            they are set to zero.
        magnames : `list`
            Magnitude names.
        """
        if not isinstance(priors_table, Table):
            priors_table = Table.read(priors_table)

        # TODO: how to do this when magbinsize is 'auto'
        bins = cls._midvals_to_bins(priors_table['MAG'])

        prior_dict = {}
        for mag in magnames:
            maghist = {}
            maghist['bins'] = bins
            maghist['target'] = priors_table['PRIOR_' + mag].data

            try:
                maghist['field'] = priors_table['PRIOR_BKG_' + mag].data
            except KeyError:
                maghist['field'] = np.zeros_like(bins)

                message = 'Field prior for {} set to zero.'.format(mag)
                warnings.warn(message, AstropyUserWarning)

            prior_dict[mag] = maghist

        return cls(prior_dict=prior_dict)

    def to_nway_hists(self, output_path=None):
        """
        Returns a dictionary with the prior histograms in
        a format compatible with nway. If `output_path` is not ``None``,
        a text file is created with a formatting compatible with nway.
        """
        nhists = []
        for magcol in self.magnames:
            if output_path is not None:
                filename = '{}_{}_fit.txt'.format(self.scat.name, magcol)
                filename = os.path.join(output_path, filename)
            else:
                filename = None

            maghist = self._to_nway_maghist(self.prior_dict[magcol], filename)
            nhists.append(maghist)

        return nhists

    def interp(self, mags, magcol):
        """
        Return the prior at magnitude values `mags` for magnitude `magcol`.

        Parameters
        ----------
        """
        if magcol not in self.prior_dict:
            raise ValueError('Unknown magcol: {}'.format(magcol))

        bins = self.bins_midvals(magcol)
        prior = self.prior_dict[magcol]

        itp = interp1d(
            bins, prior['target'], kind='nearest', fill_value=0, bounds_error=False
        )
        pvals = itp(mags)

        return pvals

    def qcap(self, magcol):
        """
        Overall identification ratio for magnitude `magcol`
        between the two catalogues used to build the prior.
        """
        if magcol not in self.prior_dict:
            raise ValueError('Unknown magcol: {}'.format(magcol))

        prior = self.prior_dict[magcol]

        # Whatch out prior is dN/dm,
        # i.e. I have divided by dm so it is probability density and
        # Sum(dN/dm*dm)=Q ie the overall identification ratio (not 1)
        return np.sum(prior['target'] * np.diff(prior['bins']))

    def bins_midvals(self, magcol):
        if magcol not in self.prior_dict:
            raise ValueError('Unknown magcol: {}'.format(magcol))

        edges = self.prior_dict[magcol]['bins']

        return (edges[1:] + edges[:-1])/2

    def to_table(self, include_bkg_priors=False):
        """
        Dump prior data into an Astropy Table.
        """
        # TODO: how to build this table when magbinsize is 'auto'
        priors_table = Table()
        priors_table['MAG'] = self.bins_midvals(self.magnames[0])
        for mag in self.magnames:
            priors_table['PRIOR_' + mag] = self.prior_dict[mag]['target']

            if include_bkg_priors:
                priors_table['PRIOR_BKG_' + mag] = self.prior_dict[mag]['field']

        return priors_table

    def plot(self, magname, filename=None):
        """
        Plot priors for magnitude `magname`.

        Parameters
        """
        import matplotlib.pyplot as plt

        mbins = self.bins_midvals(magname)
        prior = self.prior_dict[magname]

        plt.plot(mbins, prior['target'])
        plt.plot(mbins, prior['field'])
        plt.title(magname)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()


    def _get_match_mags(self, pcat, scat, radius):
        _, idx_near, _, _ = scat.coords.search_around_sky(pcat.coords, radius)

        return scat.mags[idx_near]

    def _calc_prior_dict(
        self,
        pcat,
        scat,
        radius,
        match_mags,
        magmin,
        magmax,
        magbinsize,
        mask_radius=1*u.arcmin
    ):
        if self.rndcat is None:
            field_cat = self._field_sources(pcat, scat, mask_radius)
        else:
            field_cat = self._random_sources(scat, radius)

        renorm_factor = len(pcat) * np.pi*radius**2 / field_cat.area  # area_match / area field

        prior_dict = {}
        for magcol in match_mags.colnames:
            target_mags = match_mags[magcol]
            field_mags = field_cat.mags[magcol]
            prior_dict[magcol] = self._mag_hist(
                len(pcat), target_mags, field_mags, renorm_factor, magmin, magmax, magbinsize
            )
        return prior_dict

    def _field_sources(self, pcat, scat, mask_radius):
        # Find sources within the mask_radius
        pcoords = pcat.coords
        scoords = scat.coords
        _, idx_near, _, _ = scoords.search_around_sky(pcoords, mask_radius)

        # Select all sources but those within the mask_radius
        idx_all = range(len(scat))
        idx_far = list(set(idx_all) - set(idx_near))
        field_cat = scat[idx_far]

        # Area covered by the new catalogue
        field_cat.area = scat.area - len(pcat)*np.pi*mask_radius**2
        field_cat.moc = None

        return field_cat

    def _random_sources(self, scat, radius):
        assert self.rndcat is not None

        # Select sources from the secondary catalogue within radius of random sources
        pcoords = self.rndcat.coords
        scoords = scat.coords
        _, sidx, _, _ = scoords.search_around_sky(pcoords, radius)
        rnd_scat = scat[sidx]

        # Area covered by the new catalogue
        rnd_scat.area = len(self.rndcat)*np.pi*radius**2
        rnd_scat.moc = None

        return rnd_scat

    def _mag_hist(
        self,
        pcat_nsources,
        target_mags,
        field_mags,
        renorm_factor,
        magmin,
        magmax,
        magbinsize
    ):
        if magmin == 'auto':
            magmin = np.nanmin(target_mags)

        if magmax == 'auto':
            magmax = np.nanmax(target_mags)

        bins, magrange = _define_magbins(magmin, magmax, magbinsize)
        target_counts, bins = np.histogram(target_mags, range=magrange, bins=bins)
        field_counts, _ = np.histogram(field_mags, range=magrange, bins=bins)
        magbinsize = np.diff(bins)

        target_prior = target_counts - field_counts * renorm_factor
        target_prior[target_prior < 0] = 0.0
        # TODO: calculate general values for the convolution parameters
        # (magbinsize dependent)
        target_prior = convolve(target_prior, [0.25, 0.5, 0.25])
        # target_prior = convolve(target_prior, [magbinsize[0]/2., magbinsize[0], magbinsize[0]/2.])

        # renormalise here to 0.999 in case
        # prior sums to a value above unit
        # Not unit because then zeros in Reliability
        # estimation, i.e. (1-QCAP) term
#        test = target_prior.sum() / len(self.pcat)
#        if test > 1:
#            target_prior = 0.999 * target_prior / test

        maghist = {
            'bins': bins,
            'target': target_prior / pcat_nsources / magbinsize,
            'field': 1.0*field_counts / len(field_mags) / magbinsize,
        }

        return maghist

    @staticmethod
    def _to_nway_maghist(maghist, filename=None):
        nrows = maghist['target'].size

        hist_data = np.zeros((nrows, 4))
        hist_data[:, 0] = maghist['bins'][:-1]
        hist_data[:, 1] = maghist['bins'][1:]
        hist_data[:, 2] = (
            maghist['target'] / np.sum(maghist['target'] * np.diff(maghist['bins']))
        )
        hist_data[:, 3] = maghist['field']

        if filename is not None:
            header = '{}\nlo hi selected others'.format(filename)
            np.savetxt(filename, hist_data, fmt='%10.5f', header=header)

        return [row for row in hist_data.T]

    @staticmethod
    def _from_nway_maghist(filename, renorm_factor):
        hist_data = Table.read(filename, format='ascii')

        maghist = {
            'bins': np.concatenate((hist_data['lo'], [hist_data['hi'][-1]])),
            'target': renorm_factor * hist_data['selected'].data,
            'field': hist_data['others'].data,
        }

        return maghist

    @staticmethod
    def _midvals_to_bins(midvals):
        dbins = np.diff(midvals) / 2
        bins_lo = set(midvals[:-1] - dbins)
        bins_hi = set(midvals[1:] + dbins)
        bins = np.array(list(bins_lo.union(bins_hi)))
        bins.sort()

        return bins


class BKGpdf(object):

    def __init__(self, cat, magmin='auto', magmax='auto', magbinsize='auto'):
        """
        Magnitude probability distribution of sources in ``Catalogue`` 'cat'.

        Parameters
        ----------
        cat : ``Catalogue``
            ``Catalogue`` object.
        magmin : `float` or 'auto', optional
            Lower magnitude limit when estimating magnitude distributions.
            Default to 'auto'.
        magmax : `float` or 'auto', optional
            Upper magnitude limit when estimating magnitude distributions.
            Default to 'auto'.
        magbinsize : `float` or 'auto', optional
            Magnitude bin width when estimating magnitude distributions.
            Default to 'auto'.

        Return
        ------
        bkg : Astropy ``Table``
            Table with the background probability distribution for each
            available magnitude in the secondary catalogue.
        """
        if cat.mags is None:
            raise ValueError('No magnitudes defined in the catalogue!')

        #self.magnames = self._set_magnames(cat)
        self.pdf_dict = self._calc_pdf(cat, magmin, magmax, magbinsize)

    @property
    def magnames(self):
        return list(self.pdf_dict.keys())

    def bins_midvals(self, magcol):
        edges = self.pdf_dict[magcol]['bins']

        return (edges[1:] + edges[:-1])/2

    def interp(self, mags, magcol):
        assert magcol in self.pdf_dict

        bins = self.bins_midvals(magcol)
        pdf = self.pdf_dict[magcol]['pdf']
        itp = interp1d(
            bins, pdf, kind='nearest', fill_value=np.inf, bounds_error=False
        )
        # We use inf as fill_value because these results are mostly used
        # as divisor (e.g. LR method), this way we avoid dividing by zero.

        return itp(mags)

    def to_table(self):
        # TODO: how to build this table when magbinsize is 'auto'
        pdf_table = Table()
        pdf_table['MAG'] = self.bins_midvals(self.magnames[0])
        for mag in self.magnames:
            pdf_table['BKG_' + mag] = self.pdf_dict[mag]['pdf']

        return pdf_table

    def _set_magnames(self, cat):
        return cat.mags.colnames

    def _calc_pdf(self, cat, magmin, magmax, magbinsize):
        mags = cat.mags
        area = cat.area.to(u.arcsec**2)

        pdf_dict = {}
        for magcol in mags.colnames:
            pdf_dict[magcol] = self._mag_hist(
                mags[magcol], area, magmin, magmax, magbinsize
            )

        return pdf_dict

    def _mag_hist(self, mags, area, magmin, magmax, magbinsize):

        if magmin == 'auto':
            magmin = np.nanmin(mags)

        if magmax == 'auto':
            magmax = np.nanmax(mags)

        bins, magrange = _define_magbins(magmin, magmax, magbinsize)
        counts, bins = np.histogram(mags, range=magrange, bins=bins)
        magbinsize = np.diff(bins)

        maghist = {}
        maghist['bins'] = bins
        maghist['pdf'] = counts / magbinsize / area  ## in arcsec**2!!!

        return maghist


def _define_magbins(magmin, magmax, magbinsize):
    if magbinsize == 'auto':
        bins = 'auto'
    else:
        nbins = 1 + (magmax - magmin)/magbinsize
        bins = np.linspace(magmin, magmax, num=nbins)

    limits = (magmin, magmax)

    return bins, limits