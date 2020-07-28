# -*- coding: utf-8 -*-
"""
astromatch module for calculation of magnitude priors.

@author: A.Ruiz
"""
import os
import warnings
from itertools import count, combinations

import numpy as np
from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter

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
        radius=5 * u.arcsec,
        mags=None,
        magmin=None,
        magmax=None,
        magbinsize=None,
        match_mags=None,
        prior_dict=None,
    ):
        """
        N-dimensional prior probability distribution for a source in the
        primary catalogue `pcat` having a counterpart in the secondary
        catalogue `scat` with observed parameters in a user defined
        N-dimensional parameter space, e.g. magnitude, colour, optical
        morphology, etc.

        The calculation of the prior probability can use the input data.
        All possible counterparts from the secondary catalogue (scat)
        within a given search radius off the  pcat (primary catalogue)
        source positions are extracted. These are referred to as the
        "good" catalogue. A "background" of "field" expectation is
        estimated by repeating this excersise at random positions
        within the FOV (defined by the input MOCs) of the
        observation. The difference between "good" and "field"
        provides an estimate of the properties of the true counteparts
        of the pcat sources within the scat catalogue. In practice the
        user defines properties of interest via the "mags"
        parameter. These can be magnitudes in a given filter, colours
        between two bands, optical morphology (point like vs
        extended) or any other parameter listed in the secondary
        catalogue. It is emphasised that properties of interest are
        not created (e.g. the code does not produce colours between
        filters) but should exist as columns in the secondary
        catalogue table. Combination of properties are also possible to
        define multi-dimensional spaces. The "mags" attribute should be
        a python list. Properties of combination of properties of interest
        are also python lists within the "mags" list:

        mags = [[R, GR], Z]

        The above notation defines two priors, the first is a
        2-dimensional one that includes the scat columns "R" (for
        example R-band magnitude) and "GR" (could be G-R colour
        between the G and R optical bands). The second prior is a
        1-Dimensional prior of the "Z" column (could be Z-band
        magnitude). This notation could be extend to more than
        2-dimensions (e.g. [[R, GR, RZ], Z]).

        The code then uses the columns listed in the mags to build N-D
        histograms within user defined bins and ranges. This requires
        defining the range of a given parameter and the size of the
        bin within which the histograms will be calculated. These are
        defined via the parameters "magmin", "magmax",
        "magbinsize". These are expected to be python lists that
        contain for each input scat column (or combination of scat
        columns) in "mags" the corresponding bin size as well as the
        minimum and maximum of the range within which the data will be
        binned. Values of the column outside this range will be
        ignored. The expected form of these parameters that are
        consistent with the above "mags" example:

         magmin =     [[10, -3], 8]
         magmax =     [[27, 3], 24]
         magbinsize = [[0.5, 0.1], 0.25]

        The code estimates N-dimensional histograms for both the
        "target" and "background" or "field" populations. It then
        substracts the two to determine the N-dimensional distribution
        of the true counterparts. Negative histograms bins are set to
        zero. Smoothing is also applied to resulting difference
        distribution. The current hard-coded smoothing uses an
        N-dimensional Guassian distribution with sigma equal to
        one bin-size.

        The result of this process is a prior probability distribution
        density function for the counterparts of the sources in the
        primary catalogue. There will be one prior distribution for
        each entry in "mags". It is emphasized that the priors are
        probability distribution density functions, i.e. the values
        are divided by the bin size (or bin volume in the case of
        n-dimensional histograms and their integral sums to unity.

        The class includes methods to save the resulting priors into a
        fits file.

        Priors can also be provided externally via fits file that
        follow the format conventions of the code. In this case the
        calculation described above is skipped and the prior
        probability density functions, "mags", "magmin", "magmax",
        "magbinsize" are determined from the input file.

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
        mags : python list that includes lists of strings, optional
            Columns or combinations of columns in the scat for which
            histograms will be build. Defaults to None. If None then
            all the columns listed in the scat.mags Catalogue
            extension are used o define independent 1-Dimensional priors
        magmin : python list that includes lists of `floats` or 'auto', optional
            Lower boundary of the histograms. Defaults to None. If
            None or "auto" then the minimum is determined from the data.
        magmax : python list that includes lists of `floats` or 'auto', optional
            Upper boundary of the histograms. Defaults to None.  If
            None or "auto" then the maximum is determined from the data.
        magbinsize : python list that includes lists of `floats` or 'auto', optional
            Bin width of the histograms. Defaults to None. If
            None or "auto" then it defauls to 0.5.
        """
        if prior_dict is None:
            message = "Estimating priors using Catalogues: {} and {}..."
            log.info(message.format(pcat.name, scat.name))
            self._from_catalogues(
                pcat, scat, match_mags, rndcat, radius, mags, magmin, magmax, magbinsize
            )
        else:
            log.info("Using provided prior...")
            self.prior_dict = prior_dict
            self.rndcat = None

    def _from_catalogues(
        self, pcat, scat, match_mags, rndcat, radius, mags, magmin, magmax, magbinsize
    ):
        if None in [pcat, scat]:
            raise ValueError("Two Catalogues must be passed!")

        if rndcat is True:
            self.rndcat = pcat.randomise()

        elif isinstance(rndcat, Catalogue):
            self.rndcat = rndcat

        else:
            warnings.warn(
                "Using mask method for the prior calculation.", AstropyUserWarning
            )
            self.rndcat = None

        if match_mags is None:
            match_mags = self._get_match_mags(pcat, scat, radius)

        self.prior_dict = self._calc_prior_dict(
            pcat, scat, radius, match_mags, mags, magmin, magmax, magbinsize
        )

    @property
    def magnames(self):
        return list(self.prior_dict.keys())

    @classmethod
    def from_nway_hists(cls, cat, renorm_factors, path="."):
        """
        Create a ``Prior`` object using nway histogram files.
        """
        # TODO: modify this to work with new ndpriors
        prior_dict = {}
        for mag in cat.mags.colnames:
            filename = "{}_{}_fit.txt".format(cat.name, mag)
            filename = os.path.join(path, filename)
            prior_dict[mag] = cls._from_nway_maghist(filename, renorm_factors[mag], mag)

        return cls(prior_dict=prior_dict)

    @classmethod
    def from_table(cls, priors_table, magnames):
        """
        Create a ``Prior`` object using an Astropy Table.
        *Note:* Only 1D priors can be constructed from Astropy tables.

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
        # TODO: modify this to work with new ndpriors
        if not isinstance(priors_table, Table):
            priors_table = Table.read(priors_table)

        # TODO: how to do this when magbinsize is 'auto'
        edges = cls._midvals_to_bins(priors_table["MAG"])

        prior_dict = {}
        for mag in magnames:
            maghist = {}
            maghist["name"] = [priors_table.meta[mag]]
            maghist["edges"] = [edges]
            maghist["vol"] = edges[1] - edges[0]
            maghist["target"] = priors_table["PDF_" + mag].data

            try:
                maghist["field"] = priors_table["BKG_PDF_" + mag].data

            except KeyError:
                maghist["field"] = np.zeros_like(edges)
                warnings.warn(
                    "Field prior for {} set to zero.".format(mag), AstropyUserWarning
                )

            prior_dict[mag] = maghist

        return cls(prior_dict=prior_dict)

    @classmethod
    def from_fits(cls, filename=None, include_bkg_priors=False):
        """
        Create a ``Prior`` object from a fits file with the appropritate format.

        Parameters
        ----------
        filename : 'str'
            name of the file with the priors
        include_bkg_priors : `logical`
            read bkg counts from the file
        """
        prior_dict = {}

        hdul = fits.open(filename)

        # read the fits header looking for
        # extensions 'PRIOR\d"
        # the fits may also have "PRIOR\d_FIELD"
        # for the background counts
        enames, eprior = [], []
        for h in hdul:
            enames.append(h.name)
            if ("PRIOR" in h.name) and not ("FIELD" in h.name):
                eprior.append(h.name)

        if len(enames) == 0:
            raise ValueError(
                "File {} does not inlcude PRIOR extensions".format(filename)
            )

        mags, magmin, magmax, magbin = [], [], [], []
        for col in eprior:
            maghist = {}
            # maghist['name'] = col

            # INSTANCE IMAGE IE N-D PRIOR
            if isinstance(hdul[col], fits.ImageHDU):
                maghist["target"] = hdul[col].data
                (
                    maghist["edges"],
                    maghist["vol"],
                    maghist["name"],
                    pmin1,
                    pmax1,
                    pbin1,
                ) = cls._hdr2edges(hdul[col])

                if include_bkg_priors:
                    colfield = "FIELD_{}".format(col)
                    if colfield in enames:
                        maghist["field"] = hdul[colfield].data
                    else:
                        raise ValueError(
                            "No background counts for prior {} in file {}".format(
                                col, filename
                            )
                        )

            # INSTANCE TABLE IE 1-D PRIOR
            elif isinstance(hdul[col], fits.BinTableHDU):
                maghist["target"] = hdul[col].data[col]
                (
                    maghist["edges"],
                    maghist["vol"],
                    maghist["name"],
                    pmin1,
                    pmax1,
                    pbin1,
                ) = cls._hdr2edges(hdul[col])
                if include_bkg_priors:
                    colfield = "FIELD_{}".format(col)
                    if colfield in enames:
                        maghist["field"] = hdul[colfield].data[colfield]
                    else:
                        raise ValueError(
                            "No background counts for prior {} in file {}".format(
                                col, filename
                            )
                        )

            mags.append(maghist["name"])
            magmin.append(pmin1)
            magmax.append(pmax1)
            magbin.append(pbin1)
            prior_dict[col.upper()] = maghist

        hdul.close()

        return cls(
            prior_dict=prior_dict,
            magmin=magmin,
            magmax=magmax,
            magbinsize=magbin,
            mags=mags,
        )

    def to_nway_hists(self, output_path=None):
        """
        Returns a dictionary with the prior histograms in
        a format compatible with nway. If `output_path` is not ``None``,
        a text file is created with a formatting compatible with nway.
        """
        # TODO: modify this to work with new ndpriors
        nhists = []
        for magcol in self.magnames:
            if output_path is not None:
                filename = "{}_{}_fit.txt".format(self.scat.name, magcol)
                filename = os.path.join(output_path, filename)
            else:
                filename = None

            maghist = self._to_nway_maghist(self.prior_dict[magcol], filename)
            nhists.append(maghist)

        return nhists

    def to_table(self, include_bkg_priors=False):
        """
        Dump prior data into an Astropy Table.

        Note: only 1D priors can be stored as tables.
        """
        # TODO: how to build this table when magbinsize is 'auto'
        priors_table = Table()

        for mag in self.magnames:
            if self.prior_dict[mag]["target"].ndim > 1:
                warnings.warn(
                    "{} is not a 1D prior; use to_fits method.".format(mag),
                    AstropyUserWarning,
                )
                continue

            if "MAG" not in priors_table.colnames:
                priors_table["MAG"] = self.bins_midvals(mag)[0]

            if any(np.not_equal(self.bins_midvals(mag)[0], priors_table["MAG"])):
                warnings.warn(
                    "Inconsistent binning for {};  use to_fits method.".format(mag),
                    AstropyUserWarning,
                )
                continue

            try:
                priors_table["PDF_" + mag] = self.prior_dict[mag]["target"]

                if include_bkg_priors:
                    priors_table["BKG_PDF_" + mag] = self.prior_dict[mag]["field"]

                priors_table.meta[mag] = self.prior_dict[mag]["name"][0]

            except ValueError as e:
                warnings.warn("{}; use to_fits method.".format(e), AstropyUserWarning)
                continue

        return priors_table

    def to_fits(self, include_bkg_priors=False, filename=None):
        """
        Save prior data in a fits file. Each property prior is saved in a
        different HDU extension: 1D data is saved as a table and 2D or higher
        as a multidimensional image.
        """
        hdus = [fits.PrimaryHDU()]
        for col in self.magnames:
            hdrlist = self._getHDR(col, include_bkg_priors)
            hdus += hdrlist

        hdu = fits.HDUList(hdus)

        if filename:
            hdu.writeto(filename, overwrite=True)

        return hdu

    def interp(self, mags, col):
        """
        Return the prior at magnitude numpyvalues `mags` for magnitude `magcol`.

        This is not interpolation as indicates the name but rather finds
        the nearest neighbour in the grid of values.

        If value is outside the prior grid returns a zero.
        So values eg -99 or +99 will be assigned a prior 0

        Parameters
        ----------
        """
        if col not in self.prior_dict:
            raise ValueError("Unknown col: {}".format(col))

        prior = self.prior_dict[col]

        q = []
        for i, magcol in enumerate(prior["name"]):
            if i == 0:
                flags = np.ones(len(mags[magcol]), dtype=bool)

            edges = prior["edges"]
            indeces = (mags[magcol] - edges[i][0]) / (edges[i][1] - edges[i][0])
            indeces = indeces.astype(int)
            m = indeces < 0
            indeces[m] = 0
            flags[m] = False
            m = indeces > len(edges[i]) - 2
            indeces[m] = len(edges[i]) - 2
            flags[m] = False
            q.append(indeces)

        pvals = prior["target"][tuple(q)]
        pvals[np.logical_not(flags)] = 0.0

        return pvals

    def qcap(self, magcol):
        """
        Overall identification ratio for magnitude `magcol`
        between the two catalogues used to build the prior.
        """
        if magcol not in self.prior_dict:
            raise ValueError("Unknown magcol: {}".format(magcol))

        prior = self.prior_dict[magcol]

        # Whatch out prior is dN/dm,
        # i.e. I have divided by dm so it is probability density and
        # Sum(dN/dm*dm) = Q ie the overall identification ratio (not 1)
        return np.sum(prior["target"]) * prior["vol"]

    def bins_midvals(self, magcol):
        if magcol not in self.prior_dict:
            raise ValueError("Unknown prior name: {}".format(magcol))

        edges = self.prior_dict[magcol]["edges"]
        midvals = [(e[1:] + e[:-1]) / 2 for e in edges]

        return midvals

    def plot(self, priorname, filename=None, contour_levels=3):
        """
        Plot 1D/2D prior PDFs.
        """
        if priorname not in self.magnames:
            raise ValueError("Unknown prior name: {}".format(priorname))

        prior = self.prior_dict[priorname]

        if prior["target"].ndim > 2:
            self._plot_prior_nd(priorname, prior, contour_levels)

        elif prior["target"].ndim > 1:
            self._plot_prior_2d(prior, contour_levels)
            plt.title(priorname)

        else:
            self._plot_prior_1d(self.bins_midvals(priorname), prior)
            plt.title(priorname)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

        plt.close()

    @staticmethod
    def _plot_prior_1d(mbins, prior):
        plt.plot(mbins[0], prior["target"], label="target")

        if "field" in prior:
            plt.plot(mbins[0], prior["field"], label="field")

        plt.xlabel(prior["name"][0])
        plt.legend()

    @staticmethod
    def _plot_prior_2d(prior, levels):
        plt.imshow(prior["target"], aspect="equal", origin="lower")

        if "field" in prior:
            plt.contour(prior["field"], levels=levels, colors="white", alpha=0.5)

        # TODO: check that axes are named in the correct order
        # TODO: Tick labels are pixel numbers, not the corresponding magnitudes
        plt.xlabel(prior["name"][0])
        plt.ylabel(prior["name"][1])

    @staticmethod
    def _plot_prior_nd(priorname, prior, levels):
        D = prior["target"].ndim
        idx = list(range(D))

        fig = plt.figure(constrained_layout=True)
        spec = gridspec.GridSpec(ncols=D - 1, nrows=D - 1, figure=fig)

        for c in combinations(idx, 2):
            idx_reduce = set(idx) - set(c)
            reduced_prior = np.add.reduce(prior["target"], axis=tuple(idx_reduce))

            ax = fig.add_subplot(spec[c[1] - 1, c[0]])
            ax.imshow(reduced_prior, aspect="equal", origin="lower")

            if "field" in prior:
                reduced_field = np.add.reduce(prior["field"], axis=tuple(idx_reduce))
                plt.contour(reduced_field, levels=levels, colors="white", alpha=0.5)

            # TODO: check that axes are named in the correct order
            # TODO: Tick labels are pixel numbers, not the corresponding magnitudes
            ax.set_xlabel(prior["name"][c[0]])
            ax.set_ylabel(prior["name"][c[1]])

        fig.suptitle(priorname)

    @staticmethod
    def _hdr2edges(hdu):
        if isinstance(hdu, fits.ImageHDU):
            N = len(hdu.shape)
            SIZE = hdu.shape
        else:
            N = 1
            SIZE = [hdu.header["NAXIS2"]]

        vol = 1.0
        name, edges = [], []
        pmin, pmax, pbin = [], [], []
        for i in range(N):
            start = hdu.header["CRVAL{}L".format(i + 1)]
            binn = hdu.header["CDELT{}L".format(i + 1)]
            end = start + SIZE[i] * binn
            vol = vol * binn
            name.append(hdu.header["CTYPE{}L".format(i + 1)])
            edges.append(np.arange(start - binn / 2, end, binn))
            pmin.append(start - binn / 2)
            pmax.append(end - binn / 2)
            pbin.append(binn)

        return edges, vol, name, pmin, pmax, pbin

    def _hdr(self, priorname):
        mbins = self.bins_midvals(priorname)
        hdr = fits.Header()

        for i, mb in enumerate(mbins):
            hdr.set(
                "CTYPE{}L".format(i + 1),
                self.prior_dict[priorname]["name"][i],
                "WCS coordinate name",
            )
            hdr.set("CRPIX{}L".format(i + 1), 1, "WCS reference pixel")
            hdr.set("CRVAL{}L".format(i + 1), mb[0], "WCS reference pixel value")
            hdr.set("CDELT{}L".format(i + 1), mb[1] - mb[0], "WCS pixel size")

        hdr.set("WCSNAMEL", "PHYSICAL", "WCS L name")
        hdr.set("WCSAXESL", len(mbins), "No. of axes for WCS L")

        return hdr

    def _getHDR(self, priorname, include_bkg_priors):
        if priorname not in self.magnames:
            raise ValueError("Unknown Prior Name: {}".format(priorname))

        hdus = []
        if self.prior_dict[priorname]["target"].ndim > 1:
            hdr = self._hdr(priorname)

            hdus.append(
                fits.ImageHDU(
                    self.prior_dict[priorname]["target"], header=hdr, name=priorname,
                )
            )
            if include_bkg_priors:
                hdus.append(
                    fits.ImageHDU(
                        self.prior_dict[priorname]["field"],
                        header=hdr,
                        name="FIELD_{}".format(priorname),
                    )
                )
        else:
            hdr = self._hdr(priorname)
            mbins = self.bins_midvals(priorname)

            c1 = fits.Column(
                name=priorname, array=self.prior_dict[priorname]["target"], format="D"
            )
            c2 = fits.Column(name="MAG", array=mbins[0], format="D")

            hdus.append(
                fits.BinTableHDU.from_columns([c1, c2], header=hdr, name=priorname)
            )

            if include_bkg_priors:
                c3 = fits.Column(
                    name="FIELD_{}".format(priorname),
                    array=self.prior_dict[priorname]["field"],
                    format="D",
                )
                c4 = fits.Column(name="MAG", array=mbins[0], format="D")

                hdus.append(
                    fits.BinTableHDU.from_columns(
                        [c3, c4], header=hdr, name="FIELD_{}".format(priorname)
                    )
                )

        return hdus

    def _get_match_mags(self, pcat, scat, radius):
        _, idx_near, _, _ = scat.coords.search_around_sky(pcat.coords, radius)

        return scat.mags[idx_near]

    def _calc_prior_dict(
        self,
        pcat,
        scat,
        radius,
        match_mags,
        mags,
        magmin,
        magmax,
        magbinsize,
        mask_radius=1 * u.arcmin,
    ):
        if self.rndcat is None:
            field_cat = self._field_sources(pcat, scat, mask_radius)
        else:
            field_cat = self._random_sources(scat, radius)

        # area_match / area field
        renorm_factor = len(pcat) * np.pi * radius ** 2 / field_cat.area

        if mags is None:
            mags = list(match_mags.colnames)

        log.info("Using columns: {}".format(mags))

        magmin, magmax, magbinsize = self._parse_mag_params(
            mags, magmin, magmax, magbinsize
        )

        prior_dict = {}

        for iprior, col, mmin, mmax, mbin in zip(
            count(), mags, magmin, magmax, magbinsize
        ):
            edges = []
            if isinstance(col, list):
                sample = np.ndarray([len(match_mags[col[0]]), len(col)])
                field_sample = np.ndarray([len(field_cat.mags[col[0]]), len(col)])

                for i, c, mn, mx, mb in zip(count(), col, mmin, mmax, mbin):
                    if mn == "auto":
                        mn = int(min(field_cat.mags[c]) - 0.5)

                    if mx == "auto":
                        mx = int(max(field_cat.mags[c]) + 0.5)

                    if mb == "auto":
                        mb = 0.5

                    sample[:, i] = match_mags[c]
                    field_sample[:, i] = field_cat.mags[c]
                    edges.append(np.arange(mn, mx + mb / 2.0, mb))

                prior_dict["PRIOR{}".format(iprior)] = self._mag_hist(
                    len(pcat), sample, field_sample, renorm_factor, edges, col
                )

            else:
                sample = match_mags[col]
                field_sample = field_cat.mags[col]

                if mmin == "auto":
                    mmin = int(min(field_cat.mags[col]) - 0.5)

                if mmax == "auto":
                    mmax = int(max(field_cat.mags[col]) + 0.5)

                if mbin == "auto":
                    mbin = 0.5

                edges.append(np.arange(mmin, mmax + mbin / 2.0, mbin))
                prior_dict["PRIOR{}".format(iprior)] = self._mag_hist(
                    len(pcat), sample, field_sample, renorm_factor, edges, [col]
                )

        return prior_dict

    def _parse_mag_params(self, mags, magmin, magmax, magbinsize):
        if magmin is None:
            magmin = ["auto"] * len(mags)

        if isinstance(magmin, float):
            magmin = [magmin] * len(mags)

        if magmax is None:
            magmax = ["auto"] * len(mags)

        if isinstance(magmax, float):
            magmax = [magmax] * len(mags)

        if magbinsize is None:
            magbinsize = ["auto"] * len(mags)

        if isinstance(magbinsize, float):
            magbinsize = [magbinsize] * len(mags)

        return magmin, magmax, magbinsize

    def _mag_hist(
        self, pcat_nsources, target_mags, field_mags, renorm_factor, edges, col,
    ):
        target_counts, bins = np.histogramdd(target_mags, edges)
        field_counts, _ = np.histogramdd(field_mags, edges)
        vol = 1.0
        for l in bins:
            vol = vol * (l[1:-1] - l[0:-2])[0]

        target_prior = target_counts - field_counts * renorm_factor
        target_prior[target_prior < 0] = 0.0
        # TODO: calculate general values for the convolution parameters
        # (magbinsize dependent)
        target_prior = gaussian_filter(target_prior, sigma=1.5, truncate=3)

        maghist = {}
        maghist["edges"] = edges
        maghist["vol"] = vol
        maghist["target"] = target_prior / pcat_nsources / vol
        maghist["field"] = 1.0 * field_counts / len(field_mags) / vol
        maghist["name"] = col

        return maghist

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
        field_cat.area = scat.area - len(pcat) * np.pi * mask_radius ** 2
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
        rnd_scat.area = len(self.rndcat) * np.pi * radius ** 2
        rnd_scat.moc = None

        return rnd_scat

    @staticmethod
    def _to_nway_maghist(maghist, filename=None):
        if maghist["target"].ndim > 1:
            raise ValueError("Only 1D priors can be used in NWAY.")

        nrows = maghist["target"].size

        hist_data = np.zeros((nrows, 4))
        hist_data[:, 0] = maghist["edges"][0][:-1]
        hist_data[:, 1] = maghist["edges"][0][1:]
        hist_data[:, 2] = maghist["target"] / np.sum(
            maghist["target"] * np.diff(maghist["edges"][0])
        )
        hist_data[:, 3] = maghist["field"]

        if filename is not None:
            header = "{}\nlo hi selected others".format(filename)
            np.savetxt(filename, hist_data, fmt="%10.5f", header=header)

        return [row for row in hist_data.T]

    @staticmethod
    def _from_nway_maghist(filename, renorm_factor, mag):
        hist_data = Table.read(filename, format="ascii")

        maghist = {
            "edges": np.concatenate((hist_data["lo"], [hist_data["hi"][-1]])),
            "vol": hist_data["lo"][1] - hist_data["lo"][0],
            "target": renorm_factor * hist_data["selected"].data,
            "field": hist_data["others"].data,
            "name": [mag],
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
    def __init__(
        self,
        cat=None,
        mags=None,
        magmin=None,
        magmax=None,
        magbinsize=None,
        pdf_dict=None,
    ):
        """
        Magnitude probability distribution of sources in ``Catalogue`` 'cat'.
        This class store/estimate the N-dimensional binned number density
        distribution of sources.

        Parameters
        ----------
        cat : ``Catalogue``
            ``Catalogue`` object.
        mags : python list that includes lists of strings, optional
            Columns or combinations of columns in the scat for which
            histograms will be build. Defaults to None. If None then
            all the columns listed in the scat.mags Catalogue
            extension are used o define independent 1-Dimensional priors
        magmin : python list that includes lists of `floats` or 'auto', optional
            Lower boundary of the histograms. Defaults to None. If
            None or "auto" then the minimum is determined from the data.
        magmax : python list that includes lists of `floats` or 'auto', optional
            Upper boundary of the histograms. Defaults to None.  If
            None or "auto" then the maximum is determined from the data.
        magbinsize :  python list that includes lists of `floats` or 'auto', optional
            Bin width of the histograms. Defaults to None.  If
            None or "auto" then it defauls to 0.5.

        Return
        ------
        bkg : Astropy ``Table``
            Table with the background probability distribution for each
            available magnitude in the secondary catalogue.
        """

        # self.magnames = self._set_magnames(cat)
        if pdf_dict is None:
            if cat is None:
                raise ValueError("No magnitudes defined in the catalogue!")
            elif cat.mags is None:
                raise ValueError("No magnitudes defined in the catalogue!")
            self.magmin = magmin
            self.magmax = magmax
            self.magbinsize = magbinsize
            self.mags = mags
            self.pdf_dict = self._calc_pdf(cat, mags, magmin, magmax, magbinsize)
        else:
            self.pdf_dict = pdf_dict
            self.magmin = magmin
            self.magmax = magmax
            self.magbinsize = magbinsize
            self.mags = mags
        # for n in self.pdf_dict.keys():
        #    self.plot(n)

    @property
    def magnames(self):
        return list(self.pdf_dict.keys())

    def bins_midvals(self, magcol):
        if magcol not in self.pdf_dict:
            raise ValueError("Unknown Prior Name: {}".format(magcol))

        edges = self.pdf_dict[magcol]["edges"]
        midvals = []
        for e in edges:
            midvals.append((e[1:] + e[:-1]) / 2)
        return midvals

    @staticmethod
    def _hdr2edges(hdu):

        if isinstance(hdu, fits.ImageHDU):
            N = len(hdu.shape)
            SIZE = hdu.shape
        else:
            N = 1
            SIZE = [hdu.header["NAXIS2"]]

        name = []
        edges = []
        vol = 1.0
        pmin = []
        pmax = []
        pbin = []
        for i in range(N):
            start = hdu.header["CRVAL{}L".format(i + 1)]
            binn = hdu.header["CDELT{}L".format(i + 1)]
            end = start + SIZE[i] * binn
            vol = vol * binn
            name.append(hdu.header["CTYPE{}L".format(i + 1)])
            edges.append(np.arange(start - binn / 2, end, binn))
            pmin.append(start - binn / 2)
            pmax.append(end - binn / 2)
            pbin.append(binn)
        return edges, vol, name, pmin, pmax, pbin

    def interp(self, mags, col):
        """
        Return the prior at magnitude values `mags` for magnitude `magcol`.

        This is not interpolation as indicates the name but rather finds
        the nearest neigboor in the grid of values.

        If value is outside the prior grid returns a zero

        So values eg -99 or +99 will be assigned a prior 0

        Function works with both 1-d and n-d priors

        Parameters
        ----------
        """

        if col not in self.pdf_dict:
            raise ValueError("Unknown col: {}".format(col))

        prior = self.pdf_dict[col]
        q = []
        for i, magcol in enumerate(prior["name"]):

            if i == 0:
                flags = np.ones(len(mags[magcol]), dtype=bool)

            edges = prior["edges"]
            indeces = (
                (mags[magcol] - edges[i][0]) / (edges[i][1] - edges[i][0])
            ).astype(int)
            m = indeces < 0
            indeces[m] = 0
            flags[m] = False
            m = indeces > len(edges[i]) - 2
            indeces[m] = len(edges[i]) - 2
            flags[m] = False
            q.append(indeces)
        q = tuple(q)
        pvals = prior["pdf"][q]
        pvals[np.logical_not(flags)] = 0.0
        return pvals

    @classmethod
    def from_table(cls, filename=None):

        """
        Create a ``BKG`` object from a fits file with the appropritate format.

        Parameters
        ----------
        filename : 'str'
                 name of the file with the priors
        include_bkg_priors : `logical`
            read bkg counts from the file
        """

        prior_dict = {}

        hdul = fits.open(filename)

        # read the fits header looking for
        # extensions 'PRIOR\d"
        enames = []
        eprior = []
        for h in hdul:
            enames.append(h.name)
            if "PRIOR" in h.name:
                eprior.append(h.name)
        if len(enames) == 0:
            raise ValueError(
                "File {} does not inlcude FIELD counts extensions".format(filename)
            )

        mags = []
        magmin = []
        magmax = []
        magbin = []
        for col in eprior:
            maghist = {}
            # maghist['name'] = col

            # INSTANCE IMAGE IE N-D PRIOR
            if isinstance(hdul[col], fits.ImageHDU):
                maghist["pdf"] = hdul[col].data * (1.0 / u.arcsec ** 2)
                (
                    maghist["edges"],
                    maghist["vol"],
                    maghist["name"],
                    pmin1,
                    pmax1,
                    pbin1,
                ) = cls._hdr2edges(hdul[col])

            # INSTANCE TABLE IE 1-D PRIOR
            elif isinstance(hdul[col], fits.BinTableHDU):
                maghist["pdf"] = hdul[col].data[col] * (1.0 / u.arcsec ** 2)
                (
                    maghist["edges"],
                    maghist["vol"],
                    maghist["name"],
                    pmin1,
                    pmax1,
                    pbin1,
                ) = cls._hdr2edges(hdul[col])

            mags.append(maghist["name"])
            magmin.append(pmin1)
            magmax.append(pmax1)
            magbin.append(pbin1)
            prior_dict[col] = maghist
        hdul.close()

        print(prior_dict)
        return cls(
            pdf_dict=prior_dict,
            magmin=magmin,
            magmax=magmax,
            magbinsize=magbin,
            mags=mags,
        )

    def to_table(self):
        """

        AGE: this is not possible anyore: Dump prior data into an Astropy Table.

        the output has to be fits because the
        arrays may have different dimensions
        """

        hdu = fits.HDUList()
        hdu.append(fits.PrimaryHDU())
        for col in self.pdf_dict.keys():

            hdrlist = self._getHDR(col)
            for h in hdrlist:
                hdu.append(h)
        return hdu

    def _hdr(self, priorname):

        mbins = self.bins_midvals(priorname.upper())
        hdr = fits.Header()

        for i in range(len(mbins)):
            hdr.set(
                "CTYPE{}L".format(i + 1),
                self.pdf_dict[priorname.upper()]["name"][i],
                "WCS coordinate name",
            )
            hdr.set("CRPIX{}L".format(i + 1), 1, "WCS reference pixel")
            hdr.set(
                "CRVAL{}L".format(i + 1), (mbins[i])[0], "WCS reference pixel value"
            )
            hdr.set(
                "CDELT{}L".format(i + 1),
                (mbins[i])[1] - (mbins[i])[0],
                "WCS pixel size",
            )
        hdr.set("WCSNAMEL", "PHYSICAL", "WCS L name")
        hdr.set("WCSAXESL", len(mbins), "No. of axes for WCS L")

        return hdr

    def _getHDR(self, priorname):

        if priorname.upper() not in self.pdf_dict.keys():
            raise ValueError("Unknown Prior Name: {}".format(priorname.upper()))

        if len(self.pdf_dict[priorname.upper()]["pdf"].shape) > 1:
            hdr = self._hdr(priorname.upper())
            hdu = fits.HDUList()
            hdu.append(
                fits.ImageHDU(
                    self.pdf_dict[priorname.upper()]["pdf"].value,
                    header=hdr,
                    name=priorname.upper(),
                )
            )
        else:
            hdr = self._hdr(priorname.upper())
            mbins = self.bins_midvals(priorname.upper())
            hdu = fits.HDUList()
            c1 = fits.Column(
                name=priorname.upper(),
                array=self.pdf_dict[priorname.upper()]["pdf"].value,
                format="D",
            )
            c2 = fits.Column(name="MAG", array=mbins[0], format="D")
            hdu.append(
                fits.BinTableHDU.from_columns(
                    [c1, c2], header=hdr, name=priorname.upper()
                )
            )
        return hdu

    def _set_magnames(self, cat):
        return cat.mags.colnames

    def plot(self, priorname, filename=None):

        import matplotlib.pyplot as plt

        if priorname.upper() not in self.pdf_dict.keys():
            raise ValueError("Unknown Prior Name: {}".format(priorname.upper()))

        print(priorname.upper(), self.pdf_dict[priorname.upper()]["pdf"].shape)

        if (
            len(self.pdf_dict[priorname.upper()]["pdf"].shape) > 1
            and len(self.pdf_dict[priorname.upper()]["pdf"].shape) <= 3
        ):
            hdr = self._hdr(priorname.upper())
            # print(self.pdf_dict[priorname.upper()]['pdf'].value)
            hdu = fits.HDUList()
            hdu.append(
                fits.ImageHDU(
                    (self.pdf_dict[priorname.upper()]["pdf"].value).T,
                    header=hdr,
                    name="BKG",
                )
            )
            # iiprint(self.pdf_dict[priorname.upper()]['pdf'])
            hdu.writeto("{}_bkg.fits".format(priorname.upper()), overwrite=True)

        if len(self.pdf_dict[priorname.upper()]["pdf"].shape) == 1:

            mbins = self.bins_midvals(priorname.upper())
            prior = self.pdf_dict[priorname.upper()]
            plt.plot(mbins[0], prior["pdf"])
            plt.title(priorname.upper())
            if filename is None:
                plt.show()
            else:
                plt.savefig("{}_bkg.png".format(priorname.upper()))
            plt.close()

    def _calc_pdf(self, cat, mags, magmin, magmax, magbinsize):

        if mags is None:
            mags = list(cat.mags.colnames)

        magmin, magmax, magbinsize = self._parse_mag_params(
            mags, magmin, magmax, magbinsize
        )

        field = cat.mags
        area = cat.area.to(u.arcsec ** 2)

        prior_dict = {}
        for iprior, col, mmin, mmax, mbin in zip(
            count(), mags, magmin, magmax, magbinsize
        ):
            edges = []
            if isinstance(col, list):
                sample = np.ndarray([len(field[col[0]]), len(col)])

                for i, c, mn, mx, mb in zip(count(), col, mmin, mmax, mbin):
                    if mn == "auto":
                        mn = int(min(field[c]) - 0.5)
                    if mx == "auto":
                        mx = int(max(field[c]) + 0.5)
                    if mb == "auto":
                        mb = 0.5

                    sample[:, i] = field[c]
                    edges.append(np.arange(mn, mx + mb / 2.0, mb))

                prior_dict["PRIOR{}".format(iprior)] = self._mag_hist(
                    sample, area, edges, col
                )
            else:
                if mmin == "auto":
                    mmin = int(min(field[col]) - 0.5)
                if mmax == "auto":
                    mmax = int(max(field[col]) + 0.5)
                if mbin == "auto":
                    mbin = 0.5

                edges.append(np.arange(mmin, mmax + mbin / 2.0, mbin))
                prior_dict["PRIOR{}".format(iprior)] = self._mag_hist(
                    field[col], area, edges, [col]
                )

        return prior_dict

    def _parse_mag_params(self, mags, magmin, magmax, magbinsize):
        if magmin is None:
            magmin = ["auto"] * len(mags)

        if isinstance(magmin, float):
            magmin = [magmin] * len(mags)

        if magmax is None:
            magmax = ["auto"] * len(mags)

        if isinstance(magmax, float):
            magmax = [magmax] * len(mags)

        if magbinsize is None:
            magbinsize = ["auto"] * len(mags)

        if isinstance(magbinsize, float):
            magbinsize = [magbinsize] * len(mags)

        return magmin, magmax, magbinsize

    def _mag_hist(self, mags, area, edges, col):
        counts, bins = np.histogramdd(mags, edges)
        vol = 1.0
        for l in bins:
            vol = vol * (l[1:-1] - l[0:-2])[0]

        maghist = {}
        maghist["edges"] = edges
        maghist["vol"] = vol
        maghist["pdf"] = counts / vol / area  # in arcsec**-2!!!
        #maghist["pdf"] = counts / vol / len(mags)
        maghist["name"] = col

        return maghist


def _define_magbins(magmin, magmax, magbinsize):
    if magbinsize == "auto":
        bins = "auto"
    else:
        nbins = 1 + (magmax - magmin) / magbinsize
        bins = np.linspace(magmin, magmax, num=nbins)

    limits = (magmin, magmax)

    return bins, limits
