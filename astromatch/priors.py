# -*- coding: utf-8 -*-
"""
astromatch module for calculation of magnitude priors.

@author: A.Ruiz
"""
import os
import warnings
from itertools import combinations, count, product

import numpy as np
from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from KDEpy import FFTKDE
from matplotlib import gridspec
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from .catalogues import Catalogue


class KDE(object):
    def __init__(self, data, bw, var_type=None):
        self.ndim = data.shape[1]
        self.kde, self.type = self._set_kde(data, bw, var_type)

    def _set_kde(self, data, bw, var_type):
        if isinstance(bw, float):
            kde = FFTKDE(bw=bw).fit(data)
            type = "kdepy"

        else:
            if self.ndim == 1:
                kde = FFTKDE(bw="ISJ").fit(data)
                type = "kdepy"
            else:
                if var_type is None:
                    var_type = "c" * self.ndim

                # bw options: "normal_reference", "cv_ml", "cv_ls"
                kde = KDEMultivariate(data, var_type=var_type, bw="cv_ls")
                type = "statsmodels"

        return kde, type

    def eval(self, grid):
        if self.type == "kdepy":
            values = self.kde.evaluate(grid)

        elif self.type == "statsmodels":
            values = self.kde.pdf(grid)

        else:
            raise ValueError("Unknown KDE type: {}".format(self.type))

        return values


class PriorBase(object):
    """
    Base class with common methods for Prior and BKGpdf classes.
    """

    @property
    def magnames(self):
        return list(self.prior_dict.keys())

    def interp(self, mags, magcol):
        """
        Return the prior at magnitude values `mags` for prior `magcol`.

        If value is outside the prior grid returns a zero.
        So values eg -99 or +99 will be assigned a prior 0

        Parameters
        ----------
        """
        if magcol not in self.prior_dict:
            raise ValueError("Unknown prior: {}".format(magcol))

        if isinstance(mags, Table):
            mags = mags.as_array()
            mags = np.array(mags.tolist())

        prior = self.prior_dict[magcol]

        return griddata(prior["grid"], prior["target"], mags, fill_value=0.0)

    def to_fits(self, include_bkg_priors=False, filename=None):
        """
        Save all prior data in a fits file. Each property prior is
        saved as a fits table in a different HDU extension.
        """
        hdus = [fits.PrimaryHDU()]
        for name in self.magnames:
            prior_table = self.to_table(name, include_bkg_priors)
            hdus.append(fits.BinTableHDU(prior_table, name=name))

        hdu = fits.HDUList(hdus)

        if filename:
            hdu.writeto(filename, overwrite=True)

        return hdu

    def to_table(self, magcol, include_bkg_priors=False):
        """
        Return prior data for property `magcol`
        as an astropy table.
        """
        prior = self.prior_dict[magcol]

        prior_table = Table()
        prior_table["grid"] = prior["grid"]
        prior_table["target"] = prior["target"]

        if include_bkg_priors and "field" in prior:
            prior_table["field"] = prior["field"]

        prior_table.meta["vol"] = prior["vol"]
        prior_table.meta["name"] = ",".join(prior["name"])

        return prior_table

    @classmethod
    def from_fits(cls, filename):
        """
        Create a ``Prior`` object from a fits file with the appropritate format.

        Parameters
        ----------
        filename : 'str'
            Name of the file with the priors.
        """
        prior_dict = {}
        with fits.open(filename) as hdul:
            for hdu in hdul[1:]:
                magname = hdu.header["EXTNAME"]
                if magname.startswith("PRIOR"):
                    prior_dict[magname] = cls._hdu_to_dict(hdu)

        if not prior_dict:
            raise ValueError(
                "File {} does not include PRIOR extensions".format(filename)
            )

        return cls(prior_dict=prior_dict)

    @staticmethod
    def _hdu_to_dict(hdu):
        # Reshape grid array so 1D grids have the same
        # (N, D) shape as multidimensional grids
        ndim = hdu.data["grid"].ndim
        length = len(hdu.data["grid"])
        grid = hdu.data["grid"].reshape(length, ndim)

        prior = {}
        prior["name"] = hdu.header["name"].split(",")
        prior["grid"] = grid
        prior["vol"] = hdu.header["vol"]
        prior["target"] = hdu.data["target"]

        try:
            prior["field"] = hdu.data["field"]

        except KeyError:
            pass

        return prior

    @classmethod
    def from_tables(cls, *tables):
        """
        Create a ``Prior`` object from astropy Tables.

        Parameters
        ----------
        tables : Astropy Table
            Table with prior data, as in the output of the `to_table` method.
        """
        prior_dict = {}
        for i, table in enumerate(tables):
            magname = "PRIOR{}".format(i)
            prior_dict[magname] = cls._table_to_dict(table)

        return cls(prior_dict=prior_dict)

    @staticmethod
    def _table_to_dict(table):
        prior = {}
        prior["name"] = table.meta["name"].split(",")
        prior["grid"] = table["grid"].data
        prior["vol"] = table.meta["vol"]
        prior["target"] = table["target"].data

        try:
            prior["field"] = table["field"].data

        except KeyError:
            pass

        return prior

    def plot(self, priorname, filename=None, contour_levels=3):
        """
        Plot prior PDFs for property `priorname`.
        """
        if priorname not in self.magnames:
            raise ValueError("Unknown prior name: {}".format(priorname))

        prior = self.prior_dict[priorname]
        ndim = prior["grid"].shape[1]

        if ndim > 2:
            fig = self._plot_prior_nd(prior, contour_levels)
            fig.suptitle(priorname)

        elif ndim > 1:
            self._plot_prior_2d(prior, contour_levels)
            plt.title(priorname)

        else:
            self._plot_prior_1d(prior)
            plt.title(priorname)

        if filename:
            plt.savefig(filename)
        else:
            plt.show()

        plt.close()

    @staticmethod
    def _plot_prior_1d(prior):
        plt.plot(prior["grid"], prior["target"], label="target")

        if "field" in prior:
            plt.plot(prior["grid"], prior["field"], label="field")

        plt.xlabel(prior["name"][0])
        plt.legend()

    def _plot_prior_2d(self, prior, levels):
        grid_points = round(prior["grid"].shape[0] ** (1 / 2))

        z = prior["target"].reshape([grid_points] * prior["grid"].shape[1]).T
        plt.imshow(z, aspect="equal", origin="lower", interpolation="bilinear")

        if "field" in prior:
            z = prior["field"].reshape([grid_points] * prior["grid"].shape[1]).T
            plt.contour(z, levels=levels, colors="white", alpha=0.5)

        locs = list(range(0, grid_points, 10))
        xlabels, ylabels = self._tick_labels(locs, prior["grid"], grid_points)

        plt.xticks(locs, xlabels)
        plt.yticks(locs, ylabels)

        plt.xlabel(prior["name"][0])
        plt.ylabel(prior["name"][1])

    def _plot_prior_nd(self, prior, levels):
        D = prior["grid"].shape[1]
        grid_points = round(prior["grid"].shape[0] ** (1 / D))

        z_target = prior["target"].reshape([grid_points] * D).T
        if "field" in prior:
            z_field = prior["field"].reshape([grid_points] * D).T

        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        spec = gridspec.GridSpec(ncols=D - 1, nrows=D - 1, figure=fig)

        idx = list(range(D))
        for c in combinations(idx, 2):
            idx_reduce = set(idx) - set(c)
            reduced_prior = np.add.reduce(z_target, axis=tuple(idx_reduce))

            ax = fig.add_subplot(spec[c[1] - 1, c[0]])
            ax.imshow(
                reduced_prior, aspect="equal", origin="lower", interpolation="bilinear"
            )

            if "field" in prior:
                reduced_field = np.add.reduce(z_field, axis=tuple(idx_reduce))
                plt.contour(reduced_field, levels=levels, colors="white", alpha=0.5)

            locs = list(range(0, grid_points, 10))
            xlabels, ylabels = self._tick_labels(locs, prior["grid"], grid_points, c)

            ax.set_xticks(locs)
            ax.set_yticks(locs)
            ax.set_xticklabels(xlabels)
            ax.set_yticklabels(ylabels)

            ax.set_xlabel(prior["name"][c[0]])
            ax.set_ylabel(prior["name"][c[1]])

        return fig

    @staticmethod
    def _tick_labels(locs, grid, grid_points, indexes=None):
        if indexes is None:
            indexes = (0, 1)

        assert len(indexes) == 2

        labels = []
        for idx in indexes:
            points = np.linspace(
                grid[:, idx].min(), grid[:, idx].max(), num=grid_points
            )
            labels.append(["{:.1f}".format(points[i]) for i in locs])

        return labels

    @staticmethod
    def _parse_mag_params(mags, magmin, magmax, bw):
        if magmin is None:
            magmin = ["auto"] * len(mags)

        if isinstance(magmin, float):
            magmin = [magmin] * len(mags)

        if magmax is None:
            magmax = ["auto"] * len(mags)

        if isinstance(magmax, float):
            magmax = [magmax] * len(mags)

        if bw is None or bw == "auto":
            bw = [None] * len(mags)

        if isinstance(bw, float):
            bw = [bw] * len(mags)

        return magmin, magmax, bw

    @staticmethod
    def _calculate_mags_grid(magnitudes, names, magmin, magmax):
        if isinstance(names, str):
            names = [names]
            magmin = [magmin]
            magmax = [magmax]

        if len(names) > 1:
            grid_points = 50
        else:
            grid_points = 200

        x = []
        vol = 1.0

        for name, mn, mx in zip(names, magmin, magmax):
            if mn == "auto" or magmin == "auto":
                mn = np.min(magnitudes[name]) - 1

            if mx == "auto" or magmax == "auto":
                mx = np.max(magnitudes[name]) + 1

            x.append(np.linspace(mn, mx, num=grid_points))
            vol *= x[-1][1] - x[-1][0]

        grid = np.array([[*kk] for kk in product(*x)])

        return grid, vol

    @staticmethod
    def _select_mags_data(magnitudes, names, grid):
        if isinstance(names, str):
            names = [names]

        mask = True
        for i, name in enumerate(names):
            mn = grid[:, i].min()
            mx = grid[:, i].max()

            mask_name = np.logical_and(magnitudes[name] >= mn, magnitudes[name] <= mx)
            mask = np.logical_and(mask, mask_name)

        mags_data = magnitudes[names][mask].as_array()

        return np.array(mags_data.tolist())


class Prior(PriorBase):
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
        bw=None,
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
        "target" catalogue. A "background" of "field" expectation is
        estimated by repeating this excersise at random positions
        within the FOV (defined by the input MOCs) of the
        observation. The difference between "target" and "field"
        provides an estimate of the properties of the true counteparts
        of the pcat sources within the scat catalogue.

        In practice the user defines properties of interest via the "mags"
        parameter. These can be magnitudes in a given filter, colours
        between two bands, optical morphology (point like vs extended)
        or any other parameter listed in the secondary catalogue. It is
        emphasised that properties of interest are not created (e.g. the
        code does not produce colours between filters) but should exist as
        columns in the secondary catalogue table. Combination of properties
        are also possible to define multi-dimensional spaces. The "mags"
        attribute should be a python list. Properties of combination of
        properties of interest are also python lists within the "mags" list:

        mags = [[R, GR], Z]

        The above notation defines two priors, the first is a
        2-dimensional one that includes the scat columns "R" (for
        example R-band magnitude) and "GR" (could be G-R colour
        between the G and R optical bands). The second prior is a
        1-Dimensional prior of the "Z" column (could be Z-band
        magnitude). This notation could be extend to more than
        2-dimensions (e.g. [[R, GR, RZ], Z]).

        The code then uses the columns listed in the mags to build N-D
        PDF within user defined ranges using kernel density estimators.
        We use gaussian kernels and values for the kernel's bandwith
        are estimated from the data, using the "ISJ" algorithm for
        1-D priors and smooth cross-validation for N-D priors.
        (***add references***)

        The range of a given parameter is defined via the parameters
        "magmin" and "magmax". These are expected to be python lists that
        contain for each input scat column (or combination of scat
        columns) in "mags" the corresponding minimum and maximum of the
        range. Values of the column outside this range will be ignored.
        The expected form of these parameters that are consistent with the
        above "mags" example:

         magmin =     [[10, -3], 8]
         magmax =     [[27, 3], 24]

        The code estimates N-dimensional KDE for both the
        "target" and "background" or "field" populations. It then
        substracts the two to determine the N-dimensional distribution
        of the true counterparts in a predefined grid based on the
        chosen magmin and magmax values. Negative PDF values are set to
        zero.

        The result of this process is a prior probability distribution
        density function for the counterparts of the sources in the
        primary catalogue. There will be one prior distribution for
        each entry in "mags". It is emphasized that the priors are
        probability distribution density functions, i.e. the values
        are divided by the bin size (or bin volume in the case of
        N-dimensional histograms and their integral sums to unity.

        The class includes methods to save the resulting priors into a
        fits file.

        Priors can also be provided externally via fits file that
        follow the format conventions of the code. In this case the
        calculation described above is skipped and the prior
        probability density functions, "mags", "magmin", "magmax",
        are determined from the input file.

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
            extension are used to define independent 1-Dimensional priors
        magmin : python list that includes lists of `floats` or 'auto', optional
            Lower boundary of the histograms. Defaults to None. If
            None or "auto" then the minimum is determined from the data.
        magmax : python list that includes lists of `floats` or 'auto', optional
            Upper boundary of the histograms. Defaults to None.  If
            None or "auto" then the maximum is determined from the data.
        bw : `float`, python list of `floats` or 'None', optional
            Bandwith used for the prior KDEs. If 'None' then it is determined
            from the data.
        """
        if prior_dict is None:
            message = "Estimating priors using Catalogues: {} and {}..."
            log.info(message.format(pcat.name, scat.name))
            self._from_catalogues(
                pcat, scat, match_mags, rndcat, radius, mags, magmin, magmax, bw,
            )
        else:
            log.info("Using provided prior...")
            self.prior_dict = prior_dict
            self.rndcat = None

    def _from_catalogues(
        self, pcat, scat, match_mags, rndcat, radius, mags, magmin, magmax, bw,
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
            pcat, scat, radius, match_mags, mags, magmin, magmax, bw,
        )

    @classmethod
    def from_nway_hists(cls, cat, renorm_factors, path="."):
        """
        Create a ``Prior`` object using nway histogram files
        built for secondary Catalogue `cat`.
        """
        prior_dict = {}
        for mag in cat.mags.colnames:
            filename = "{}_{}_fit.txt".format(cat.name, mag)
            filename = os.path.join(path, filename)
            prior_dict[mag] = cls._from_nway_maghist(filename, renorm_factors[mag], mag)

        return cls(prior_dict=prior_dict)

    def to_nway_hists(self, output_path=None, cat_name=None):
        """
        Returns a dictionary with the prior histograms in
        a format compatible with nway. If `output_path` is not ``None``,
        a text file is created with a formatting compatible with nway.
        """
        # TODO: modify this to work with new ndpriors
        nway_hists = []
        for priorname in self.magnames:
            prior = self.prior_dict[priorname]

            if output_path is not None:
                magname = prior["name"][0]
                filename = self._set_nway_hist_filename(magname, cat_name, output_path)
            else:
                filename = None

            maghist = self._to_nway_maghist(prior, filename)
            nway_hists.append(maghist)

        return nway_hists

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
        bw,
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

        magmin, magmax, bw = self._parse_mag_params(mags, magmin, magmax, bw)

        prior_dict = {}
        for iprior, mnames, mmin, mmax, mb in zip(count(), mags, magmin, magmax, bw):
            grid, vol = self._calculate_mags_grid(field_cat.mags, mnames, mmin, mmax)
            sample_mags_data = self._select_mags_data(match_mags, mnames, grid)
            field_mags_data = self._select_mags_data(field_cat.mags, mnames, grid)

            prior_dict["PRIOR{}".format(iprior)] = self._mag_pdf(
                len(pcat),
                sample_mags_data,
                field_mags_data,
                renorm_factor,
                mnames,
                grid,
                vol,
                mb,
            )

        return prior_dict

    def _mag_pdf(
        self, pcat_nsources, target_mags, field_mags, renorm_factor, col, grid, vol, bw,
    ):
        target_kde = KDE(target_mags, bw=bw)
        field_kde = KDE(field_mags, bw=bw)

        target_pdf = target_kde.eval(grid)
        field_pdf = field_kde.eval(grid)

        target_prior = (
            len(target_mags) * target_pdf - renorm_factor * len(field_mags) * field_pdf
        )
        target_prior[target_prior < 0] = 0.0

        if target_mags.shape[1] == 1:
            col = [col]

        return {
            "name": col,
            "grid": grid,
            "vol": vol,
            "target": target_prior / pcat_nsources,
            "field": field_pdf,
        }

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
    def _set_nway_hist_filename(magname, cat_name, output_path):
        if cat_name is None:
            raise ValueError("'cat_name' is needed!")

        filename = "{}_{}_fit.txt".format(cat_name, magname)
        filename = os.path.join(output_path, filename)

        return filename

    @staticmethod
    def _to_nway_maghist(prior, filename=None):
        if prior["grid"].shape[1] > 1:
            raise ValueError("Only 1D priors can be used in NWAY.")

        nrows = prior["target"].size

        hist_data = np.zeros((nrows, 4))
        hist_data[:, 0] = prior["grid"][:, 0] - prior["vol"] / 2
        hist_data[:, 1] = prior["grid"][:, 0] + prior["vol"] / 2
        hist_data[:, 2] = prior["target"] / np.sum(prior["target"] * prior["vol"])
        hist_data[:, 3] = prior["field"]

        if filename is not None:
            header = "{}\nlo hi selected others".format(filename)
            np.savetxt(filename, hist_data, fmt="%10.5f", header=header)

        return [row for row in hist_data.T]

    @staticmethod
    def _from_nway_maghist(filename, renorm_factor, mag):
        hist_data = Table.read(filename, format="ascii")

        vol = hist_data["lo"][1] - hist_data["lo"][0]
        grid = (hist_data["lo"] + hist_data["hi"]) / 2
        grid = grid.data.reshape(len(hist_data), 1)

        maghist = {
            "name": [mag],
            "grid": grid,
            "vol": vol,
            "target": renorm_factor * hist_data["selected"].data,
            "field": hist_data["others"].data,
        }

        return maghist


class BKGpdf(PriorBase):
    def __init__(
        self, cat=None, mags=None, magmin=None, magmax=None, bw=None, prior_dict=None,
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
            all the columns listed in the cat.mags Catalogue
            extension are used to define independent 1-Dimensional priors
        magmin : python list that includes lists of `floats` or 'auto', optional
            Lower boundary of the histograms. Defaults to None. If
            None or "auto" then the minimum is determined from the data.
        magmax : python list that includes lists of `floats` or 'auto', optional
            Upper boundary of the histograms. Defaults to None.  If
            None or "auto" then the maximum is determined from the data.
        bw : `float`, python list of `floats` or 'None', optional
            Bandwith used for the prior KDEs. If 'None' then it is determined
            from the data.

        Return
        ------
        """
        self.rndcat = None

        if prior_dict is None:
            if cat is None:
                raise ValueError("One Catalogue must be passed!")

            if cat.mags is None:
                raise ValueError("No magnitudes defined in the catalogue!")

            self.prior_dict = self._calc_prior_dict(cat, mags, magmin, magmax, bw)

        else:
            self.prior_dict = prior_dict

    def _calc_prior_dict(
        self, cat, mags, magmin, magmax, bw,
    ):
        if mags is None:
            mags = list(cat.mags.colnames)

        log.info("Using columns: {}".format(mags))

        magmin, magmax, bw = self._parse_mag_params(mags, magmin, magmax, bw)

        prior_dict = {}
        for iprior, mnames, mmin, mmax, mb in zip(count(), mags, magmin, magmax, bw):
            grid, vol = self._calculate_mags_grid(cat.mags, mnames, mmin, mmax)
            sample_mags_data = self._select_mags_data(cat.mags, mnames, grid)

            prior_dict["PRIOR{}".format(iprior)] = self._mag_pdf(
                sample_mags_data, cat.area, mnames, grid, vol, mb
            )

        return prior_dict

    def _mag_pdf(self, mags, area, col, grid, vol, bw):
        target_kde = KDE(mags, bw=bw)
        target_pdf = len(mags) * target_kde.eval(grid) / area.to(u.arcsec ** 2).value

        if mags.shape[1] == 1:
            col = [col]

        return {
            "name": col,
            "grid": grid,
            "vol": vol,
            "target": target_pdf,  # in arcsec**-2!!!
        }
