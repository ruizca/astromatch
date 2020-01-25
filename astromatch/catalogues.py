"""
Module for building and manipulating astronomical catalogues.

@author: A.Ruiz
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import zip, range
from io import open

import os
import warnings
import tempfile
import subprocess
from copy import deepcopy
from string import ascii_uppercase

import numpy as np
from astropy import log
from astropy import units as u
from astropy.coordinates import SkyCoord
#from astropy.utils.misc import ShapedLikeNDArray
from astropy.table import Table, join, setdiff, unique, vstack
from astropy.units.quantity import Quantity
from astropy.utils.misc import indent
from astropy.utils.exceptions import AstropyUserWarning
from mocpy import MOC


# Global
ALLSKY_AREA_DEG = (4*np.pi * u.rad**2).to(u.deg**2)


class SkyCoordErr(object):
    """
    A class for the positional errors of a SkyCoord object
    """
    # TODO: Use ShapedLikeNDArray as base object
    ERRTYPE = ['circle', 'ellipse', 'rcd_dec_ellipse',
               'cov_ellipse', 'cor_ellipse']

    def __init__(self, data, errtype='circle', unit=None, errsys=None, check=True):

        self.errtype = self._set_errtype(errtype)
        self.components = self._set_components(data, unit)

        if errsys is not None:
            self.add_syserr(errsys)

        if check:
            self._check_components()

    def __repr__(self):
        comp_str = ', '.join(self.components.colnames)
        unit_str = ', '.join([str(col.unit) for col in self.components.itercols()])
        data_str = indent(str(self.components.as_array()))
        err_str = '<SkyCoordErr ({}): ({}) in {}\n{}>'

        return err_str.format(self.errtype, comp_str, unit_str, data_str)

    def __getitem__(self, key):
        item_data = self.components[key]
        if not isinstance(item_data, Table):
            # We do this because when key is an integer and the components
            # only have one column, it returns components[key] returns a row
            # instead of a Table.
            item_data = self.components[key:key+1]

        return SkyCoordErr(item_data, errtype=self.errtype, check=False)

    def __len__(self):
        return len(self.components)

    def transform_to(self, errtype='ellipse'):
        """
        Transform errors to `errtype`
        """
        not_implemented_errtypes = ['rcd_dec_ellipse',
                                    'cov_ellipse',
                                    'cor_ellipse']

        covmatrix = self.covariance_matrix()

        if errtype == 'circle':
            errs = self._to_circular(covmatrix)

        elif errtype == 'ellipse':
            errs = self._to_ellipse(covmatrix)

        elif errtype in not_implemented_errtypes:
            # TODO: implement remaining transformations
            raise NotImplementedError

        else:
            raise ValueError('Unknown error type: {}'.format(errtype))

        return errs

    def as_array(self):
        """
        Return error values as a numpy array.
        """
        errs = self.components
        if self.errtype == 'circle':
            #err_arrays = errs.columns[0].data << errs.columns[0].unit
            err_arrays = errs.columns[0].data * errs.columns[0].unit
        else:
            err_arrays = []
            for col in errs.itercols():
                #err_arrays.append(col.data << col.unit)
                err_arrays.append(col.data * u.Unit(col.unit))
            err_arrays = np.array(err_arrays)

        return err_arrays

    def covariance_matrix(self, inverse=False):
        """
        Returns the corresponding covariance matrix. If `inverse` is True,
        returns the inverse of the covariance matrix.
        """
        sigma_x, sigma_y, rhoxy = self._covariance_components()

        if inverse:
            V = self._inverse_covariance_matrix(sigma_x, sigma_y, rhoxy)
        else:
            V = self._covariance_matrix(sigma_x, sigma_y, rhoxy)

        return V

    def add_syserr(self, syserr):
        """
        Add systematic to the error components. Only works for circular errors.
        """
        if self.errtype == 'circle':
            data = self.components.columns[0].data
            unit = self.components.columns[0].unit
            err = data * u.Unit(unit)

            errcol = self.components.colnames[0]
            self.components[errcol] = np.sqrt(syserr**2 + err**2)

        else:
            raise NotImplementedError

    def _set_errtype(self, errtype):
        """
        Check that `errtype` is a valid value.
        """
        if errtype not in self.ERRTYPE:
            raise ValueError('Unknown error type: {}'.format(errtype))
        else:
            return errtype

    def _set_components(self, data, unit=None):
        """
        Define an astropy table with statistical positional errors
        (no systematic errors applied here). The number of columns depends
        on what kind of errors are defined
        """

        if unit is None:
            unit = self._get_default_units()

        poserr = Table()
        for col, col_unit in zip(data.colnames, unit):
            if data[col].unit is None:
                poserr[col] = data[col]*col_unit
            else:
                poserr[col] = data[col].to(col_unit)

#            # Set bad values to zero
#            good_mask = np.isfinite(poserr[col])
#            poserr[col][~good_mask] = 0.0
#
#            negative_mask = poserr[col] < 0
#            poserr[col][negative_mask] = 0.0

        return poserr

    def _check_components(self):
        """
        Check that all errors are positive and finite (not nan or inf)
        """
        for i, col in enumerate(self.components.colnames):
            if i >= 2:
                break

            if not all(np.isfinite(self.components[col])):
                raise ValueError('Some positional errors are not finite!')

            if not all(self.components[col] > 0):
                raise ValueError('Some positional errors are non positive!')

    def _get_default_units(self):
        """
        Define default units depending on the error type
        """
        if self.errtype == "circle":
            # RADEC_ERR (e.g. 3XMM)
            units = [u.arcsec]

        elif self.errtype == "ellipse":
            # major axis, minor axis, position angle (e.g. 2MASS)
            units = [u.arcsec, u.arcsec, u.deg]

        elif self.errtype == "rcd_dec_ellipse":
            # ra error, dec error (e.g. SDSS)
            units = [u.arcsec, u.arcsec]

        elif self.errtype == "cov_ellipse":
            # sigma_x, sigma_y, covariance
            units = [u.arcsec, u.arcsec, u.arcsec**2]

        elif self.errtype == "cor_ellipse":
            # sigma_x, sigma_y, correlation
            units = [u.arcsec, u.arcsec, u.arcsec/u.arcsec]

        else:
            raise ValueError('Wrong errtype!')

        return units

    def _to_ellipse(self, covmatrix):
        """
        Calculate components of the ellipse error from the covariance
        matrix and define a SkyCoordErr object with those components.
        """
        a, b, PA = self._covariance_to_ellipse(covmatrix)
        errs = Table([a, b, PA], names=['eeMaj', 'eeMin', 'eePA'])

        return SkyCoordErr(errs, errtype='ellipse')

    def _to_circular(self, covmatrix):
        """
        Estimate equivalent circular errors from the covariance matrix
        and define a SkyCoordErr object with those components.
        """
        if self.errtype != 'circle':
            message = ('Converting non-circular to circular errors! '
                       'New errors will preserve the area.')
            warnings.warn(message, AstropyUserWarning)

            # The determinat of the covariance matrix is related to the
            # 1 sigma area covered by the positional errors: A = pi * sqrt(|V|)
            # If we want a circular error that preserves the area:
            # r = |V|^(1/4)
            r = np.power(np.linalg.det(covmatrix), 0.25)

            errs = Table([r], names=['RADEC_ERR'])

            return SkyCoordErr(errs, errtype='circle')

        else:
            return self

    def _covariance_components(self):
        """
        Calculate the components of the covariance matrix from the errors
        """
        npars = len(self.components.colnames)
        errs = self.components

        if self.errtype == "circle":
            if npars != 1:
                raise ValueError('Wrong error type!')
            else:
                sigma_x = np.array(errs.columns[0])*errs.columns[0].unit
                sigma_y = np.array(errs.columns[0])*errs.columns[0].unit
                rhoxy = np.zeros(len(sigma_x))*errs.columns[0].unit**2

        elif self.errtype == "ellipse":
            if npars != 3:
                raise ValueError('Wrong error type!')
            else:
                err0 = np.array(errs.columns[0])*errs.columns[0].unit
                err1 = np.array(errs.columns[1])*errs.columns[1].unit
                err2 = np.array(errs.columns[2])*errs.columns[2].unit

                sigma_x = np.sqrt((err0*np.sin(err2))**2 +
                                  (err1*np.cos(err2))**2)
                sigma_y = np.sqrt((err0*np.cos(err2))**2 +
                                  (err1*np.sin(err2))**2)
                rhoxy = np.cos(err2)*np.sin(err2)*(err0**2 - err1**2)

        elif self.errtype == "rcd_dec_ellipse":
            if npars != 2:
                raise ValueError('Wrong error type!')
            else:
                sigma_x = np.array(errs.columns[0])*errs.columns[0].unit
                sigma_y = np.array(errs.columns[1])*errs.columns[1].unit
                rhoxy = np.zeros(len(sigma_x))*errs.columns[0].unit**2

        elif self.errtype == "cov_ellipse":
            if npars != 3:
                raise ValueError('Wrong error type!')
            else:
                sigma_x = np.array(errs.columns[0])*errs.columns[0].unit
                sigma_y = np.array(errs.columns[1])*errs.columns[1].unit
                rhoxy = np.array(errs.columns[2])*errs.columns[2].unit

        elif self.errtype == "cor_ellipse":
            if npars != 3:
                raise ValueError('Wrong error type!')
            else:
                err0 = np.array(errs.columns[0])*errs.columns[0].unit
                err1 = np.array(errs.columns[1])*errs.columns[1].unit
                err2 = np.array(errs.columns[2])*errs.columns[2].unit

                sigma_x = err0
                sigma_y = err1
                rhoxy = err2*err0*err1
        else:
            raise ValueError('Unknown error type: {}'.format(self.errtype))

        return sigma_x, sigma_y, rhoxy

    @staticmethod
    def _covariance_matrix(sigma_x, sigma_y, rhoxy):
        """
        Calculates the covariance matrix V with
        elements sigma_x, sigma_y and rhoxy.

        (Eq. 6 of Pineau+2017)
        """
        V = np.full((len(sigma_x), 2, 2), np.nan)
        V[:, 0, 0] = sigma_x**2
        V[:, 0, 1] = rhoxy
        V[:, 1, 0] = rhoxy
        V[:, 1, 1] = sigma_y**2

        return V

    @staticmethod
    def _inverse_covariance_matrix(sigma_x, sigma_y, rhoxy):
        """
        Calculates the inverse of the covariance matrix V with
        elements sigma_x, sigma_y and rhoxy

        (Eq. 7 of Pineau+2017)
        """
        K = (sigma_x*sigma_y)**2 - rhoxy**2

        Vinv = np.full((len(sigma_x), 2, 2), np.nan)
        Vinv[:, 0, 0] = sigma_y**2/K
        Vinv[:, 0, 1] = -rhoxy/K
        Vinv[:, 1, 0] = -rhoxy/K
        Vinv[:, 1, 1] = sigma_x**2/K

        return Vinv

    @staticmethod
    def _covariance_to_ellipse(V):
        """
        Given the covariance matrix V, returns the corresponding ellipse
        error with semi-major axis a, semi-minor axis b (in arcsec)
        and position angle PA (in degrees)
        """
        A = V[:, 0, 0] + V[:, 1, 1] # sigma_x**2 + sigma_y**2
        B = V[:, 1, 1] - V[:, 0, 0] # sigma_y**2 - sigma_x**2
        C = V[:, 1, 0]              # rho*sigma_x*sigma_y

        a = np.sqrt((A + np.sqrt(B**2 + 4*C**2))/2)
        b = np.sqrt((A - np.sqrt(B**2 + 4*C**2))/2)
        PA = np.arctan2(2*C, B)/2
        PA[PA < 0] += np.pi

        return a, b, PA*(180/np.pi)


class Catalogue(object):
    """
    A class for catalogue objects.

    Parameters
    ----------
    data_table : Astropy ``Table`` or ``str``
        Astropy ``Table`` with the catalogue data. Alternatively, the path
        to a file containing the catalogue data in a format compatible with
        Astropy (fits, csv, VOTable, etc) can be passed. It should contain at
        least three columns: the identification labels of the sources and their
        coordinates (e.g. RA and Dec).
    area : ``str``, ``MOC`` or ``Quantity``
        Sky area covered by the catalogue. the area can be defined as a path
        to the catalogue MOC, a mocpy ``MOC`` object or an Astropy ``Quantity``
        with units consistents with square deg.
    name : ``str`  or ``None``, optional
        Catalogue identification label. If None, it uses the file name of
        `data_table`. Defaults to ``None``.
    id_col : ``str`` or ``None``, optional
        Name of the column in `data_table` with the identification labels. If
        ``None``, it assumes that the first column contains the id labels.
    coord_cols : ``list``, optional
        Two element list with the column names for the coordinates. Defaults
        to ['RA', 'DEC'].
    frame : ``str`` or Astropy ``BaseCoordinateFrame``, optional
        Coordinates reference frame of `coord_cols`. Defaults to 'icrs'.
    poserr_cols : ``list``, optional
        List with the column names for the psotional errors. The size of
        the list depend on the error type.  See the SkyCoordErr documentation
        for details. Defaults to ['RADEC_ERR'].
    poserr_type : ``str``, optional
        Type of the positional errors. It can be 'circle', 'ellipse',
        'rcd_dec_ellipse', 'cov_ellipse' or 'cor_ellipse'. See the SkyCoordErr
        documentation for details. Defaults to 'circle'.
    mag_cols : ``list``, optional
        List with the column names for the magnitudes.

    Attributes
    ----------
    name : ``str``
        Catalogue identification label.
    ids : ``str`` or ``int``
        Source identification labels.
    coords : Astropy ``SkyCoord``
        Catalogue coordinates in ICRS frame.
    poserr : Astropy ``Quantity`` or ``None``
        Average positional error coords in units consistent with arcsec.
    moc : mocpy ``MOC`` or ``None``
        MOC of the catalogue.
    area : Astropy ``Quantity``
        Sky area covered by the catalogue in square deg.
    mags : Astropy ``Table`` or ``None``
        Source magnitudes.
    """

    def __init__(self, data_table, area, name=None, id_col=None,
                 coord_cols=['RA', 'DEC'], frame='icrs',
                 poserr_cols=['RADEC_ERR'], poserr_type='circle',
                 mag_cols=None):

        self.name = self._set_name(name, data_table)

        # if data_table is a string, assumes it is the path to the data file
        if isinstance(data_table, str):
            data_table = Table.read(data_table)

        self.ids = self._set_ids(data_table, id_col)
        self.coords = self._set_coords(data_table, coord_cols, frame)
        self.mags = self._set_mags(data_table, mag_cols)
        self.area, self.moc = self._set_area(area)
        self.poserr = self._set_poserr(data_table, poserr_cols, poserr_type)

        self._self_apply_moc() # keep only sources within self.moc, if exists


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return str(self.save(filename=None))

    def __getitem__(self, key):
        newcat = deepcopy(self)
        newcat.ids = self.ids[key]
        newcat.coords = self.coords[key]
        newcat.poserr = self.poserr[key]

        if self.mags is not None:
            newcat.mags = self.mags[key]

        return newcat

    @property
    def poserr_type(self):
        return self.poserr.errtype

    def apply_moc(self, moc, outside=False):
        """
        Returns a new ``Catalogue`` including only sources
        within the area defined by `moc`.

        Parameters
        ----------
        moc : mocpy ``MOC``
            MOC to be applied to the catalogue.
        outside : ``bolean``, optional
            If True, it returns the id labels of the sources outside `moc`.
            Defaults to False.
        """
        idx = moc.contains(self.coords.ra, self.coords.dec)

        if len(idx) > 0:
            newcat = self[idx]

            if self.moc is None:
                newcat.moc = moc
            else:
                newcat.moc = moc.intersection(self.moc)

            newcat.area = newcat.moc.sky_fraction * ALLSKY_AREA_DEG
        else:
            warnings.warn('No sources in moc!!!', AstropyUserWarning)
            newcat = None

        if outside:
            idx_out = moc.contains(self.coords.icrs.ra, self.coords.icrs.dec,
                                   keep_inside=False)
            return newcat, self.ids[idx_out]
        else:
            return newcat

#    def to_moc(self, radius=1*u.arcmin, moc_order=12):
#        """
#        Returns a moc defining the areas around the sources
#        of the Catalogue. It can be used as a source mask.
#
#        Parameters
#        ----------
#        radius : Astropy ``Quantity``, optional
#            Radius of the circular area to be selected around Catalogue
#            `coords` in units consistent with arcsec. Defaults to one arcmin
#        moc_order : ``int``
#            Maximum order of the resulting moc.
#        """
#        # PYMOC!!!
#        moc_srcs = catalog_to_moc(self.coords, radius, moc_order, inclusive=True)
#
#        # Convert PYMOC to MOCPY
#        mocdict = {order: list(cells) for order, cells in moc_srcs}
#        moc_srcs = MOC.from_json(mocdict)
#
#        return moc_srcs

    def select_by_id(self, ids):
        """
        Returns a new ``Catalogue`` including only sources with ids equal
        to `ids`. Sources in the new catalogue are ordered as in `ids`.

        Parameters
        ----------
        ids : ``list``
            List of ids to be selected.
        """
        catids = Table()
        catids['ID'] = self.ids
        catids['IDX'] = range(len(self.ids))

        newids = Table()
        newids['ID'] = ids#.columns[0]
        newids['newIDX'] = range(len(ids))

        joincat = join(newids, catids, keys='ID', join_type='left')
        joincat.sort('newIDX') # This way we always get the same row order as in ids
        joinidx = joincat['IDX'].data

        return self[joinidx]

    def remove_by_id(self, ids):
        """
        Returns a new ``Catalogue`` with `ids` sources removed

        Parameters
        ----------
        ids : ``list`` or ``Column``
            List of ids to be selected.
        """
        catids = Table()
        catids['ID'] = self.ids
        catids['IDX'] = range(len(self.ids))

        rmids = Table()
        rmids['ID'] = ids
        rmids['newIDX'] = range(len(ids))

        rmcat_ids = setdiff(catids, rmids, keys='ID')
        rmcat_ids.sort('IDX')

        return self.select_by_id(rmcat_ids['ID'])

    def join(self, cat, name=None):
        """
        Returns a new ``Catalogue`` joining the current catalogue with 'cat'. Both
        catalogue must be consistent: same coordinates, positional errors and
        magnitudes, if they are included.
        If the original catalogues have areas defined through MOCs, the final area is
        the union of their MOCs, otherwise the area of the current catalogue is used.
        If the original catalogues have common sources, repeated entries will be
        remove from the final catalogue.
        """
        if name is None:
            name = self.name

        join_cat_data = vstack([self.save(), cat.save()])
        join_cat_data = unique(join_cat_data)

        try:
            area = self.moc.union(cat.moc)
        except:
            area = self.area

        mag_cols = None
        if self.mags is not None:
            mag_cols = self.mags.colnames

        join_cat = Catalogue(
            join_cat_data,
            poserr_cols=self.poserr.components.colnames,
            poserr_type=self.poserr.errtype,
            area=area,
            name=self.name,
            mag_cols=mag_cols
        )

        return join_cat

    def randomise(self, r_min=20*u.arcsec, r_max=120*u.arcsec,
                  numrepeat=10, seed=None):
        """
        Returns a ``Catalogue`` object with random coordinates away
        from the positions of the original catalogue.

        Parameters
        ----------
        r_min : Astropy ``Quantity``, optional
            Minimum distance from original catalogue coordinates in angular
            units. Defaults to 20 arcsec.
        r_max : Astropy ``Quantity``, optional
            Maximum distance from original catalogue coordinates in angular
            units. Defaults to 120 arcsec.
        numrepeat : ``int``, optional
            The total number of sources in the new catalogue is `numrepeat`
            times the number of sources in the original catalogue. Defaults to
            10. If `numrepeat` is 1, the nway library is used to create a
            random catalogue with the same number of sources and preserving
            the spatial structure.
        """
        if self.moc is None:
            area = self.area
        else:
            area = self.moc

        if numrepeat == 1:
            # Use nway tool to generate a random catalogue:
            # good balance between reproducing local structures
            # and filling the field.
            r_min = r_min.to(u.arcsec).value
            poserr_cols = self.poserr.components.colnames

            with tempfile.NamedTemporaryFile() as input_file:
                filename = input_file.name
                self.save(filename)

                rnd_cat_data = self._nway_fake_catalogue(filename, radius=r_min)
                rnd_cat = Catalogue(
                    rnd_cat_data, area=area, poserr_cols=poserr_cols, name=self.name
                )
        else:
            # Use seed != None only for testing, to obtain the same random catalogue
            ra, dec = self._random_coords(
                0*u.deg, 360*u.deg, r_min, r_max, numrepeat, seed
            )
            ids = ['RND{:06d}'.format(i) for i in range(len(ra))]

            rnd_cat_data = Table()
            rnd_cat_data['SRCID'] = ids
            rnd_cat_data['RA'] = ra
            rnd_cat_data['DEC'] = dec

            # This catalogue have positional errors set to zero and hence it
            # shows a warning when the random catalogue is created. We use
            # this context manager to avoid showing the warning, which could
            # be misleading for the user.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=AstropyUserWarning)
                rnd_cat = Catalogue(
                    rnd_cat_data, area=area, poserr_cols=None, name=self.name
                )

        return rnd_cat

    def set_fake_counterparts(self, candidates):
        from scipy.stats import rayleigh

        # Assign fake counterparts
        idx_fake = np.random.choice(len(candidates), len(self))
        cat_fake = candidates[idx_fake]

        # Calculate coordinates for fake candidates
        # We randomize the positions of the fake counterpart around the
        # positions of the primary sources using a Rayleigh distribution
        mean_ra_fake = self.coords.ra.deg
        mean_dec_fake = self.coords.dec.deg

        # To estimate the variance for the Rayleigh distribution, we
        # circularize the errors of both catalogues:
        pcat_poserr_circ = self.poserr.transform_to('circle')
        cat_fake_poserr_circ = cat_fake.poserr.transform_to('circle')

        sig_fake = np.sqrt(
            (pcat_poserr_circ.components.columns[0].to(u.deg))**2 +
            (cat_fake_poserr_circ.components.columns[0].to(u.deg))**2
        )

        dr = rayleigh.rvs(loc=0.0, scale=sig_fake.value)
        theta = 2 * np.pi * np.random.random_sample(size=len(cat_fake))

        coords_ra_fake = mean_ra_fake + dr * np.cos(theta)
        coords_dec_fake = mean_dec_fake + dr * np.sin(theta)
        cat_fake.coords = SkyCoord(coords_ra_fake, coords_dec_fake, unit="deg")

        cat_fake.moc = self.moc
        cat_fake.area = self.area

        # We set the ids of the fake counterparts as the ids of this catalogue
        # for an easy identification of true counterparts
        cat_fake.ids = self.ids

        return cat_fake

    def save(self, filename=None, format='fits', include_mags=True):
        """
        Dump Catalogue to an Astropy Table and save it to a file.

        Parameters
        ----------
        filename : ``str``
            File path. If ``None``, only returns an Astropy Table.
        format : ``str``, optional
            Format of the output file (compatible with Astropy tables).
            Defaults to 'fits'.
        include_mags : ``bolean``, optional
            If ``True``, magnitudes are also included in the Astropy Table.
            Defaults to ``True``.
        """
        data_table = Table()
        try:
            data_table['SRCID_' + self.name] = self.ids
        except TypeError:
            # We do this in case len(ids) = 1
            data_table['SRCID_' + self.name] = [self.ids]

        data_table['RA'] = self.coords.ra
        data_table['DEC'] = self.coords.dec

        for errcol in self.poserr.components.colnames:
            data_table[errcol] = self.poserr.components[errcol]

        if self.mags and include_mags:
            for magcol in self.mags.colnames:
                data_table[magcol] = self.mags[magcol]

        if filename:
            data_table.write(filename, format=format, overwrite=True)

        return data_table

    def nway_dict(self, use_mags=True):
        """
        Converts the Catalogue object into a python dictionary
        with a structure compatible with the nway library.

        Parameters
        ----------
        use_mags : ``bolean``, optional
            If True, magnitudes are also included in the dictionary.
            Defaults to True.
        """
        # Code adapted from https://github.com/JohannesBuchner/nway/blob/api/nway-apitest.py
        # area in square degrees
        # poserr_col: error column name or numerical value (in arcsec)
        # coord_cols: ra/dec columns (in degrees)
        # magnitude_columns: list with (mag, magfile) sequence or empty list []
        # mag: column of something
        # magfile: file with mag histogram (bin, sel, all) or None (for auto)

        if self.poserr_type != 'circle':
            raise ValueError('Nway catalogues must have circular positional errors!')

        cat_dict = {}
        cat_dict['name'] = self.name
        cat_dict['srcid'] = self.ids.data
        cat_dict['ra'] = self.coords.ra.deg
        cat_dict['dec'] = self.coords.dec.deg
        cat_dict['area'] = self.area.value  # sky coverage in square degrees

        # Astrometrical errors in arcsec
        poserr = self.poserr.as_array()
        cat_dict['error'] = poserr.to(u.arcsec).value

        # magnitude columns
        # maghists: either (bin, sel, all) tuple or None (for auto)
        mags, magnames = [], []
        if use_mags and self.mags is not None:
            for magcol in self.mags.itercols():
                mag_all = magcol.data
                # mark negative magnitudes (e.g. -99 or -9.9999949E8) as undefined
                mag_all[mag_all < 0] = np.nan

                mags.append(mag_all)
                magnames.append(magcol.name)

        cat_dict['mags'] = mags
        cat_dict['maghists'] = []
        cat_dict['magnames'] = magnames

        return cat_dict

    def _set_name(self, name, data_table):
        if name is not None:
            return name

        if isinstance(data_table, str):
            # We assume that data_table is the path to the catalogue data.
            # We use as name of the catalogue the name of the file, without extension
            filename = os.path.basename(data_table)
            filename, ext = os.path.splitext(filename)
            return filename

    def _set_ids(self, data_table, id_col):
        if id_col is None:
            # Assume first column is the SRCID
            id_col = data_table.colnames[0]

        # set ids as strings
        ids = data_table[id_col].astype(str)
        #ids = np.array(data_table[id_col].data, dtype=str)

        # Workaround for a bug in hdf5 with Python 3
        # In python 3 strings are unicode by default,
        # and hdf5 doesn't handle that well
        # if ids.dtype.kind == 'U':
        #    ids = Column([iid.encode('utf8') for iid in ids], name=id_col)

        return ids

    def _set_coords(self, data_table, coord_cols, frame):
        coords = SkyCoord(ra=data_table[coord_cols[0]],
                          dec=data_table[coord_cols[1]],
                          unit='deg', frame=frame)
        return coords.icrs

    def _set_mags(self, data_table, mag_cols):
        # If data_table is a masked table, we convert it to a normal table
        # by filling masked values with -99 (assuming that they mask non-valid
        # magnitude values). This solves the problem of using a masked ndarray
        # in scipy interpolate. Then, we search for non-finite values in
        # the table (e.g. nan or inf) and change it to -99. This solves some
        # problems when using numpy histogram in python 3 (e.g. it fails to
        # automatically define a finite range if there are nans in the input,
        # even when the edges of the bins are passed).
        if mag_cols is not None:
            mags = data_table[mag_cols].filled(-99)
            for column in mag_cols:
                good_mask = np.isfinite(mags[column])
                mags[column][~good_mask] = -99

            return mags

    def _set_moc(self, mocfile):
        if mocfile is not None:
            return MOC.from_fits(mocfile)

    def _set_area(self, area):
        """
        Returns the area covered by the catalogue and the corresponding
        MOC, if defined.

        Parameters
        ----------
        area : ``str``, ``MOC`` or ``Quantity``
            area can be defined as a path to the catalogue MOC, a mocpy
            ``MOC`` object or an Astropy ``Quantity`` with units consistents
            with square deg.
        """
        # If area is a string, we assume is the path for a moc file
        if isinstance(area, str):
            moc = MOC.from_fits(area)
            area = moc.sky_fraction * ALLSKY_AREA_DEG

        elif isinstance(area, MOC):
            moc, area = area, area.sky_fraction * ALLSKY_AREA_DEG

        elif isinstance(area, Quantity):
            area = area.to(u.deg**2)
            moc = None

        else:
            raise ValueError('Invalid `area` value!')

        return area, moc

    def _set_poserr(self, data, columns, errtype):
        """
        Define a SkyCoordErr object with statistical positional errors
        (no systematic errors applied here). The number of components depends
        on what kind of errors are defined, given by `errtype`.
        """
        if columns is not None:
            errs = data[columns]
            check = True

        else:
            message = 'Positional errors are set to zero!!!'
            warnings.warn(message, AstropyUserWarning)

            r = np.zeros([len(data)], dtype=float) * u.arcsec
            errs = Table([r], names=['RADEC_ERR'])
            errtype = 'circle'
            check = False

        return SkyCoordErr(errs, errtype=errtype, check=check)

    def _self_apply_moc(self):
        if self.moc is not None:
            self = self.apply_moc(self.moc)

    def _random_coords(self, a_min, a_max, r_min, r_max, numrepeat, seed):
        # a_min, a_max, r_min, r_max: Quantity type
        num_rand  = numrepeat * len(self)
        np.random.seed(seed)
        r = r_min + (r_max - r_min)*np.random.random_sample(num_rand) # large kick
        a = a_min + (a_max - a_min)*np.random.random_sample(num_rand)

        dra = r.to(self.coords.ra.unit) * np.cos(a)    # offset in RA
        ddec = r.to(self.coords.dec.unit) * np.sin(a)  # offset in DEC

        rnd_dec = np.repeat(self.coords.dec, numrepeat) + ddec
        rnd_ra  = np.repeat(self.coords.ra, numrepeat) \
                  + dra/np.cos(rnd_dec)

        if self.moc is not None:
            idx = self.moc.contains(rnd_ra, rnd_dec)
            rnd_ra = rnd_ra[idx]
            rnd_dec = rnd_dec[idx]

        return rnd_ra, rnd_dec

    @staticmethod
    def _nway_fake_catalogue(input_file, radius=20):
        # Create a fake catalogue based on the positions of input_file.
        # No fake sources closer to `radius` arcsec with respect to the
        # original sources
        root, ext = os.path.splitext(input_file)
        output_file = '{}_fake{}'.format(root, ext)

        command = ('nway-create-fake-catalogue.py --radius {} {} {}')
        command = command.format(radius, input_file, output_file)
        subprocess.check_output(command, shell=True)

        fake_data = Table.read(output_file)
        os.remove(output_file)

        return fake_data


def xmatch_mock_catalogues(xmatchserver_user=None, seed=None, **kwargs):
    """
    Create mock catalogues using the tool provided by the XMatch service.

    Parameters
    ----------
    xmatchserver_user : ``str`` or ``None``, optional
        User name for the XMatch server. If ``None``, it uses anonymous access.
        Default is ``None``.
    seed : ``long`` or ``None``, optional
        Long integer to be used as seed for the random generator in the XMatch
        server. Default is `None`.
    **kwargs :
        Check the XMatch documentation to see all accepted arguments.

    Returns
    -------
    catalogues : ``list``
        List of `Catalogue` objects with the mock catalogues created
        by XMatch.
    """
    from .xmatch import XMatchServer

    if 'nTab' not in kwargs:
        raise ValueError('nTab parameter is missing!')

    catalogues = []
    cat_prefix = 'tmp_mock'
    cat_fmt = 'fits'

    area = _mockcat_area(**kwargs)

    xms = XMatchServer(user=xmatchserver_user)

    try:
        files_in_server = []
        for tag in ascii_uppercase[:kwargs['nTab']]:
            histfile_key = 'poserr{}file'.format(tag)
            if histfile_key in kwargs:
                files_in_server.append(os.path.basename(kwargs[histfile_key]))
                xms.put(kwargs[histfile_key])

        log.info('Creating mock catalogues in XMatch server...')
        with tempfile.NamedTemporaryFile() as xms_file:
            _make_xms_file(
                xms_file.name, prefix=cat_prefix, fmt=cat_fmt, seed=seed, **kwargs
            )
            xms.run(xms_file.name)

        log.info('Downloading results...')
        for tag in ascii_uppercase[:kwargs['nTab']]:
            cat_file = '{}{}.{}'.format(cat_prefix, tag, cat_fmt)
            files_in_server.append(cat_file)
            xms.get(cat_file)

            _mockcat_idcol_padwithzeros(cat_file)

            cat = Catalogue(
                cat_file,
                area=area,
                id_col='id',
                coord_cols=['posRA', 'posDec'],
                poserr_cols=['ePosA', 'ePosB', 'ePosPA'],
                poserr_type='ellipse',
                name=tag + 'mock',
            )

            catalogues.append(cat)
            os.remove(cat_file)

        cat_file = '{}.{}'.format(cat_prefix, cat_fmt)
        files_in_server.append(cat_file)

        log.info('Delete data from the server...')
        xms.remove(*files_in_server)
        xms.logout()

    except:
        xms.logout()
        raise

    return catalogues

def _mockcat_area(**kwargs):
    geometry = kwargs['geometry']

    if geometry == 'allsky':
        area = ALLSKY_AREA_DEG

    elif geometry == 'cone':
        r = kwargs['r'] * u.deg
        area = np.pi * r**2

    elif geometry == 'moc':
        area = MOC.from_fits(kwargs['mocfile'])

    else:
        raise ValueError('Unknown geometry: {}'.format(geometry))

    return area

def _mockcat_idcol_padwithzeros(catfile, len_idstr=None):
    cat = Table.read(catfile)

    if not len_idstr:
        len_idstr = len(cat['id'][0])

    cat['id'] = [idstr.strip().zfill(len_idstr) for idstr in cat['id']]
    cat.write(catfile, overwrite=True)

def _make_xms_file(filename, prefix='tmp_mock', fmt='fits', seed=None, **kwargs):

    if 'mocfile' in kwargs:
        kwargs['mocfile'] = os.path.basename(kwargs['mocfile'])

    for tag in ascii_uppercase[:kwargs['nTab']]:
        histfile_key = 'poserr{}file'.format(tag)
        if histfile_key in kwargs:
            kwargs[histfile_key] = os.path.basename(kwargs[histfile_key])

    args_str =  ' '.join(
        '{}={}'.format(key, value) for key, value in kwargs.items()
    )

    save_str = 'save prefix={0} suffix=.{1} common={0}.{1} format={1}'
    save_str = save_str.format(prefix, fmt)

    with open(filename, 'w') as f:
        f.write('synthetic ')

        if seed is not None:
            f.write('seed={} '.format(seed))

        f.write('{}\n'.format(args_str))
        f.write(save_str)
