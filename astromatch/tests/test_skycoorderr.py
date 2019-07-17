import pytest

from astropy import units as u
from astropy.table import Table
import numpy as np

from ..catalogues import SkyCoordErr

ERRTYPE = ['circle', 'ellipse', 'rcd_dec_ellipse',
               'cov_ellipse', 'cor_ellipse']

def test_circle_nounits_default():
    n = 10
    errdata = Table()
    errdata['RADECERR'] = [0.1]*n

    errtype = 'circle'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == 1
    assert errs.components['RADECERR'].unit == 'arcsec'
    
def test_circle_units_default():
    n = 10
    errdata = Table()
    errdata['RADECERR'] = np.array([0.1]*n) * u.deg

    errtype = 'circle'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == 1
    assert errs.components['RADECERR'].unit == 'arcsec'

def test_circle_nounits_nodefault():
    n = 10
    errdata = Table()
    errdata['RADECERR'] = [0.1]*n

    errtype = 'circle'
    strunit = 'deg'
    errs = SkyCoordErr(errdata, errtype=errtype, unit=[u.Unit(strunit)])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == 1
    assert errs.components['RADECERR'].unit == strunit
    
def test_circle_units_nodefault():
    n = 10
    errdata = Table()
    errdata['RADECERR'] = np.array([0.1]*n) * u.deg

    errtype = 'circle'
    strunit = 'deg'
    errs = SkyCoordErr(errdata, errtype=errtype, unit=[u.Unit(strunit)])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == 1
    assert errs.components['RADECERR'].unit == strunit

def test_ellipse_nounits_default():
    n = 10
    errdata = Table()
    errdata['ERRA'] = [0.1]*n
    errdata['ERRB'] = [0.1]*n
    errdata['PA'] = [5.0]*n

    errtype = 'ellipse'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    assert errs.components['ERRA'].unit == 'arcsec'
    assert errs.components['ERRB'].unit == 'arcsec'
    assert errs.components['PA'].unit == 'deg'
    
def test_ellipse_units_default():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['PA'] = np.array([31500.0]*n) * u.arcsec

    errtype = 'ellipse'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    assert errs.components['ERRA'].unit == 'arcsec'
    assert errs.components['ERRB'].unit == 'arcsec'
    assert errs.components['PA'].unit == 'deg'

def test_ellipse_nounits_nodefault():
    n = 10
    errdata = Table()
    errdata['ERRA'] = [0.1]*n
    errdata['ERRB'] = [0.1]*n
    errdata['PA'] = [5.0]*n

    errtype = 'ellipse'
    strunit = ['deg', 'deg', 'arcmin']
    errs = SkyCoordErr(errdata, errtype=errtype,
                       unit=[u.Unit(su) for su in strunit])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    for col, su in zip(errs.components.itercols(), strunit):
        assert col.unit == su

def test_ellipse_units_nodefault():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['PA'] = np.array([31500.0]*n) * u.arcsec

    errtype = 'ellipse'
    strunit = ['deg', 'deg', 'arcmin']
    errs = SkyCoordErr(errdata, errtype=errtype,
                       unit=[u.Unit(su) for su in strunit])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    for col, su in zip(errs.components.itercols(), strunit):
        assert col.unit == su

def test_rcddec_nounits_default():
    n = 10
    errdata = Table()
    errdata['RAERR'] = [0.1]*n
    errdata['DECERR'] = [0.1]*n

    errtype = 'rcd_dec_ellipse'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    assert errs.components['RAERR'].unit == 'arcsec'
    assert errs.components['DECERR'].unit == 'arcsec'
    
def test_rcddec_units_default():
    n = 10
    errdata = Table()
    errdata['RAERR'] = np.array([0.1]*n) * u.deg
    errdata['DECERR'] = np.array([0.1]*n) * u.deg

    errtype = 'rcd_dec_ellipse'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    assert errs.components['RAERR'].unit == 'arcsec'
    assert errs.components['DECERR'].unit == 'arcsec'

def test_rcddec_nounits_nodefault():
    n = 10
    errdata = Table()
    errdata['RAERR'] = [0.1]*n
    errdata['DECERR'] = [0.1]*n

    errtype = 'rcd_dec_ellipse'
    strunit = ['deg', 'deg']
    errs = SkyCoordErr(errdata, errtype=errtype,
                       unit=[u.Unit(su) for su in strunit])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    for col, su in zip(errs.components.itercols(), strunit):
        assert col.unit == su

def test_rcddec_units_nodefault():
    n = 10
    errdata = Table()
    errdata['RAERR'] = np.array([0.1]*n) * u.arcmin
    errdata['DECERR'] = np.array([0.1]*n) * u.arcmin

    errtype = 'rcd_dec_ellipse'
    strunit = ['deg', 'deg']
    errs = SkyCoordErr(errdata, errtype=errtype,
                       unit=[u.Unit(su) for su in strunit])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    for col, su in zip(errs.components.itercols(), strunit):
        assert col.unit == su

def test_cov_nounits_default():
    n = 10
    errdata = Table()
    errdata['ERRA'] = [0.1]*n
    errdata['ERRB'] = [0.1]*n
    errdata['COV'] = [5.0]*n

    errtype = 'cov_ellipse'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    assert errs.components['ERRA'].unit == 'arcsec'
    assert errs.components['ERRB'].unit == 'arcsec'
    assert errs.components['COV'].unit == 'arcsec2'
    
def test_cov_units_default():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['COV'] = np.array([2.0]*n) * u.arcmin**2

    errtype = 'cov_ellipse'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    assert errs.components['ERRA'].unit == 'arcsec'
    assert errs.components['ERRB'].unit == 'arcsec'
    assert errs.components['COV'].unit == 'arcsec2'

def test_cov_nounits_nodefault():
    n = 10
    errdata = Table()
    errdata['ERRA'] = [0.1]*n
    errdata['ERRB'] = [0.1]*n
    errdata['COV'] = [5.0]*n

    errtype = 'cov_ellipse'
    strunit = ['deg', 'deg', 'arcmin**2']
    errs = SkyCoordErr(errdata, errtype=errtype,
                       unit=[u.Unit(su) for su in strunit])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    for col, su in zip(errs.components.itercols(), strunit):
        assert col.unit == su

def test_cov_units_nodefault():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['COV'] = np.array([31500.0]*n) * u.arcsec**2

    errtype = 'cov_ellipse'
    strunit = ['deg', 'deg', 'arcmin**2']
    errs = SkyCoordErr(errdata, errtype=errtype,
                       unit=[u.Unit(su) for su in strunit])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    for col, su in zip(errs.components.itercols(), strunit):
        assert col.unit == su

def test_cor_nounits_default():
    n = 10
    errdata = Table()
    errdata['ERRA'] = [0.1]*n
    errdata['ERRB'] = [0.1]*n
    errdata['COR'] = [0.3]*n

    errtype = 'cor_ellipse'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    assert errs.components['ERRA'].unit == 'arcsec'
    assert errs.components['ERRB'].unit == 'arcsec'
    assert errs.components['COR'].unit == ''

def test_cor_units_default():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['COR'] = np.array([0.3]*n)

    errtype = 'cor_ellipse'
    errs = SkyCoordErr(errdata, errtype=errtype)

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    assert errs.components['ERRA'].unit == 'arcsec'
    assert errs.components['ERRB'].unit == 'arcsec'
    assert errs.components['COR'].unit == ''

def test_cor_nounits_nodefault():
    n = 10
    errdata = Table()
    errdata['ERRA'] = [0.1]*n
    errdata['ERRB'] = [0.1]*n
    errdata['COR'] = [0.3]*n

    errtype = 'cov_ellipse'
    strunit = ['deg', 'deg', '']
    errs = SkyCoordErr(errdata, errtype=errtype,
                       unit=[u.Unit(su) for su in strunit])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    for col, su in zip(errs.components.itercols(), strunit):
        assert col.unit == su

def test_cor_units_nodefault():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['COR'] = np.array([0.3]*n)

    errtype = 'cov_ellipse'
    strunit = ['deg', 'deg', '']
    errs = SkyCoordErr(errdata, errtype=errtype,
                       unit=[u.Unit(su) for su in strunit])

    assert len(errs) == n
    assert errs.errtype == errtype
    assert len(errs.components.colnames) == len(errdata.colnames)

    for col, su in zip(errs.components.itercols(), strunit):
        assert col.unit == su

def test_badtype():
    n = 10
    errdata = Table()
    errdata['RADECERR'] = [0.1]*n

    with pytest.raises(ValueError):
        SkyCoordErr(errdata, errtype='foo')

def test_covariance_matrix():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.arcsec
    errdata['ERRB'] = np.array([0.3]*n) * u.arcsec
    errdata['PA'] = np.array([5.0]*n) * u.deg

    errs = SkyCoordErr(errdata, errtype='ellipse')    
    cov = errs.covariance_matrix()

    assert cov.shape == (n, 2, 2)
    assert np.all(cov != 0)
    
def test_covariance_matrix_inverse():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.arcsec
    errdata['ERRB'] = np.array([0.3]*n) * u.arcsec
    errdata['PA'] = np.array([5.0]*n) * u.deg

    errs = SkyCoordErr(errdata, errtype='ellipse')    
    icov = errs.covariance_matrix(inverse=True)

    assert icov.shape == (n, 2, 2)
    assert np.all(icov != 0)
    
def test_transform_to_circle():
    from astropy.utils.exceptions import AstropyUserWarning

    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['COV'] = np.array([2.0]*n) * u.arcmin**2

    errs = SkyCoordErr(errdata, errtype='cov_ellipse')
    
    with pytest.warns(AstropyUserWarning):
        newerrs = errs.transform_to(errtype='circle')

    assert len(newerrs) == n
    assert newerrs.errtype == 'circle'

def test_transform_to_ellipse():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['COV'] = np.array([2.0]*n) * u.arcmin**2

    errs = SkyCoordErr(errdata, errtype='cov_ellipse')
    newerrs = errs.transform_to(errtype='ellipse')

    assert len(newerrs) == n
    assert newerrs.errtype == 'ellipse'
    
def test_transform_to_other():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['COV'] = np.array([2.0]*n) * u.arcmin**2

    errs = SkyCoordErr(errdata, errtype='cov_ellipse')
    
    with pytest.raises(NotImplementedError):
        errs.transform_to(errtype='rcd_dec_ellipse')
    
def test_transform_to_bad():
    n = 10
    errdata = Table()
    errdata['ERRA'] = np.array([0.1]*n) * u.deg
    errdata['ERRB'] = np.array([0.1]*n) * u.deg
    errdata['COV'] = np.array([2.0]*n) * u.arcmin**2

    errs = SkyCoordErr(errdata, errtype='cov_ellipse')
    
    with pytest.raises(ValueError):
        errs.transform_to(errtype='foo')
    
    