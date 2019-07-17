import pytest

from astropy.coordinates import SkyCoord
from astropy.utils.data import get_pkg_data_filename

from ..catalogues import Catalogue, SkyCoordErr


def test_set_fromfile_areafrommocfile_nomags():
    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    
    cat = Catalogue(datafile, area=mocfile, name='test')

    assert cat.name == 'test'
    assert cat.area > 0
    assert cat.poserr_type == 'circle'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)
    assert cat.mags is None

def test_set_fromfile_areafrommoc_nomags():
    from mocpy import MOC

    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    moc = MOC.from_fits(mocfile)
    
    cat = Catalogue(datafile, area=moc, name='test')

    assert cat.name == 'test'
    assert cat.area > 0
    assert cat.poserr_type == 'circle'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)
    assert cat.mags is None

def test_set_fromfile_areafromquantity_nomags():
    from astropy import units as u

    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    
    area = 0.8393*u.deg**2
    cat = Catalogue(datafile, area=area, name='test')

    assert cat.name == 'test'
    assert cat.area == area
    assert cat.poserr_type == 'circle'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)
    assert cat.mags is None

def test_set_fromfile_areafrommocfile_mags():
    datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    
    mag_cols = ['uMag', 'gMag']
    cat = Catalogue(datafile, area=mocfile, name='test', coord_cols=['RA', 'DEC'],
                    poserr_cols=['raErr', 'decErr'], poserr_type='rcd_dec_ellipse',
                    mag_cols=mag_cols)

    assert cat.name == 'test'
    assert cat.area > 0
    assert cat.poserr_type == 'rcd_dec_ellipse'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)

    assert len(cat.mags) > 0
    assert len(cat.mags.colnames) == len(mag_cols)

def test_set_fromfile_areafrommoc_mags():
    from mocpy import MOC

    datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    moc = MOC.from_fits(mocfile)

    mag_cols = ['uMag', 'gMag']
    cat = Catalogue(datafile, area=moc, name='test', coord_cols=['RA', 'DEC'],
                    poserr_cols=['raErr', 'decErr'], poserr_type='rcd_dec_ellipse',
                    mag_cols=mag_cols)

    assert cat.name == 'test'
    assert cat.area > 0
    assert cat.poserr_type == 'rcd_dec_ellipse'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)

    assert len(cat.mags) > 0
    assert len(cat.mags.colnames) == len(mag_cols)

def test_set_fromfile_areafromquantity_mags():
    from astropy import units as u

    datafile = get_pkg_data_filename('data/testcat_moc_2.fits')

    area = 1.6786*u.deg**2
    mag_cols = ['uMag', 'gMag']
    cat = Catalogue(datafile, area=area, name='test', coord_cols=['RA', 'DEC'],
                    poserr_cols=['raErr', 'decErr'], poserr_type='rcd_dec_ellipse',
                    mag_cols=mag_cols)

    assert cat.name == 'test'
    assert cat.area == area
    assert cat.poserr_type == 'rcd_dec_ellipse'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)

    assert len(cat.mags) > 0
    assert len(cat.mags.colnames) == len(mag_cols)

def test_set_fromtable_areafrommocfile_nomags():
    from astropy.table import Table

    mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    data = Table.read(datafile)
    
    cat = Catalogue(data, area=mocfile, name='test')

    assert cat.name == 'test'
    assert cat.area > 0
    assert cat.poserr_type == 'circle'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)
    assert cat.mags is None
    
    assert len(cat) == len(data)
    assert len(cat.coords) == len(data)
    assert len(cat.poserr) == len(data)

def test_set_fromtable_areafrommoc_nomags():
    from mocpy import MOC
    from astropy.table import Table

    mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    moc = MOC.from_fits(mocfile)
    
    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    data = Table.read(datafile)

    cat = Catalogue(data, area=moc, name='test')

    assert cat.name == 'test'
    assert cat.area > 0
    assert cat.poserr_type == 'circle'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)
    assert cat.mags is None

    assert len(cat) == len(data)
    assert len(cat.coords) == len(data)
    assert len(cat.poserr) == len(data)

def test_set_fromtable_areafromquantity_nomags():
    from astropy import units as u
    from astropy.table import Table

    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    data = Table.read(datafile)

    area = 0.8393*u.deg**2
    cat = Catalogue(data, area=area, name='test')

    assert cat.name == 'test'
    assert cat.area == area
    assert cat.poserr_type == 'circle'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)
    assert cat.mags is None

    assert len(cat) == len(data)
    assert len(cat.coords) == len(data)
    assert len(cat.poserr) == len(data)

def test_set_fromtable_areafrommocfile_mags():
    from astropy.table import Table

    mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    data = Table.read(datafile)

    mag_cols = ['uMag', 'gMag']
    cat = Catalogue(data, area=mocfile, name='test', coord_cols=['RA', 'DEC'],
                    poserr_cols=['raErr', 'decErr'], poserr_type='rcd_dec_ellipse',
                    mag_cols=mag_cols)

    assert cat.name == 'test'
    assert cat.area > 0
    assert cat.poserr_type == 'rcd_dec_ellipse'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)

    assert len(cat) == len(data)
    assert len(cat.coords) == len(data)
    assert len(cat.poserr) == len(data)
    assert len(cat.mags) == len(data)
    assert len(cat.mags.colnames) == len(mag_cols)

def test_set_fromtable_areafrommoc_mags():
    from mocpy import MOC
    from astropy.table import Table

    mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    moc = MOC.from_fits(mocfile)

    datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    data = Table.read(datafile)

    mag_cols = ['uMag', 'gMag']
    cat = Catalogue(data, area=moc, name='test', coord_cols=['RA', 'DEC'],
                    poserr_cols=['raErr', 'decErr'], poserr_type='rcd_dec_ellipse',
                    mag_cols=mag_cols)

    assert cat.name == 'test'
    assert cat.area > 0
    assert cat.poserr_type == 'rcd_dec_ellipse'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)

    assert len(cat) == len(data)
    assert len(cat.coords) == len(data)
    assert len(cat.poserr) == len(data)
    assert len(cat.mags) == len(data)
    assert len(cat.mags.colnames) == len(mag_cols)

def test_set_fromtable_areafromquantity_mags():
    from astropy import units as u
    from astropy.table import Table

    datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    data = Table.read(datafile)

    area = 1.6786*u.deg**2
    mag_cols = ['uMag', 'gMag']
    cat = Catalogue(data, area=area, name='test', coord_cols=['RA', 'DEC'],
                    poserr_cols=['raErr', 'decErr'], poserr_type='rcd_dec_ellipse',
                    mag_cols=mag_cols)

    assert cat.name == 'test'
    assert cat.area == area
    assert cat.poserr_type == 'rcd_dec_ellipse'
    assert isinstance(cat.coords, SkyCoord)
    assert isinstance(cat.poserr, SkyCoordErr)

    assert len(cat) == len(data)
    assert len(cat.coords) == len(data)
    assert len(cat.poserr) == len(data)
    assert len(cat.mags) == len(data)
    assert len(cat.mags.colnames) == len(mag_cols)

def test_set_badarea():
    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    
    with pytest.raises(ValueError):
        Catalogue(datafile, area=0.8393)

def test_getitem():
    from astropy import units as u
    from numpy.random import choice

    area = 0.8393*u.deg**2
    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    cat = Catalogue(datafile, area=area, name='test')

    nitems = 10
    items = choice(len(cat), nitems, replace=False)
    
    assert len(cat[items]) == nitems

def test_select_by_id():
    from astropy import units as u
    from numpy.random import choice

    area = 0.8393*u.deg**2
    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    cat = Catalogue(datafile, area=area, name='test')

    nitems = 10
    ids = choice(cat.ids, nitems, replace=False)

    subcat = cat.select_by_id(ids)

    assert len(subcat) == nitems

def test_save():
    from astropy import units as u
    from astropy.table import Table

    area = 0.8393*u.deg**2
    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    cat = Catalogue(datafile, area=area, name='test')

    assert isinstance(cat.save(), Table)

def test_randomise():
    from astropy import units as u

    area = 0.8393*u.deg**2    
    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    cat = Catalogue(datafile, area=area, name='test')
    
    numrepeat = 3
    rndcat = cat.randomise(numrepeat=numrepeat)
    
    assert len(rndcat) == 3*len(cat)

def test_randomise_withmoc():
    from mocpy import MOC

    mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    moc = MOC.from_fits(mocfile)

    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    cat = Catalogue(datafile, area=moc, name='test')

    numrepeat = 3
    rndcat = cat.randomise(numrepeat=numrepeat)

    assert len(rndcat) > len(cat)
    assert all(moc.contains(rndcat.coords.ra, rndcat.coords.dec))

def test_randomise_nway():
    from astropy import units as u

    area = 0.8393*u.deg**2    
    datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    cat = Catalogue(datafile, area=area, name='test')

    rndcat = cat.randomise(numrepeat=1)

    assert len(rndcat) == len(cat)

def test_apply_moc():
    from mocpy import MOC

    datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    cat = Catalogue(datafile, area=mocfile, name='test',
                    poserr_cols=['raErr', 'decErr'],
                    poserr_type='rcd_dec_ellipse')

    mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    moc = MOC.from_fits(mocfile)
    newcat = cat.apply_moc(moc)

    assert len(cat) > len(newcat)
    assert moc.intersection(newcat.moc) is not None
