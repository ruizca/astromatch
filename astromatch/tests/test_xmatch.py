import pytest

from astropy.utils.data import get_pkg_data_filename

from ..catalogues import Catalogue
from ..xmatch import XMatch


def set_catalogues():
    mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')

    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')    
    pcat = Catalogue(pcat_datafile, area=mocfile, name='pcat')

    scat_datafile = get_pkg_data_filename('data/testcat_3.fits')
    scat = Catalogue(scat_datafile, area=mocfile, name='scat',
                     coord_cols=['RA', 'DEC'], poserr_cols=['raErr', 'decErr'],
                     poserr_type='rcd_dec_ellipse', mag_cols=['uMag', 'gMag'])

    scat_poserr_circle = scat.poserr.transform_to(errtype='circle')
    scat.poserr = scat_poserr_circle

    return pcat, scat

@pytest.mark.remote_data
def test_xmatch_nomags():
    pcat, scat = set_catalogues()

    xm = XMatch(pcat, scat)    
    match = xm.run(use_mags=False)
    
    assert len(match) >= len(pcat)
    assert all(match['prob_this_match'] >= 0)
    assert all(match['prob_this_match'] <= 1)
    assert all(match['p_single'] >= 0)
    assert all(match['p_single'] <= 1)
    assert all(match['p_single'] == match['dist_post'])

@pytest.mark.remote_data
def test_xmatch_mags():
    pcat, scat = set_catalogues()

    xm = XMatch(pcat, scat)    
    match = xm.run(use_mags=True)

    assert len(match) >= len(pcat)
    assert all(match['prob_this_match'] >= 0)
    assert all(match['prob_this_match'] <= 1)
    assert all(match['p_single'] >= 0)
    assert all(match['p_single'] <= 1)
    assert any(match['p_single'] != match['dist_post'])
