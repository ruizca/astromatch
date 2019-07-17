from astropy.utils.data import get_pkg_data_filename

from ..catalogues import Catalogue
from ..lr import LRMatch


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

def test_lr_rndprior():
    pcat, scat = set_catalogues()

    xm = LRMatch(pcat, scat)    
    match = xm.run(prior_method='random')

    assert len(match) >= len(pcat)
    assert all(match['prob_has_match'] >= 0)
    assert all(match['prob_has_match'] <= 1)
    assert all(match['prob_this_match'] >= 0)
    assert all(match['prob_this_match'] <= 1)

    match_mask = ~match['LR_BEST'].mask 
    assert all(match['LR_BEST'][match_mask] >= 0)
    assert all(match['Separation_pcat_scat'][match_mask] >= 0)

def test_lr_maskprior():
    pcat, scat = set_catalogues()

    xm = LRMatch(pcat, scat)    
    match = xm.run(prior_method='mask')

    assert len(match) >= len(pcat)
    assert all(match['prob_has_match'] >= 0)
    assert all(match['prob_has_match'] <= 1)
    assert all(match['prob_this_match'] >= 0)
    assert all(match['prob_this_match'] <= 1)

    match_mask = ~match['LR_BEST'].mask 
    assert all(match['LR_BEST'][match_mask] >= 0)
    assert all(match['Separation_pcat_scat'][match_mask] >= 0)
