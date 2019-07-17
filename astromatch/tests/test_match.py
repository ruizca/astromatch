import pytest

from astropy.utils.data import get_pkg_data_filename
from numpy import isfinite

from ..catalogues import Catalogue
from ..core import Match


def set_catalogues_moc():
    pcat_mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')    
    pcat = Catalogue(pcat_datafile, area=pcat_mocfile, name='pcat')

    scat_mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    scat_datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    scat = Catalogue(scat_datafile, area=scat_mocfile, name='scat',
                     coord_cols=['RA', 'DEC'], poserr_cols=['raErr', 'decErr'],
                     poserr_type='rcd_dec_ellipse', mag_cols=['uMag', 'gMag'])

    return pcat, scat

def set_catalogues_nomoc():
    from astropy import units as u

    area = 0.8393 * u.deg**2

    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')    
    pcat = Catalogue(pcat_datafile, area=area, name='pcat')

    scat_datafile = get_pkg_data_filename('data/testcat_3.fits')
    scat = Catalogue(scat_datafile, area=area, name='scat',
                     coord_cols=['RA', 'DEC'], poserr_cols=['raErr', 'decErr'],
                     poserr_type='rcd_dec_ellipse', mag_cols=['uMag', 'gMag'])

    return pcat, scat

def test_match_lr_cats_moc():
    pcat, scat = set_catalogues_moc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='lr')
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scat, Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scat) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in scat.mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    assert all(stats['completeness'] >= 0)
    assert all(stats['completeness'] <= 1)
    assert all(stats['reliability'] >= 0)
    assert all(stats['reliability'] <= 1)
    
def test_match_lr_cats_nomoc():
    pcat, scat = set_catalogues_nomoc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='lr')
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scat, Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scat) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in scat.mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    assert all(stats['completeness'] >= 0)
    assert all(stats['completeness'] <= 1)
    assert all(stats['reliability'] >= 0)
    assert all(stats['reliability'] <= 1)

def test_match_lr_params_moc():
    from astropy.table import Table

    pcat_mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    scat_datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[pcat_mocfile, scat_mocfile],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'],
               mag_cols=[None, ['uMag', 'gMag']])

    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='lr')
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scat, Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scat) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in xm._match.scat.mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    assert all(stats['completeness'] >= 0)
    assert all(stats['completeness'] <= 1)
    assert all(stats['reliability'] >= 0)
    assert all(stats['reliability'] <= 1)
    
def test_match_lr_params_nomoc():
    from astropy.table import Table
    from astropy import units as u

    area = 0.8393 * u.deg**2

    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_datafile = get_pkg_data_filename('data/testcat_3.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[area, area],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'],
               mag_cols=[None, ['uMag', 'gMag']])

    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='lr')
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scat, Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scat) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in xm._match.scat.mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    assert all(stats['completeness'] >= 0)
    assert all(stats['completeness'] <= 1)
    assert all(stats['reliability'] >= 0)
    assert all(stats['reliability'] <= 1)

def test_match_nway_cats_moc_nomags():
    pcat, scat = set_catalogues_moc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='nway', prior_completeness=0.9, use_mags=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scats[0]) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)
    
def test_match_nway_cats_nomoc_nomags():
    pcat, scat = set_catalogues_nomoc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='nway', prior_completeness=0.9, use_mags=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scats[0]) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

def test_match_nway_params_moc_nomags():
    from astropy.table import Table

    pcat_mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    scat_datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[pcat_mocfile, scat_mocfile],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'])

    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='nway', prior_completeness=0.9, use_mags=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scats[0]) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)
    
def test_match_nway_params_nomoc_nomags():
    from astropy.table import Table
    from astropy import units as u

    area = 0.8393 * u.deg**2

    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_datafile = get_pkg_data_filename('data/testcat_3.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[area, area],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'])

    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='nway', prior_completeness=0.9, use_mags=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scats[0]) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

def test_match_nway_cats_moc_mags():
    pcat, scat = set_catalogues_moc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='nway', prior_completeness=0.9,
           use_mags=True, bayes_prior=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scats[0]) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in scat.mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)
    
def test_match_nway_cats_nomoc_mags():
    pcat, scat = set_catalogues_nomoc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='nway', prior_completeness=0.9,
           use_mags=True, bayes_prior=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scats[0]) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in scat.mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

def test_match_nway_params_moc_mags():
    from astropy.table import Table

    pcat_mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    scat_datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[pcat_mocfile, scat_mocfile],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'],
               mag_cols=[None, ['uMag', 'gMag']])

    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='nway', prior_completeness=0.9,
           use_mags=True, bayes_prior=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scats[0]) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in xm._match.scats[0].mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)
    
def test_match_nway_params_nomoc_mags():
    from astropy.table import Table
    from astropy import units as u

    area = 0.8393 * u.deg**2

    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_datafile = get_pkg_data_filename('data/testcat_3.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[area, area],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'],
               mag_cols=[None, ['uMag', 'gMag']])

    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='nway', prior_completeness=0.9,
           use_mags=True, bayes_prior=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scats[0]) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in xm._match.scats[0].mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

@pytest.mark.remote_data
def test_match_xmatch_cats_moc_nomags():
    pcat, scat = set_catalogues_moc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='xmatch', use_mags=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scats[0]) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

@pytest.mark.remote_data
def test_match_xmatch_cats_nomoc_nomags():
    pcat, scat = set_catalogues_nomoc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='xmatch', use_mags=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scats[0]) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

@pytest.mark.remote_data
def test_match_xmatch_params_moc_nomags():
    from astropy.table import Table

    pcat_mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    scat_datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[pcat_mocfile, scat_mocfile],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'])

    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='xmatch', use_mags=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scats[0]) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

@pytest.mark.remote_data
def test_match_xmatch_params_nomoc_nomags():
    from astropy.table import Table
    from astropy import units as u

    area = 0.8393 * u.deg**2

    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_datafile = get_pkg_data_filename('data/testcat_3.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[area, area],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'])

    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='xmatch', use_mags=False)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scats[0]) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

@pytest.mark.remote_data
def test_match_xmatch_cats_moc_mags():
    pcat, scat = set_catalogues_moc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='xmatch', use_mags=True)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scats[0]) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in scat.mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

@pytest.mark.remote_data
def test_match_xmatch_cats_nomoc_mags():
    pcat, scat = set_catalogues_nomoc()
    
    xm = Match(pcat, scat)
    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='xmatch', use_mags=True)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat)
    assert len(xm._match.scats[0]) <= len(scat)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in scat.mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

@pytest.mark.remote_data
def test_match_xmatch_params_moc_mags():
    from astropy.table import Table

    pcat_mocfile = get_pkg_data_filename('data/testcat_moc_1.moc')
    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_mocfile = get_pkg_data_filename('data/testcat_moc_2.moc')
    scat_datafile = get_pkg_data_filename('data/testcat_moc_2.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[pcat_mocfile, scat_mocfile],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'],
               mag_cols=[None, ['uMag', 'gMag']])

    total_moc = xm.total_moc()
    assert total_moc is not None
    assert total_moc.sky_fraction > 0

    xm.run(method='xmatch', use_mags=True)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scats[0]) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in xm._match.scats[0].mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

@pytest.mark.remote_data
def test_match_xmatch_params_nomoc_mags():
    from astropy.table import Table
    from astropy import units as u

    area = 0.8393 * u.deg**2

    pcat_datafile = get_pkg_data_filename('data/testcat_moc_1.fits')
    pcat_data = Table.read(pcat_datafile)

    scat_datafile = get_pkg_data_filename('data/testcat_3.fits')
    scat_data = Table.read(scat_datafile)

    xm = Match(pcat_data, scat_data, 
               area=[area, area],
               name=['pcat', 'scat'],
               coord_cols=[['RA', 'DEC'], ['RA', 'DEC']],
               poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],
               poserr_type=['circle', 'rcd_dec_ellipse'],
               mag_cols=[None, ['uMag', 'gMag']])

    total_moc = xm.total_moc()
    assert total_moc is None

    xm.run(method='xmatch', use_mags=True)
    results = xm.results

    assert isinstance(xm._match.pcat, Catalogue)
    assert isinstance(xm._match.scats[0], Catalogue)
    assert len(xm._match.pcat) <= len(pcat_data)
    assert len(xm._match.scats[0]) <= len(scat_data)
    assert len(results) >= len(xm._match.pcat)

    xm.set_best_matchs()
    assert 'best_match_flag' in xm.results.colnames

    xm.set_best_matchs(false_rate=0.1)
    assert 'best_match_flag' in xm.results.colnames

    priors = xm.priors    
    for magcol in xm._match.scats[0].mags.colnames:
        assert priors.qcap(magcol) >= 0
        assert magcol in priors.prior_dict

    ncutoff = 101
    stats = xm.stats(ncutoff=ncutoff)

    assert len(stats) == ncutoff
    
    mask = isfinite(stats['completeness'])
    assert all(stats['completeness'][mask] >= 0)
    assert all(stats['completeness'][mask] <= 1)
    
    mask = isfinite(stats['reliability'])
    assert all(stats['reliability'][mask] >= 0)
    assert all(stats['reliability'][mask] <= 1)

def test_get_matchs():
    pcat, scat = set_catalogues_moc()
    
    xm = Match(pcat, scat)
    xm.run(method='lr')
    xm.set_best_matchs()
    
    matchs = xm.get_matchs(match_type='all')
    assert all(matchs['ncat'] > 1)

    matchs = xm.get_matchs(match_type='primary')
    assert all(matchs['ncat'] > 1)
    assert all(matchs['match_flag'] == 1)

    matchs = xm.get_matchs(match_type='best')
    assert all(matchs['ncat'] > 1)
    assert all(matchs['best_match_flag'] == 1)

def test_offset():
    pcat, scat = set_catalogues_moc()
    
    xm = Match(pcat, scat)
    xm.run(method='lr')
    
    dra, ddec = xm.offset(pcat.name, scat.name, only_best=False)
    assert all(dra >= 0)
    assert all(ddec >= 0)
    
    xm.set_best_matchs()
    dra, ddec = xm.offset(pcat.name, scat.name, only_best=True)
    assert all(dra >= 0)
    assert all(ddec >= 0)

