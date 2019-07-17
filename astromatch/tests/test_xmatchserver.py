import pytest

from ..xmatch import XMatchServer, XMatchServerError


XMS = XMatchServer()

#def test_xms_login_user(user='user'):
#    xms = XMatchServer(user=user)
#    islogged = xms.logged()
#
#    assert islogged

@pytest.mark.remote_data
def test_xms_login_anonymous():
    assert XMS.logged()

@pytest.mark.remote_data
def test_xms_ls():
    lsdata = XMS.ls()
    assert len(lsdata) == 4

@pytest.mark.remote_data
def test_xms_put_fail():
    with pytest.raises(XMatchServerError):
        XMS.put('foo.fits')

@pytest.mark.remote_data
def test_xms_put(tmpdir):
    nfiles = len(XMS.ls())
    
    p = tmpdir.mkdir('sub').join('test.txt')
    p.write('test')

    XMS.put(str(p))
    lsdata = XMS.ls()

    assert len(lsdata) == nfiles + 1
    assert lsdata['name'][0] == 'test.txt'

@pytest.mark.remote_data
def test_xms_get_fail():
    with pytest.raises(XMatchServerError):
        XMS.get('foo.fits')

@pytest.mark.remote_data
def test_xms_get(tmpdir):
    p = tmpdir.mkdir('sub')
    xmsfile = '2mass.174.10491_7.22343_12.3arcmin.fits'    

    XMS.get(xmsfile, output_dir=str(p))
    assert len(tmpdir.listdir()) == 1

@pytest.mark.remote_data
def test_xms_rm_fail():
    with pytest.raises(XMatchServerError):
        XMS.remove('foo.fits')

@pytest.mark.remote_data
def test_xms_rm():
    nfiles = len(XMS.ls())
    XMS.remove('3xmme_uniquesources_v1.2.fits')

    assert len(XMS.ls()) == nfiles - 1

@pytest.mark.remote_data
def test_xms_run_fail(tmpdir):
    p = tmpdir.mkdir('sub').join('test.xms')
    p.write('bad')

    with pytest.raises(XMatchServerError):
        XMS.run(str(p))

@pytest.mark.remote_data
def test_xms_run(tmpdir):
    nfiles = len(XMS.ls())

    xms_cmd = ('get VizieRLoader tabname=V/139/sdss9 mode=cone '
               'center="174.10491 +7.22343" radius=12.3arcmin allcolumns\n'
               'save sdss9.vot votable')

    p = tmpdir.mkdir('sub').join('test.xms')
    p.write(xms_cmd)
    XMS.run(str(p))

    lsdata = XMS.ls()

    assert len(lsdata) == nfiles + 1
    assert 'sdss9.vot' in lsdata['name']

@pytest.mark.remote_data
def test_xms_logout():
    XMS.logout()
    assert not XMS.logged()
