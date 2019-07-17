# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:01:24 2019

@author: alnoah
"""
import os
from itertools import count

from astropy import units as u
from astropy.table import Table, unique
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import ICRS
from astropy_healpix import HEALPix
from mocpy import MOC


from astromatch.catalogues import Catalogue, xmatch_mock_catalogues
from astromatch.priors import Prior
from astromatch import Match
from astromatch.xmatch import XMatchServer

cat_xmm_path = './astromatch/test_data/xa_xmm.fits'
moc_xmm_path = './astromatch/test_data/xa_xmm_moc.fits'

cat_sdss_path = './astromatch/test_data/xa_sdss_strid.fits'
moc_sdss_path = './astromatch/test_data/xa_sdss_moc.fits'
cat_wise_path = './astromatch/test_data/xa_allwise_strid.fits'
moc_wise_path = './astromatch/test_data/xa_allwise_moc.fits'
cat_hsc_path = './astromatch/test_data/xa_hsc.fits'
moc_hsc_path = './astromatch/test_data/xa_hsc_moc.fits'

cat_xmm_histerr = './astromatch/test_data/histox.fits'
cat_sdss_histerr = './astromatch/test_data/histos.fits'

test_data_folder = './astromatch/tests/data'


data = Table.read(cat_xmm_path)

hpring = 24289
data_mask = data['HEAL'] == hpring
data_sub = data[data_mask]
data_sub.keep_columns(['UXID', 'RA', 'DEC', 'RADEC_ERR'])

catfile = 'testcat_moc_1.fits'
catfile = os.path.join(test_data_folder, catfile)
data_sub.write(catfile, format='fits', overwrite=True)

nside = 2**6
hpRING = HEALPix(nside=nside, order='ring', frame=ICRS())
hpnested = hpRING.ring_to_nested(hpring)
moc = MOC.from_json({np.log2(nside): [hpnested]})

mocfile = 'testcat_moc_1.moc'
mocfile = os.path.join(test_data_folder, mocfile)
moc.write(mocfile, format='fits', overwrite=True)


hpring_sdss = 24544
hpnested_sdss = hpRING.ring_to_nested(hpring_sdss)
moc_sdss = MOC.from_json({np.log2(nside): [hpnested_sdss]})
moc_sdss = MOC.union(moc, moc_sdss)

mocfile = 'testcat_moc_2.moc'
mocfile = os.path.join(test_data_folder, mocfile)
moc_sdss.write(mocfile, format='fits', overwrite=True)

data = Table.read(cat_sdss_path)
data_mask = moc_sdss.contains(data['RA']*u.deg, data['DEC']*u.deg)
data_sub = data[data_mask]

data_subidx = np.random.choice(len(data_sub), size=int(len(data_sub)/50), replace=False)
data_sub = data_sub[data_subidx]
data_sub.keep_columns(['SRCID', 'RA', 'DEC', 'raErr', 'decErr', 'uMag', 'gMag'])

catfile = 'testcat_moc_2.fits'
catfile = os.path.join(test_data_folder, catfile)
data_sub.write(catfile, format='fits', overwrite=True)

data_mask = moc.contains(data_sub['RA']*u.deg, data_sub['DEC']*u.deg)
data_sub = data_sub[data_mask]

catfile = 'testcat_3.fits'
catfile = os.path.join(test_data_folder, catfile)
data_sub.write(catfile, format='fits', overwrite=True)


#cats = xmatch_mock_catalogues(xmatchserver_user='angel',# seed=1,
#                nTab=3, geometry='cone', ra=12.3647, dec=-68.4297, r=2,
#                #geometry='allsky', #mocfile=moc_xmm_path,
#                nA=40000, nB=25000, nC=30000,
#                nAB=6000, nAC=7000, nBC=8000, nABC=10000,
#                poserrAtype='CIRCLE', poserrAmode='histogram',
#                poserrAfile=cat_xmm_histerr, countA='COUNT',
#                paramA1step=0.1, paramA1col='0.5*(LOW+HIGH)',
#                paramA2step=0.1, paramA2col='0.5*(LOW+HIGH)',
#                paramA3step=0.1, paramA3col='0.5*(LOW+HIGH)',
#                #poserrAtype='CIRCLE', poserrAmode='formula', paramA1=0.4,
#                poserrBtype='RCD_DEC_ELLIPSE', poserrBmode='histogram',
#                poserrBfile=cat_sdss_histerr, countB='COUNT',
#                paramB1step=0.025, paramB1col='0.5*(LOW+HIGH)',
#                paramB2step=0.025, paramB2col='0.5*(LOW+HIGH)',
#                paramB3step=0.025, paramB3col='0.5*(LOW+HIGH)',
#                #poserrBtype='CIRCLE', poserrBmode='function', paramB1func='x',
#                #paramB1xmin=0.8, paramB1xmax=1.2, paramB1nstep=100,
#                poserrCtype='CIRCLE', poserrCmode='function',
#                paramC1func='exp(-0.5*(x-0.75)*(x-0.75)/0.01)/(0.1*sqrt(2*PI))',
#                paramC1xmin=0.5, paramC1xmax=1, paramC1nstep=10)
#
#for cat in cats:
#    print(cat.name)    
#    print(cat.area)
#    cat.save(filename='{}cone_hist.fits'.format(cat.name))

#cat_mockA = Catalogue('mockAcone_hist.fits', area=12.5664*u.deg**2,
#                      name='A', poserr_cols=['ePosA'],
#                      poserr_type='circle')
#cat_mockB = Catalogue('mockBcone_hist.fits', area=12.5664*u.deg**2,
#                      name='B', poserr_cols=['ePosA', 'ePosB'],
#                      poserr_type='rcd_dec_ellipse')
#cat_mockC = Catalogue('mockCcone_hist.fits', area=12.5664*u.deg**2,
#                      name='C', poserr_cols=['ePosA'], 
#                      poserr_type='circle')
#
#xm = Match(cat_mockA, cat_mockB, cat_mockC)
#
##results = xm.run(method='xmatch', use_mags=False, xmatchserver_user='angel', 
##                 completeness=0.9973) #0.999936
##results.write('mockABC_xmatch_nomags.fits', format='fits', overwrite=True)
#
#
#results = xm.run(method='nway', radius=6*u.arcsec, 
#                 use_mags=False, bayes_prior=False, store_mag_hists=False,
#                 prior_completeness=0.4)
#results.write('mockABC_nway_nomags.fits', format='fits', overwrite=True)
#
#stats = xm.stats()
#stats.write('mockABC_nway_nomags_stats.fits', format='fits', overwrite=True)
#
##results = Table.read('mockABC_xmatch_nomags.fits')
##stats = Table.read('mockABC_xmatch_nomags_stats.fits')
#
#best_match = results['match_flag'] == 1
#match = results['ncat'] > 1
#
#true_match_ABC = np.logical_and(results['SRCID_A'] == results['SRCID_B'],
#                                results['SRCID_A'] == results['SRCID_C'])
#
##true_match_AB = np.logical_and(results['SRCID_A'] == results['SRCID_B'],
##                               results['SRCID_C'] == ' ')
##
##true_match_AC = np.logical_and(results['SRCID_A'] == results['SRCID_C'],
##                               results['SRCID_B'] == ' ')
#
#true_match_AB = np.logical_and(results['SRCID_A'] == results['SRCID_B'],
#                               results['SRCID_C'] == '0.0')
#
#true_match_AC = np.logical_and(results['SRCID_A'] == results['SRCID_C'],
#                               results['SRCID_B'] == '0.0')
#                               
#true_match = np.logical_or(true_match_AB, true_match_AC)
#true_match = np.logical_or(true_match, true_match_ABC)
#
#false_match = np.logical_and(~true_match, match)
#
#mask_true = np.logical_and(best_match, true_match)
#mask_false = np.logical_and(best_match, false_match)
#
#print(results[mask_true])
#print(results[mask_false])
#print(stats.colnames)
#
#real_stats = np.zeros((len(stats) - 1, 5))
#pany = results['prob_has_match']
#for i, row in zip(count(), stats[:-1]):
#    mask_cut = np.logical_and(best_match, pany > row['cutoff'])
#    n = len(results[mask_cut])
#    
#    mask_true_cut = np.logical_and(mask_cut, mask_true)
#    n_true = len(results[mask_true_cut])
#    sample_true = np.array([1]*n_true + [0]*(n - n_true))
#
#    mask_false_cut = np.logical_and(mask_cut, mask_false)
#    n_false = len(results[mask_false_cut])
#    sample_false = np.array([1]*n_false + [0]*(n - n_false))
#    
#    # Use bootstraping to estimate fraction values and errors
#    nbootstrap = 10
#    subsample_ratio = np.zeros((nbootstrap, 2))
#    subsample_size = n//2
#    for j in range(nbootstrap):
#        subsample = np.random.choice(sample_true, subsample_size, replace=True)
#        subsample_ratio[j, 0] = len(np.where(subsample == 1)[0]) / subsample_size
#
#        subsample = np.random.choice(sample_false, subsample_size, replace=True)
#        subsample_ratio[j, 1] = len(np.where(subsample == 1)[0]) / subsample_size
#
#    real_stats[i, 0] = row['cutoff']
#    real_stats[i, 1] = subsample_ratio[:,0].mean()
#    real_stats[i, 2] = subsample_ratio[:,0].std()
#    real_stats[i, 3] = subsample_ratio[:,1].mean()
#    real_stats[i, 4] = subsample_ratio[:,1].std()
#
#fig = plt.figure()
#plt.errorbar(stats['cutoff'][:-1:5], real_stats[::5,1], yerr=real_stats[::5,2],
#             fmt="o", ms=0, elinewidth=1.5, ls="None", capsize=3, capthick=1.5)
#plt.plot(stats['cutoff'][:-1], stats['reliability'][:-1])
#
#plt.errorbar(stats['cutoff'][:-1:5], real_stats[::5,3], yerr=real_stats[::5,4],
#             fmt="o", ms=0, elinewidth=1.5, ls="None", capsize=3, capthick=1.5)
#plt.plot(stats['cutoff'][:-1], stats['error_rate'][:-1])
#
#plt.savefig('mocktest_stats_nway.pdf')
#plt.close()
#
#
#
#
#
#
##results = xm.run(method='nway', radius=6*u.arcsec, 
##                 use_mags=False, bayes_prior=False, store_mag_hists=False,
##                 prior_completeness=0.4)
##results.write('mockABC_nway_nomags.fits', format='fits', overwrite=True)
#
#
#
#
#
#
#
##cat_xmm = Catalogue(cat_xmm_path, area=moc_xmm_path)
##xms = XMatchServer(user='angel')
##try:
##    #xms.get('wise_srclist_binid001_ukidss.fits')
##    #xms.remove('kktest.txt')    
##    #files = xms.ls()
##    print(xms.logged())
##    #print('kktest3.txt' in files['name'])
##except:
##    pass
##xms.logout()
#    
##xm = Match(cat_xmm_path, cat_sdss_path,
##           area=[moc_xmm_path, moc_sdss_path],
##           name=['xmm', 'sdss'],
##           poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr']],bayes_prior=False, s
##           poserr_type=['circle', 'rcd_dec_ellipse'],
##           mag_cols=[None, ['uMag', 'gMag', 'rMag', 'iMag', 'zMag']])
#
##xm = Match(cat_xmm_path, cat_sdss_path, cat_wise_path,
##           area=[moc_xmm_path, moc_sdss_path, moc_wise_path],
##           name=['xmm', 'sdss', 'wise'],
##           poserr_cols=[['RADEC_ERR'], ['raErr', 'decErr'], ['eeMaj', 'eeMin', 'eePA']],
##           poserr_type=['circle', 'rcd_dec_ellipse', 'ellipse'],
##           mag_cols=[None, ['uMag', 'gMag', 'rMag', 'iMag', 'zMag'], ['w1Mag', 'w2Mag']])
##
##results = xm.run(method='xmatch', use_mags=False, xmatchserver_user='angel', completeness=0.999936)
##results.write('xa_xmmsdsswise_xm_nomags.fits', format='fits', overwrite=True)
##stats = xm.stats(plot_to_file='xa_xmmsdsswise_xm_nomags_cutoff_stats')
#
##xm.set_best_matchs(false_rate=None, calibrate_with_random_cat=False)
##xm.results.write('xa_xmmsdsswise_xm_nomags_best.fits', format='fits', overwrite=True)
##stats = xm.stats()
##
##match_rnd = xm.set_best_matchs(false_rate=None, calibrate_with_random_cat=True, xmatchserver_user='angel', completeness=0.999936)
##xm.results.write('xa_xmmsdsswise_xm_nomags_bestrnd.fits', format='fits', overwrite=True)
##stats_rnd = xm.stats(match_rnd)bayes_prior=False, s
##
##plt.plot(stats['completeness'], 1 - stats['reliability'], label='probabilities')
##plt.plot(stats_rnd['completeness'], stats_rnd['error_rate'], label='random match')
##plt.xlabel('completeness')
##plt.ylabel('error rate')
##plt.legend()
##plt.savefig('xa_xmmsdsswise_xm_nomags_stats_comparison.pdf')
#
#
#
#
##xm = Match(cat_xmm_path, cat_hsc_path,bayes_prior=False, s
##           area=[moc_xmm_path, moc_hsc_path],
##           name=['xmm', 'hsc'],
##           poserr_cols=[['RADEC_ERR'], ['RADEC_ERR']],
##           poserr_type=['circle', 'circle'],
##           mag_cols=[None, ['gMag', 'rMag', 'iMag', 'zMag', 'yMag']])
#
##p = Prior(xm.catalogues[0], xm.catalogues[1])
##print(p.prior_dict)
##print(xm.catalogues[0].name, xm.catalogues[1].name)
##print(xm.catalogues[1].nway_dict())
##print(xm.catalogues[1].poserr)
##xm.offset('xmm', 'sdss')
#
##for key, prior in p.prior_dict.items():
##    mbins = p.bins_midvals(key)
##    plt.plot(mbins, prior['good'])
##    plt.plot(mbins, prior['field'])
##    plt.title(key)
##    plt.show()
#
##p = Prior.from_nway_hists(xm.catalogues[1].name, xbayes_prior=False, sm.catalogues[1].mags.colnames)
##print(p.prior_dict)
##print(p.bins_midvals('uMag'))
#
##results = xm.run(method='lr', radius=6*u.arcsec, prior_method='random')
###results.write('xa_xmmsdss_lr.fits', format='fits', overwrite=True)
####
##match_rnd = xm.set_best_matchs(false_rate=None, calibrate_with_random_cat=False)
###xm.results.write('xa_xmmsdss_lr_best_randomcat.fits', format='fits', overwrite=True)
##stats = xm.stats()
##
##match_rnd = xm.set_best_matchs(false_rate=None, calibrate_with_random_cat=True)
###xm.results.write('xa_xmmhsc_lr_best_random.fits', format='fits', overwrite=True)
##stats_rnd = xm.stats(match_rnd)
##
##plt.plot(stats['completeness'], 1 - stats['reliability'], label='probabilities')
##plt.plot(stats_rnd['completeness'], stats_rnd['error_rate'], label='random match')
##plt.xlabel('completeness')
##plt.ylabel('error rate')
##plt.legend()
##plt.savefig('xa_xmmhsc_lr_stats_comparison.pdf')
#
#
#
##for mag in xm.priors.get_magnames():
##    filename = 'xa_xmmhsc_lr_prior_random_{}.pdf'.format(mag)
##    xm.priors.plot(mag, filename=filename)
#
#
##results = xm.run(method='lr', radius=6*u.arcsec, prior_method='mask')
##match_rnd = xm.set_best_matchs(false_rate=None, calibrate_with_random_cat=False)
##xm.results.write('xa_xmmhsc_lr_best_mask.fits', format='fits', overwrite=True)
##
##for mag in xm.priors.get_magnames():
##    filename = 'xa_xmmhsc_lr_prior_mask_{}.pdf'.format(mag)
##    xm.priors.plot(mag, filename=filename)
#    
##
##stats = xm.stats()
##stats.write('xa_xmmsdss_lr_stats.fits', format='fits', overwrite=True)
#
##stats = xm.stats(match_rnd)
##stats.write('xa_xmmsdss_lr_statsrnd.fits', format='fits', overwrite=True)
#
##priors = xm.priors
##for mag in priors.get_magnames():
##    plotfile = '{}_lr_prior.pdf'.format(mag)
##    priors.plot(mag, plotfile)
#
#
##xm.lr.write('kk_lr.fits', format='fits', overwrite=True)
##dra, ddec = xm.offset('xmm', 'sdss', only_best=False)
##dra2, ddec2 = xm.offset('xmm', 'sdss', only_best=True)
##plt.hist(dra, bins='auto', alpha=0.5)
##plt.hist(dra2, bins='auto', alpha=0.5)
##plt.show()
#
##xm.set_best_matchs(calibrate_with_random_cat=True)
#
#
##results = xm.run(method='nway', radius=6*u.arcsec, 
##                 use_mags=True, bayes_prior=False, store_mag_hists=False,
##                 prior_completeness=0.6)
##
##results.write('xa_xmmsdsswise_nway_nomags.fits', format='fits', overwrite=True)
###
##print(xm.priors)
##match_rnd = xm.set_best_matchs(false_rate=0.1, calibrate_with_random_cat=False)#, prior_completeness=0.6)
##match_rnd = xm.set_best_matchs(false_rate=0.1, calibrate_with_random_cat=True)#, prior_completeness=0.6)
##xm.results.write('xa_xmmsdss_nway_mags_mypriors_best.fits', format='fits', overwrite=True)
#
##stats = xm.stats()
##print(stats)
##stats.write('xa_xmmsdss_nway_stats.fits', format='fits', overwrite=True)
#
##stats = xm.stats(match_rnd)
##print(stats)
##stats.write('xa_xmmsdss_nway_statsrnd.fits', format='fits', overwrite=True)
#
##priors = xm.priors
##for mag in priors.get_magnames():
##    plotfile = '{}_nway_prior.pdf'.format(mag)
##    priors.plot(mag, plotfile)
#
#
##print(results)
#
##print(results)
###print(xm.lr.colnames)
##print(xm.priors.to_table())
##print(xm.priors.rndcat)
##cat_sdss = Catalogue(cat_sdss_path, area=moc_sdss_path, 
##                     poserr_cols=['raErr', 'decErr'],
##                     poserr_type='rcd_dec_ellipse')
##print(cat_sdss.poserr)
##
##cat_wise = Catalogue(cat_wise_path, area=moc_wise_path, 
##                     poserr_cols=['eeMaj', 'eeMin', 'eePA'],
##                     poserr_type='ellipse')
##print(cat_wise.poserr)
##print(cat_wise.poserr.transform_to('circle'))