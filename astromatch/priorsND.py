# -*- coding: utf-8 -*-
"""
astromatch module for calculation of magnitude priors.

@author: A.Ruiz
"""
import os
import warnings

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from scipy.interpolate import interp1d, interpn
from scipy.ndimage import convolve
from itertools import count
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy import log
from .catalogues import Catalogue


class PriorND(object):
    """
    Class for probability priors.
    """

    def __init__(
        self,
        pcat=None,
        scat=None,
        rndcat=True,
        radius=5*u.arcsec,
        mags=None,
        magmin=None,
        magmax=None,
        magbinsize=None,
        match_mags=None,
        prior_dict=None
    ):
        """
        Estimates the prior probability distribution for a source in the
        primary catalogue `pcat` having a counterpart in the secondary 
        catalogue `scat` with magnitude m. 
        
        The a priori probability is determined as follows. First, we estimate
        the magnitude distribution of the spurious matches and it is scaled
        to the area within which we search for counterparts. This is then 
        subtracted from the magnitude distribution of all counterparts in the
        secondary catalogue to determine the magnitude distribution of the 
        true associations.

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
        magmin : `float` or 'auto', optional
            Lower magnitude limit when estimating magnitude distributions.
            Default to 'auto'.
        magmax : `float` or 'auto', optional
            Upper magnitude limit when estimating magnitude distributions.
            Default to 'auto'.
        magbinsize : `float` or 'auto', optional
            Magnitude bin width when estimating magnitude distributions.
            Default to 'auto'.
        """

        if prior_dict is None:
            log.info('Estinating priors from Catalogue using columns {}...'.format(mags))
            self.magmin=magmin
            self.magmax=magmax
            self.magbinsize=magbinsize
            self.mags=mags
            self._from_catalogues(pcat, scat, match_mags, rndcat, 
                                  radius, mags, magmin, magmax, magbinsize)
            
        else:
            log.info('Using provided prior...')
            self.prior_dict = prior_dict
            self.rndcat = None
            self.magmin=magmin
            self.magmax=magmax
            self.magbinsize=magbinsize
            self.mags=mags

    def _from_catalogues(self, pcat, scat, match_mags, rndcat,
                              radius, mags, magmin, magmax, magbinsize):
        if None in [pcat, scat]:
            raise ValueError('Two Catalogues must be passed!')

        if rndcat is True:
            self.rndcat = pcat.randomise()
            
        elif isinstance(rndcat, Catalogue):
            self.rndcat = rndcat
            
        else:
            message = 'Using mask method for the prior calculation.'
            warnings.warn(message, AstropyUserWarning)
            self.rndcat = None

        #if match_mags is None:
        #    match_mags = self._get_match_mags(pcat, scat, radius)

        self.prior_dict = self._calc_prior_dict(
            pcat, scat, radius, match_mags, mags, magmin, magmax, magbinsize
        )

        #self.plot("prior0".upper())
        
    @property
    def magnames(self):
        return list(self.prior_dict.keys())

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
        #print("PRIOR:",self.prior_dict)
        #print("PRIOR:",col)

        if col not in self.prior_dict:
            raise ValueError('Unknown col: {}'.format(col))

        prior =  self.prior_dict[col.upper()]
        q=[]
        for i, magcol in enumerate(prior['name']):

            if(i==0):
                flags = np.ones(len(mags[magcol]), dtype=bool)
            
            edges = prior['edges']
            indeces = ((mags[magcol] - edges[i][0]) / (edges[i][1] - edges[i][0])).astype(int)
            m = indeces < 0
            indeces[m]=0
            flags[m]=False
            m = indeces>len(edges[i])-2
            indeces[m]=len(edges[i])-2
            #print(edges)
            #print(indeces[m])
            flags[m]=False
            #print("V: ",mags[magcol])
            #print(edges[i][0], edges[i][1] - edges[i][0])
            #print("I: ", indeces)
            #print('E: ',edges[i][indeces])
            q.append(indeces)
        q=tuple(q)
        pvals = prior['good'][q]
        pvals[np.logical_not(flags)]=0.0
        return pvals;
    

    def qcap(self, magcol):
        """
        Overall identification ratio for magnitude `magcol` 
        between the two catalogues used to build the prior.
        """
        if magcol.upper() not in self.prior_dict:
            raise ValueError('Unknown magcol: {}'.format(magcol.upper()))

        prior = self.prior_dict[magcol.upper()]

        # Whatch out prior is dN/dm,
        # i.e. I have divided by dm so it is probability density and
        # Sum(dN/dm*dm)=Q ie the overall identification ratio (not 1)
        return np.sum(prior['good']) * prior['vol']

    def bins_midvals(self, magcol):
        if magcol.upper() not in self.prior_dict:
            raise ValueError('Unknown Prior Name: {}'.format(magcol.upper()))

        edges = self.prior_dict[magcol.upper()]['edges']
        midvals = []
        for e in edges:
            midvals.append((e[1:] + e[:-1])/2)
        return midvals
    
    @staticmethod
    def _hdr2edges(hdu):

        if(isinstance(hdu, fits.ImageHDU)):
            N = len(hdu.shape)
            SIZE = hdu.shape
        else:
            N=1
            SIZE = [hdu.header['NAXIS2']]

        name=[];edges=[];vol=1.0
        pmin=[];pmax=[];pbin=[]
        for i in range(N):
            start = hdu.header['CRVAL{}L'.format(i+1)] 
            binn =  hdu.header['CDELT{}L'.format(i+1)]
            end = start + SIZE[i] * binn
            vol=vol*binn
            name.append(hdu.header['CTYPE{}L'.format(i+1)])
            edges.append(np.arange(start-binn/2, end, binn))
            pmin.append(start-binn/2)
            pmax.append(end - binn/2)
            pbin.append(binn)
        return edges, vol, name, pmin, pmax, pbin;

    @classmethod
    def from_table(cls, filename=None, include_bkg_priors=False):
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
        enames=[];eprior=[]         
        for h in hdul:
            enames.append(h.name)
            if( ("PRIOR" in h.name) and not ('FIELD' in h.name)):
               eprior.append(h.name)            
        if(len(enames)==0):
            raise ValueError('File {} does not inlcude PRIOR extensions'.format(filename))

        mags=[]; magmin=[];magmax=[];magbin=[]
        for col in eprior:
            maghist={}
            #maghist['name'] = col
            
            # INSTANCE IMAGE IE N-D PRIOR
            if(isinstance(hdul[col], fits.ImageHDU)):
               maghist['good'] = hdul[col].data
               maghist['edges'], maghist['vol'], maghist['name'], pmin1, pmax1, pbin1 = cls._hdr2edges(hdul[col])
               if(include_bkg_priors):
                   colfield="FIELD_{}".format(col)
                   if(colfield in enames):
                       maghist['field'] = hdul[colfield].data
                   else:
                       raise ValueError('No background counts for prior {} in file {}'.format(col, filename))

            # INSTANCE TABLE IE 1-D PRIOR
            elif(isinstance(hdul[col], fits.BinTableHDU)):
                maghist['good'] = hdul[col].data[col]
                maghist['edges'],maghist['vol'], maghist['name'],pmin1, pmax1, pbin1 = cls._hdr2edges(hdul[col])
                if(include_bkg_priors):
                    colfield="FIELD_{}".format(col)
                    if(colfield in enames):
                        maghist['field'] = hdul[colfield].data[colfield]
                    else:
                        raise ValueError('No background counts for prior {} in file {}'.format(col, filename))
                        
            mags.append(maghist['name'])
            magmin.append(pmin1);magmax.append(pmax1);magbin.append(pbin1)
            prior_dict[col.upper()] = maghist
        hdul.close()
        return cls(prior_dict=prior_dict, magmin=magmin, magmax=magmax, magbinsize=magbin, mags=mags)
            
    def to_table(self, include_bkg_priors=False):
        """
        
        AGE: this is not possible anyore: Dump prior data into an Astropy Table.
        
        the output has to be fits because the
        arrays may have different dimensions
        """
 
        hdu = fits.HDUList()
        hdu.append(fits.PrimaryHDU())
        for col in self.prior_dict.keys():

            hdrlist = self._getHDR(col, include_bkg_priors)
            for h in hdrlist:
                hdu.append(h)
        return hdu


    def _hdr(self,priorname):

        mbins = self.bins_midvals(priorname)
        hdr = fits.Header()            

        for i in range(len(mbins)):
            hdr.set('CTYPE{}L'.format(i+1), self.prior_dict[priorname]['name'][i], 'WCS coordinate name')
            hdr.set('CRPIX{}L'.format(i+1),   1, 'WCS reference pixel')
            hdr.set('CRVAL{}L'.format(i+1), (mbins[i])[0],  'WCS reference pixel value')                      
            hdr.set('CDELT{}L'.format(i+1), (mbins[i])[1] - (mbins[i])[0], 'WCS pixel size')            
        hdr.set('WCSNAMEL', 'PHYSICAL','WCS L name')
        hdr.set('WCSAXESL', len(mbins), 'No. of axes for WCS L')

        return hdr;
     
    def _getHDR(self, priorname, include_bkg_priors):

        if priorname.upper() not in self.prior_dict.keys():
            raise ValueError('Unknown Prior Name: {}'.format(priorname.upper()))

        if(len(self.prior_dict[priorname.upper()]['good'].shape) > 1):            
            hdr = self._hdr(priorname.upper())
            hdu = fits.HDUList()
            hdu.append(fits.ImageHDU(self.prior_dict[priorname.upper()]['good'], header=hdr, name=priorname.upper()))
            if(include_bkg_priors):
                hdu.append(fits.ImageHDU(self.prior_dict[priorname.upper()]['field'], header=hdr, name="FIELD_{}".format(priorname.upper())) )
        else:
            hdr = self._hdr(priorname.upper())
            mbins = self.bins_midvals(priorname.upper())
            hdu = fits.HDUList()
            c1 = fits.Column(name=priorname.upper(), array=self.prior_dict[priorname.upper()]['good'], format='D')
            c2 = fits.Column(name="MAG", array=mbins[0], format='D')
            hdu.append(fits.BinTableHDU.from_columns([c1,c2], header=hdr, name=priorname.upper()))
            if(include_bkg_priors):
                c3 = fits.Column(name="FIELD_{}".format(priorname.upper()), array=self.prior_dict[priorname.upper()]['field'], format='D')            
                c4 = fits.Column(name="MAG", array=mbins[0], format='D')
                hdu.append(fits.BinTableHDU.from_columns([c3,c4], header=hdr, name="FIELD_{}".format(priorname.upper())))
        return hdu;
            
    def plot(self, priorname, filename=None):

        import matplotlib.pyplot as plt

        if priorname.upper() not in self.prior_dict.keys():
            raise ValueError('Unknown Prior Name: {}'.format(priorname.upper()))

        if(len(self.prior_dict[priorname.upper()]['good'].shape) > 1 and len(self.prior_dict[priorname.upper()]['good'].shape)<=3):            
            hdr = self._hdr(priorname.upper())
            hdu = fits.HDUList()
            hdu.append(fits.ImageHDU(self.prior_dict[priorname.upper()]['good'].T, header=hdr, name="PRIOR"))
            hdu.append( fits.ImageHDU(self.prior_dict[priorname.upper()]['field'].T, header=hdr, name="FIELD"))
            hdu.writeto("{}.fits".format(priorname.upper()), overwrite=True)
                
        if(len(self.prior_dict[priorname.upper()]['good'].shape) == 1):            
                
            mbins = self.bins_midvals(priorname.upper())
            
            prior = self.prior_dict[priorname.upper()]
            plt.plot(mbins[0], prior['good'])
            plt.plot(mbins[0], prior['field'])            
            plt.title(priorname.upper())
            if filename is None:
                plt.show()
            else:
                plt.savefig("{}.png".format(priorname.upper()))
            plt.close()

        
    
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
        mask_radius=1*u.arcmin
    ):
        if self.rndcat is None:
            field_cat = self._field_sources(pcat, scat, mask_radius)
        else:
            field_cat = self._random_sources(scat, radius)

        renorm_factor = len(pcat) * np.pi*radius**2 / field_cat.area  # area_match / area field

        prior_dict = {}        
        
        if(mags is None):
            mags=list(match_mags.colnames)
        if(magmin is None):
            magmin=['auto']*len(mags)
        if(magmax is None):
            magmax=['auto']*len(mags)
        if(magbinsize is None):
            magbinsize=['auto']*len(mags)
            
        for iprior, col, mmin, mmax, mbin in zip(count(), mags, magmin, magmax, magbinsize):

            edges=[]
            if(isinstance(col, list)):
            
                sample=np.ndarray([len(match_mags[col[0]]), len(col)])
                field_sample=np.ndarray([len(field_cat.mags[col[0]]), len(col)])                               
                for i, c, mn, mx, mb in zip(count(), col, mmin, mmax, mbin):
                    if( (mn is "auto")):
                        mn = int(min(field_cat.mags[c])-0.5)
                    if( (mx is "auto")):
                        mx = int(max(field_cat.mags[c])+0.5)
                    if( (mb is "auto")):
                        mb=0.5            
                    sample[:,i] = match_mags[c]
                    field_sample[:,i] = field_cat.mags[c]                    
                    edges.append(np.arange(mn, mx+mb/2.0, mb))
                    
                prior_dict['prior{}'.format(iprior)] = self._mag_hist(len(pcat), sample, field_sample, renorm_factor, edges, col)
            else:
                if( (mmin is "auto")):
                    mmin = int(min(field_cat.mags[col])-0.5)
                if( (mmax is "auto")):
                    mmax = int(max(field_cat.mags[col])+0.5)
                if( (mbin is "auto")):
                    mbin=0.5            
                sample = match_mags[c]
                field_sample = field_cat.mags[c]                    
                edges.append(np.arange(mmin, mmax+mbin/2.0, mbin))
                prior_dict["PRIOR{}".format(iprior)]=self._mag_hist(len(pcat), sample, field_sample, renorm_factor, edges, col) 
            
        #print(prior_dict['prior0'])
        #sys.exit()
        return prior_dict

    def _mag_hist(
        self,
        pcat_nsources,
        good_mags,
        field_mags,
        renorm_factor,
        edges,
        col,
    ):

        good_counts, bins = np.histogramdd(good_mags, edges)
        field_counts, _ = np.histogramdd(field_mags, edges)
        vol=1.0
        for l in bins:
            vol=vol*(l[1:-1]-l[0:-2])[0]

        #print("VOLUME",vol)
        #print("EDGES", edges)
        #print("GOOD", good_counts)
            
        good_prior = good_counts - field_counts * renorm_factor
        good_prior[good_prior < 0] = 0.0
        # TODO: calculate general values for the convolution parameters
        # (magbinsize dependent)
        good_prior=gaussian_filter(good_prior, sigma=1.5, truncate=3)

        #ones = np.ones_like(good_prior)
        #ones_area=gaussian_filter(ones, sigma=1.5, truncate=3)
        #good_prior = good_prior / ones_area
        #print(ones_area)
        
#        # renormalise here to 0.999 in case
#        # prior sums to a value above unit
#        # Not unit because then zeros in Reliability
#        # estimation, i.e. (1-QCAP) term
#        test = good_prior.sum() / len(self.pcat) 
#        if test > 1:
#            good_prior = 0.999 * good_prior / test
#
        maghist = {}
        maghist['edges'] = edges
        maghist['vol'] = vol
        maghist['good'] = good_prior / pcat_nsources / vol
        maghist['field'] = 1.0*field_counts / len(field_mags) / vol
        maghist['name'] = col
        
        #print(len(edges[0]), len(edges[1]), vol, col, maghist['good'].shape)
        
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
        field_cat.area = scat.area - len(pcat)*np.pi*mask_radius**2
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
        rnd_scat.area = len(self.rndcat)*np.pi*radius**2
        rnd_scat.moc = None

        return rnd_scat


    @staticmethod
    def _to_nway_maghist(maghist, filename=None):
        nrows = maghist['good'].size

        hist_data = np.zeros((nrows, 4))
        hist_data[:, 0] = maghist['bins'][:-1]
        hist_data[:, 1] = maghist['bins'][1:]
        hist_data[:, 2] = maghist['good']
        hist_data[:, 3] = maghist['field']

        if filename is not None:
            header = '{}\nlo hi selected others'.format(filename)
            np.savetxt(filename, hist_data, fmt='%10.5f', header=header)

        return [row for row in hist_data.T]

    @staticmethod
    def _from_nway_maghist(filename):
        hist_data = Table.read(filename, format='ascii')
        bins = np.concatenate((hist_data['lo'], [hist_data['hi'][-1]]))

        maghist = {}        
        maghist['bins'] = bins
        maghist['good'] = hist_data['selected'].data
        maghist['field'] = hist_data['others'].data

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
    
    def __init__(self, cat=None, mags=None, magmin=None, magmax=None, magbinsize=None, pdf_dict=None):
        """
        Magnitude probability distribution of sources in ``Catalogue`` 'cat'.
        
        Parameters
        ----------
        cat : ``Catalogue``
            ``Catalogue`` object.
        magmin : `float` or 'auto', optional
            Lower magnitude limit when estimating magnitude distributions.
            Default to 'auto'.
        magmax : `float` or 'auto', optional
            Upper magnitude limit when estimating magnitude distributions.
            Default to 'auto'.
        magbinsize : `float` or 'auto', optional
            Magnitude bin width when estimating magnitude distributions.
            Default to 'auto'.

        Return
        ------
        bkg : Astropy ``Table``
            Table with the background probability distribution for each 
            available magnitude in the secondary catalogue.
        """

        #self.magnames = self._set_magnames(cat)        
        if pdf_dict is None:
            if cat is None:
                raise ValueError('No magnitudes defined in the catalogue!')
            elif cat.mags is None:
                raise ValueError('No magnitudes defined in the catalogue!')
            self.magmin=magmin
            self.magmax=magmax
            self.magbinsize=magbinsize
            self.mags=mags
            self.pdf_dict = self._calc_pdf(cat, mags, magmin, magmax, magbinsize)
        else:
            self.pdf_dict = pdf_dict
            self.magmin=magmin
            self.magmax=magmax
            self.magbinsize=magbinsize
            self.mags=mags
        #for n in self.pdf_dict.keys():
        #    self.plot(n)
        
        
    @property
    def magnames(self):
        return list(self.pdf_dict.keys())


    def bins_midvals(self, magcol):
        if magcol not in self.pdf_dict:
            raise ValueError('Unknown Prior Name: {}'.format(magcol))

        edges = self.pdf_dict[magcol]['edges']
        midvals = []
        for e in edges:
            midvals.append((e[1:] + e[:-1])/2)
        return midvals
 
    @staticmethod
    def _hdr2edges(hdu):

        if(isinstance(hdu, fits.ImageHDU)):
            N = len(hdu.shape)
            SIZE = hdu.shape
        else:
            N=1
            SIZE = [hdu.header['NAXIS2']]

        name=[];edges=[];vol=1.0
        pmin=[];pmax=[];pbin=[]
        for i in range(N):
            start = hdu.header['CRVAL{}L'.format(i+1)] 
            binn =  hdu.header['CDELT{}L'.format(i+1)]
            end = start + SIZE[i] * binn
            vol=vol*binn
            name.append(hdu.header['CTYPE{}L'.format(i+1)])
            edges.append(np.arange(start-binn/2, end, binn))
            pmin.append(start-binn/2)
            pmax.append(end - binn/2)
            pbin.append(binn)            
        return edges, vol, name, pmin, pmax, pbin;

    
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
        #print("BKG:",self.pdf_dict)
        #print("BKG:",col)
        if col not in self.pdf_dict:
            raise ValueError('Unknown col: {}'.format(col))

        prior =  self.pdf_dict[col]
        q=[]
        for i, magcol in enumerate(prior['name']):

            if(i==0):
                flags = np.ones(len(mags[magcol]), dtype=bool)
            
            edges = prior['edges']
            indeces = ((mags[magcol] - edges[i][0]) / (edges[i][1] - edges[i][0])).astype(int)
            m = indeces < 0
            indeces[m]=0
            flags[m]=False
            m = indeces>len(edges[i])-2
            indeces[m]=len(edges[i])-2
            flags[m]=False
            #print("V: ",mags[magcol])
            #print(edges[i][0], edges[i][1] - edges[i][0])
            #print("I: ", indeces)
            #print('E: ',edges[i][indeces])
            q.append(indeces)
        q=tuple(q)
        pvals = prior['pdf'][q]
        pvals[np.logical_not(flags)]=0.0
        return pvals;
    

    
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
        enames=[];eprior=[]         
        for h in hdul:
            enames.append(h.name)
            if("PRIOR" in h.name):
               eprior.append(h.name)            
        if(len(enames)==0):
            raise ValueError('File {} does not inlcude FIELD counts extensions'.format(filename))


        mags=[]; magmin=[];magmax=[];magbin=[]
        for col in eprior:
            maghist={}
            #maghist['name'] = col
            
            # INSTANCE IMAGE IE N-D PRIOR
            if(isinstance(hdul[col], fits.ImageHDU)):
               maghist['pdf'] = hdul[col].data * (1.0/u.arcsec**2)
               maghist['edges'], maghist['vol'], maghist['name'], pmin1, pmax1, pbin1 = cls._hdr2edges(hdul[col])

            # INSTANCE TABLE IE 1-D PRIOR
            elif(isinstance(hdul[col], fits.BinTableHDU)):
                maghist['pdf'] = hdul[col].data[col]  * (1.0/u.arcsec**2)
                maghist['edges'],maghist['vol'], maghist['name'], pmin1, pmax1, pbin1 = cls._hdr2edges(hdul[col])
                
            mags.append(maghist['name'])
            magmin.append(pmin1);magmax.append(pmax1);magbin.append(pbin1)
            prior_dict[col] = maghist        
        hdul.close()
        
        print(prior_dict)
        return cls(pdf_dict=prior_dict,  magmin=magmin, magmax=magmax, magbinsize=magbin, mags=mags)

    
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

    
    def _hdr(self,priorname):

        mbins = self.bins_midvals(priorname.upper())
        hdr = fits.Header()            

        for i in range(len(mbins)):
            hdr.set('CTYPE{}L'.format(i+1), self.pdf_dict[priorname.upper()]['name'][i], 'WCS coordinate name')
            hdr.set('CRPIX{}L'.format(i+1),   1, 'WCS reference pixel')
            hdr.set('CRVAL{}L'.format(i+1), (mbins[i])[0],  'WCS reference pixel value')                      
            hdr.set('CDELT{}L'.format(i+1), (mbins[i])[1] - (mbins[i])[0], 'WCS pixel size')
        hdr.set('WCSNAMEL', 'PHYSICAL','WCS L name')
        hdr.set('WCSAXESL', len(mbins), 'No. of axes for WCS L') 
        
        return hdr;
        
    def _getHDR(self, priorname):

        if priorname.upper() not in self.pdf_dict.keys():
            raise ValueError('Unknown Prior Name: {}'.format(priorname.upper()))

        if(len(self.pdf_dict[priorname.upper()]['pdf'].shape) > 1):            
            hdr = self._hdr(priorname.upper())
            hdu = fits.HDUList()
            hdu.append(fits.ImageHDU(self.pdf_dict[priorname.upper()]['pdf'].value, header=hdr, name=priorname.upper()))
        else:
            hdr = self._hdr(priorname.upper())
            mbins = self.bins_midvals(priorname.upper())
            hdu = fits.HDUList()
            c1 = fits.Column(name=priorname.upper(), array=self.pdf_dict[priorname.upper()]['pdf'].value, format='D')
            c2 = fits.Column(name="MAG", array=mbins[0], format='D')
            hdu.append(fits.BinTableHDU.from_columns([c1,c2], header=hdr, name=priorname.upper()))
        return hdu;

    def _set_magnames(self, cat):
        return cat.mags.colnames

    
 

    def plot(self, priorname, filename=None):

        import matplotlib.pyplot as plt

        if priorname.upper() not in self.pdf_dict.keys():
            raise ValueError('Unknown Prior Name: {}'.format(priorname.upper()))


        print(priorname.upper(), self.pdf_dict[priorname.upper()]['pdf'].shape)

        if(len(self.pdf_dict[priorname.upper()]['pdf'].shape) > 1 and len(self.pdf_dict[priorname.upper()]['pdf'].shape)<=3):            
            hdr = self._hdr(priorname.upper())
            #print(self.pdf_dict[priorname.upper()]['pdf'].value)
            hdu = fits.HDUList()
            hdu.append(fits.ImageHDU((self.pdf_dict[priorname.upper()]['pdf'].value).T, header=hdr, name="BKG"))
            #iiprint(self.pdf_dict[priorname.upper()]['pdf'])
            hdu.writeto("{}_bkg.fits".format(priorname.upper()), overwrite=True)
                
        if(len(self.pdf_dict[priorname.upper()]['pdf'].shape) == 1):            
                
            mbins = self.bins_midvals(priorname.upper())
            prior = self.pdf_dict[priorname.upper()]
            plt.plot(mbins[0], prior['pdf'])
            plt.title(priorname.upper())
            if filename is None:
                plt.show()
            else:
                plt.savefig("{}_bkg.png".format(priorname.upper()))
            plt.close()

    
    def _calc_pdf(self, cat, mags, magmin, magmax, magbinsize):

        if(mags is None):
            mags=list(cat.mags)
        if(magmin is None):
            magmin=['auto']*len(mags)
        if(magmax is None):
            magmax=['auto']*len(mags)
        if(magbinsize is None):
            magbinsize=['auto']*len(mags)

        field = cat.mags
        area = cat.area.to(u.arcsec**2)
        #print(field)
        #print(mags, magmin, magmax, magbinsize)
        prior_dict = {}  
        for iprior, col, mmin, mmax, mbin in zip(count(), mags, magmin, magmax, magbinsize):
            #print(iprior, col, mmin, mmax, mbin)
            edges=[]
            if(isinstance(col, list)):
            
                sample=np.ndarray([len(field[col[0]]), len(col)])
                
                for i, c, mn, mx, mb in zip(count(), col, mmin, mmax, mbin):
                    if( (mn is "auto")):
                        mn = int(min(field[c])-0.5)
                    if( (mx is "auto")):
                        mx = int(max(field[c])+0.5)
                    if( (mb is "auto")):
                        mb=0.5            
                    sample[:,i] = field[c]
                    edges.append(np.arange(mn, mx+mb/2.0, mb))

                prior_dict['PRIOR{}'.format(iprior)] = self._mag_hist(sample, area, edges, col)    
            else:
                if( (mmin is "auto")):
                    mmin = int(min(field[col])-0.5)
                if( (mmax is "auto")):
                    mmax = int(max(field[col])+0.5)
                if( (mbin is "auto")):
                    mbin=0.5            
                sample = field[c]
                edges.append(np.arange(mmin, mmax+mbin/2.0, mbin))
                prior_dict["PRIOR{}".format(iprior)]=self._mag_hist(sample, area, edges, col)
        #print(prior_dict)
        
        return prior_dict
        
    def _mag_hist(self, mags, area, edges, col):


        #bins, magrange = _define_magbins(magmin, magmax, magbinsize)
        counts, bins = np.histogramdd(mags, edges)
        vol=1.0
        for l in bins:
            vol=vol*(l[1:-1]-l[0:-2])[0]

        #print("VOLUME",vol)
        #print("EDGES", edges)
        
        maghist = {}
        maghist['edges'] = edges
        maghist['vol'] = vol
        maghist['pdf'] = counts / vol / area  ## in arcsec**2!!!
        maghist['name'] = col

        return maghist
    

def _define_magbins(magmin, magmax, magbinsize):
    if magbinsize == 'auto':
        bins = 'auto'
    else:
        nbins = 1 + (magmax - magmin)/magbinsize
        bins = np.linspace(magmin, magmax, num=nbins)

    limits = (magmin, magmax)
    
    return bins, limits
