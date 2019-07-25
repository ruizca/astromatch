"""
astromatch module for cross-matching astronomical 
catalogues using the ARCHES XMatch server.

Reference: Pineau et al. 2017

@author: A.Ruiz
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import zip, range
from io import open

try:
    # Python 2: "unicode" is built-in
    unicode
except NameError:
    unicode = str

import os
import getpass
import hashlib
import tempfile
import warnings
from base64 import b64encode
from shutil import rmtree
from copy import deepcopy

try:
    # python 3
    from contextlib import redirect_stdout
except:
    # python 2
    from contextlib2 import redirect_stdout

try:
    # python 3
    FileNotFoundError
except NameError:
    # python 2
    FileNotFoundError = IOError

import requests
from astropy import log
from astropy import units as u
from astropy.table import Table, Column, join, unique, vstack
from nwaylib.magnitudeweights import fitfunc_histogram
import numpy as np

from .priors import Prior
from .match import BaseMatch

# dictionary for storing xmatch user/passwd used during the session
_passwd_dict_xms = {}


class XMatchServerError(Exception):
    """Base class for exceptions in XMatchServer class."""
    pass


class XMatchServer(object):
    """
    Class for accessing the ARCHES X-Match tool   

    * The anonymous session last 30 minutes after the last operation: after
    this delay, the content of the remote directory is removed.
    * If more than 10 jobs are already executing, your job will be rejected and
    you will have to re-submit it later.
    * Your job will be automatically stopped after 10 minutes.
    * The tool is designed to process small jobs: to deal with large catalogues,
    it is the user responsability to slice the problem into smaller parts.
    * Idem if the job last more than 10 minutes: you must slice into smaller pieces.
    """
    _host = 'http://serendib.unistra.fr'
    url = os.path.join(_host, 'ARCHESWebService/XMatchARCHES')
    
    def __init__(self, user=None):
        """
        Start a session in the ARCHES X-Match tool.
        """
        if user is None:
            user = 'anonymous'

        self.session = self._start_session(user)
    
    def _start_session(self, user):
        if user != 'anonymous':
            try:
                passwd = _passwd_dict_xms[user]
            except KeyError:
                passwd = getpass.getpass("Password for {}:".format(user))
                passwd = self._encode_passwd(passwd)
        else:
            passwd = 'anonymous'

        data = {'cmd': 'login', 'username': user, 'password': passwd}

        session = requests.Session()
        response = session.post(self.url, data=data)
        self._check_status(response)
                
        if user != 'anonymous':
            _passwd_dict_xms[user] = passwd

        return session     

    def _encode_passwd(self, passwd):
        m = hashlib.sha1(passwd.encode())
        hexpasswd = m.hexdigest()

        return b64encode(hexpasswd.encode())

        
    def logout(self):
        """
        logout from X-Match server.
        """
        response = self.session.get(self.url, params=(('cmd', 'quit'),))        
        self._check_status(response)

    def logged(self):
        """
        Returns ``True`` if logged into the X-Match server, ``False`` otherwise.
        """
        response = self.session.get(self.url, params=(('cmd', 'islogged'),))

        try:
            self._check_status(response)

        except XMatchServerError:
            return False

        return True

    def ls(self):
        """
        Returns an Astropy ``Table`` listing the files hosted in the X-Match
        server. It contains four columns: creation date and time, size, and
        name of the file.
        """
        response = self.session.get(self.url, params=(('cmd', 'ls'),))
        self._check_status(response)

        ls_table = Table.read(response.text, format='ascii')
        for i, colname in enumerate(['date', 'time', 'size', 'name']):
            ls_table.rename_column('col{}'.format(i+1), colname)

        return ls_table

    def put(self, *args):
        """
        Upload files into the X-Match server.
        
        Parameters
        ----------
        args : path of the files to be uploaded.
        """
        for f in args:
            try:
                basename  = os.path.basename(f)
                files = {
                    'cmd': (None, 'put'),
                    basename: (basename, open(f, 'rb')),
                }
                response = self.session.post(self.url, files=files)

            except FileNotFoundError:
                log.error('File {} does not exist!'.format(f))
                raise XMatchServerError

            self._check_status(response)

    def get(self, *args, **kwargs):
        """
        Download files from the X-Match server.
        
        Parameters
        ----------
        args : files to be downloaded.
        output_dir : path of the directory where files are downloaded.
        """
        output_dir = kwargs.pop('output_dir', '.')

        for f in args:
            data = {
              'cmd': 'get',
              'fileName': f,
            }
            log.info('Downloading file "{}"'.format(f))
            response = self.session.post(self.url, data=data)

            if response.status_code != 200:
                log.error(response.text)
                raise XMatchServerError
            else:
                local_file = os.path.join(output_dir, f)
                with open(local_file, 'wb') as fp:
                    fp.write(response.content)

    def remove(self, *args):
        """
        Delete files in the X-Match server.
        
        Parameters
        ----------
        args : path of the files to be deleted.
        """
        for f in args:
            data = {
              'cmd': 'rm',
              'fileName': f,
            }
            response = self.session.post(self.url, data=data)
            self._check_status(response)

    def run(self, filename):
        """
        Execute the X-Match script contained in ``filename``.
        """
        files = {
            'cmd': (None, 'xmatch'),
            'script': (os.path.basename(filename), open(filename, 'rb')),
        }

        response = self.session.post(self.url, files=files)

        if self._find_run_error(response.text):
            log.error(response.text)
            raise XMatchServerError

        self._check_status(response)

    @staticmethod
    def _find_run_error(response):
        if 'java.lang.Exception' in response:
            error = True
        elif 'java.lang.Error' in response:
            error = True
        else:
            error = False

        return error

    @staticmethod
    def _check_status(response):
        if response.status_code != 200:
            log.error(response.text)
            raise XMatchServerError
        else:
            log.info(response.text)


class XMatch(BaseMatch):
    """
    Class for crossmatching catalogues using the XMATCH server.
    """

    _cutoff_column = 'p_single'

    ### Class Properties
    @property
    def results_raw(self):
        if self._match_raw is None:
            raise AttributeError('Match has not been performed yet!')
        else:
            return self._match_raw
            
    ### Public Methods
    def run(self, use_mags=False, xmatchserver_user=None, **kwargs):
        """
        Perform the cross-matching between the defined catalogues.
        
        Parameters
        ----------
        xmatchserver_user : ``str`` or ``None``
            user name for the X-Match server. If ``None``, anonymous access
            is used. Defaults to ``None``.
        use_mags : ``boolean``, optional
            apply corrections to the association probabilities based on the
            magnitudes of the counterpart sources. To this end, probability
            priors are constructed for each magnitude contained in the
            secondary catalogues. See ``Prior`` documentation for details on
            how these are calculated. Defaults to ``False``.
        mag_radius : ``Quantity``, optional
            Search radius around sources in the primary catalogue for
            building the magnitude priors. It must be an angular ``Quantity``.
            Defaults to 6 arcsec.
        prob_ratio_secondary : `float`, optional
            Minimum value of the probability ratio between two counterparts
            for the same primary source to be flagged as a secondary match.            
        """
        prob_ratio_secondary = kwargs.pop('prob_ratio_secondary', 0.5)
        mag_radius = kwargs.pop('mag_radius', 6.0*u.arcsec) 

        self._match_raw = self._xmatch(xmatchserver_user, **kwargs)

#        match_file = 'tmp_match.fits'
#        self._match_raw = Table.read(match_file)

        if use_mags:
            self._priors = self._calc_priors(mag_radius)
            
        match = self._final_table(self._match_raw, prob_ratio_secondary)

        return match

    def stats_rndmatch(self, match, match_rnd, ncutoff=101, plot_to_file=None):
        """
        Calculates match statistics (completness and reliability), using a
        random match, for a range of thresholds. This can be used later to 
        select the optimal threshold.
        """
        mask = match['match_flag'] == 1
        p_any0 = match[self._cutoff_column][mask]

        mask = match_rnd['match_flag'] == 1
        p_any0_offset = match_rnd[self._cutoff_column][mask]

        cutoffs = np.linspace(0, 1, num=ncutoff)

        stats = Table()
        stats['cutoff'] = cutoffs
        stats['completeness'] = [(p_any0 > c).mean() for c in cutoffs]
        stats['error_rate'] = [(p_any0_offset > c).mean() for c in cutoffs]
        stats['reliability'] = 1 - stats['error_rate']
        stats['CR'] = stats['completeness'] + stats['reliability']

        if plot_to_file is not None:
            self._plot_stats(stats, plot_to_file)

        return stats


    ### Internal Methods
    def _xmatch(self, xmatchserver_user, match_file='tmp_match.fits', **kwargs):
        # Use the XMatch server for the cross-matching
        xms = XMatchServer(user=xmatchserver_user)

        try:
            log.info('Uploading data to the XMatch server...')
            files_in_server = self._upload_catalogues(xms)

            log.info('Running cross-match and downloading results...')
            with tempfile.NamedTemporaryFile() as xms_file:
                make_xms_file(self.catalogues, xms_file.name, match_file, **kwargs)
                xms.run(xms_file.name)

            files_in_server.append(match_file)
            xms.get(match_file)

            log.info('Delete data from the server...')
            xms.remove(*files_in_server)
            xms.logout()

            match = Table.read(match_file)
            os.remove(match_file)

            return match

        except:
            xms.logout()
            raise

    def _upload_catalogues(self, xms):
        tmpdir = tempfile.mkdtemp()

        files_list = []
        for cat in self.catalogues:
            filename = os.path.join(tmpdir, 'tmp_{}.fits'.format(cat.name))
            cat.save(filename=filename, include_mags=False)
            files_list.append(filename)

        xms.put(*files_list)
        rmtree(tmpdir)

        files_in_server = [os.path.basename(f) for f in files_list]

        return files_in_server

    def _calc_priors(self, mag_radius):
        priors_dict = {}
        for cat in self.scats:
            prior = Prior(self.pcat, cat, radius=mag_radius)
            priors_dict[cat.name] = prior
            
        return priors_dict

    def _final_table(self, match_raw, prob_ratio_secondary):
        match = match_raw.copy()
        match['ncat'] = match['nPos'] #.rename_column('nPos', 'ncat')

        # Add distance between counterparts????

        log.info('Calculating final probabilities...')
        #match = self._calc_proba_null(match)
        #match = self._calc_dist_post_null(match)

        match = self._calc_dist_post(match)

        match = self._add_single_sources(match)

        match = self._calc_psingle(match)

        match = self._calc_pi(match)
        #match = self._calc_pany_pi(match)

        log.info('Flagging and sorting final results...')
        match = self._add_match_flags(match, prob_ratio_secondary)

        match = self._sort(match)

        match = self._clean_table(match)

        return match

    def _get_proba_dict(self, match):
        # Build a dictionary with the columns with posterior
        # probabilities and other related data
        proba_cols = np.array([col for col in match.colnames
                               if col.startswith('proba')])
        proba_cols_ncat = []
        proba_cols_catgroups = []
        for col in proba_cols:
            groups = col.split('_')[1:]
            proba_cols_ncat.append(len(''.join(groups)))
            proba_cols_catgroups.append(tuple(groups))

        proba_dict = {}
        proba_dict['cols'] = proba_cols
        proba_dict['ncat'] = np.array(proba_cols_ncat)
        proba_dict['groups'] = np.array(proba_cols_catgroups)

        return proba_dict

    def _calc_proba_null(self, match):
        # Estimate the probability that the primary catalague
        # has no match for each association
        pdict = self._get_proba_dict(match)

        match['proba_match'] = 0.0
        match['proba_null'] = 1.0

        for n in range(len(self.catalogues), 1, -1):
            mask = match['ncat'] == n
            nidx = np.where(pdict['ncat'] == n)

            for groups, col in zip(pdict['groups'][nidx], pdict['cols'][nidx]):
                for g in groups:
                    if self.pcat.name[0] in g and len(g) >= 2:
                        proba = match[col][mask]
                        proba[~np.isfinite(proba)] = 0

                        match['proba_match'][mask] += proba
                        match['proba_null'][mask] -= proba

        return match

    def _calc_dist_post_null(self, match):
        # Calculate the posterior probability that
        # the primary source has no counterpart, i.e.
        # the probability that all possible asssociations are
        # false, i.e. the product of the proba_null columns
        # for each primary source
        pidcol = 'SRCID_{}'.format(self.pcat.name)
        match = match.group_by(keys=pidcol)
        group_size = np.diff(match.groups.indices)

        dist_post_null = match['proba_null'].groups.aggregate(np.prod)
        match['dist_post_null'] = np.repeat(dist_post_null, group_size)

        return match

    def _calc_dist_post(self, match):
        pdict = self._get_proba_dict(match)
        sids_cols = ['SRCID_{}'.format(cat.name) for cat in self.scats]

        match['dist_post'] = 0.0
        split_assoc_tables = {}

        for n in range(len(self.catalogues), 1, -1):
            mask = match['ncat'] == n
            nidx = np.where(pdict['ncat'] == n)
            split_assoc_tables[n] = {}

            for groups, col in zip(pdict['groups'][nidx], pdict['cols'][nidx]):
                pcat_group = [g for g in groups if self.pcat.name[0] in g]
                if pcat_group:
                    pcat_group = pcat_group[0]

                if len(pcat_group) == n:
                    # Assign dist_post for the whole association
                    proba = match[col][mask]
                    proba[~np.isfinite(proba)] = 0

                    match['dist_post'][mask] += proba

                elif n > len(pcat_group) >= 2:
                    # Split partial association in new rows
                    split_table = match[mask]
                    split_table['ncat'] = len(pcat_group)
                    split_table['dist_post'] = split_table[col]

                    # Remove srcids from catalogues not included in the group
                    ids_cols_group = ['SRCID_{}'.format(cat.name)
                                      for cat in self.catalogues
                                      if cat.name[0] in pcat_group]

                    for srcid in sids_cols:
                        if srcid not in ids_cols_group:
                            split_table[srcid] = ''

                    mask_nan = ~np.isfinite(split_table['dist_post'])
                    split_table['dist_post'][mask_nan] = 0

                    try:
                        split_assoc_tables[n][pcat_group]['dist_post'] += split_table['dist_post']
                    except KeyError:
                        split_assoc_tables[n][pcat_group] = split_table

                    split_assoc_tables[n][pcat_group + '_idcols'] = ids_cols_group

        # remove repeated tuples (keep highest probability)
        split_tables_list = []
        for n_dict in split_assoc_tables.values():
            for key, group_table in n_dict.items():
                if 'idcols' not in key:
                    group_table.sort(['dist_post'])
                    group_table.reverse()
                    group_table = unique(group_table, keys=n_dict[key + '_idcols'])
                    
                    split_tables_list.append(group_table)            
       
        match = vstack([match] + split_tables_list)

        return match

    def _add_single_sources(self, match):
        # Add rows with singles sources in the primary catalogue with associations
        # sources with no associations are already included
        pid = 'SRCID_{}'.format(self.pcat.name)
        mask = match['ncat'] > 1
        match_single = unique(match[mask], keys=pid)

        match_single['ncat'] = 1
        match_single['chi2Pos'] = 0.0
        match_single['dist_post'] = 0.0#match_single['dist_post_null']

        sids_cols = ['SRCID_{}'.format(cat.name) for cat in self.scats]
        for srcid in sids_cols:
            match_single[srcid] = ''

        return vstack([match, match_single])

    def _calc_psingle(self, match):
        # dist_post modified by mag bias
        #match['p_single'] = 1 / (1 + (1 - match['prior'])/match['post_weight'])
        if self._priors:
            match = self._add_magbias(match)

        else:
            match['p_single'] = match['dist_post']
            
        return match

#    def _calc_match_prior0(self, match):
#        # Prior of the null hypothesis (no real association)
#        # is the overall identification ratio:
#        # number of matches divided by the total number 
#        # of sources in the primary catalogue.
#        # NWAY uses the completeness estimation provided by the user.
#
#        # Find sources with counterparts
#        mask = match['ncat'] > 1
#        idkey = 'SRCID_{}'.format(self.pcat.name)
#        nmatches = len(unique(match[mask], keys=idkey))
#
#        # TODO: may be this could be corrected taking into account
#        # the completeness value given to xmatch 
#        # (not related with the nway completeness)
#        
#        return 1.0 * nmatches / len(self.pcat)
#
#    def _calc_match_priors(self, match):
#        nscats = len(self.scats)
#        area_total = (4*np.pi * u.rad**2).to(u.deg**2) #( (4 * pi * (180 / pi)**2)
#        prior0 = self._calc_match_prior0(match)
#        
#        match_priors = {}
#        for n in range(nscats):
#            for c in combinations(self.scats, n+1):
#                prior_name = self.pcat.name[0] + ''.join(cat.name[0] for cat in c)
#                match_priors[prior_name] = 1.0                
#
#                for cat in c:
#                    completeness = prior0**(1.0/nscats)
#                    K = completeness * cat.area / area_total
#                    match_priors[prior_name] *=  K.value / (len(cat) + 1)
#
#        return match_priors, prior0

    def _add_magbias(self, match):        
        total_bias = np.zeros(len(match))
        match['SIDX'] = np.arange(len(match))

        for cat in self.scats:
            match_idcol = 'SRCID_{}'.format(cat.name)
            prior = self._priors[cat.name].prior_dict

            match_withcat = unique(match, keys=match_idcol)
            match_withcat = [row[match_idcol] for row in match_withcat
                             if not row[match_idcol].isspace()]
            cat_withmatch = cat.select_by_id(match_withcat)

            for magcol in cat.mags.colnames:
                # using nway implementation for this
                hist = prior[magcol]
                func = fitfunc_histogram(hist['bins'], hist['good'], hist['field'])

                weights = np.log10(func(cat_withmatch.mags[magcol]))
                weights[np.isnan(weights)] = 0 # undefined magnitudes do not contribute
                
                magbias = Table()
                magbias[match_idcol] = cat_withmatch.ids
                magbias_col = 'log_bias_{}'.format(magcol)
                magbias[magbias_col] = weights
                
                match = join(match, magbias, keys=match_idcol, join_type='left')
                match[magbias_col][match[magbias_col].mask] = 0.0
                match.sort('SIDX')

                total_bias += match[magbias_col]

        #match['post_weight'] = match['post_weight'] * 10**total_bias
        match['p_single'] = 1/(1 + (1/match['dist_post'] - 1) / 10**total_bias)

        mask = np.logical_and(total_bias == 0.0, match['dist_post'] == 0.0)
        match['p_single'][mask] = 0.0

        mask = np.isneginf(total_bias)
        match['p_single'][mask] = 0.0

        match.remove_column('SIDX')
        
        return match

    def _calc_pi(self, match):
        ## Estimate relative probabilities for different
        ## matchs of a single primary source
        # Group match by primary sources
        pidcol = 'SRCID_{}'.format(self.pcat.name)
        match = match.group_by(pidcol)
        group_size = np.diff(match.groups.indices)

        # For each primary source, find the sum of posterior probabilities
        psum = match['p_single'].groups.aggregate(np.sum)

        # The previous array has a length equal to the number of groups.
        # We need to rebuild the array having the same length as the
        # original table for using element-wise operations. We repeat
        # each element of psum as many times as the size of the
        # corresponding group
        psum = np.repeat(psum, group_size)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            match['prob_this_match'] = match['p_single']/psum

        mask = match['ncat'] == 1
        match['prob_this_match'][mask] = 0 # Set p_i to zero for no match rows

        return match

    def _calc_pany_pi(self, match):
        # Group match by primary sources
        pidcol = 'SRCID_{}'.format(self.pcat.name)
        match = match.group_by(pidcol)
        group_size = np.diff(match.groups.indices)

        ## For each primary source, find the sum of posterior probabilities
        psum = match['dist_post'].groups.aggregate(np.sum)

        # The previous array has a length equal to the number of groups.
        # We need to rebuild the array having the same length as the
        # original table for using element-wise operations. We repeat
        # each element of psum as many times as the size of the
        # corresponding group
        psum = np.repeat(psum, group_size)
        psum_not_null = psum - match['dist_post_null']

        match['renorm_dist_post'] = match['dist_post']/psum

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            match['prob_has_match'] = 1 - match['dist_post_null']/psum
            match['prob_this_match'] = match['dist_post']/psum_not_null

        mask = match['ncat'] == 1
        match['prob_this_match'][mask] = 0 # Set p_i to zero for no match rows

        mask = np.logical_and(mask, match['dist_post'] == 0)
        match['prob_has_match'][mask] = 0

        return match

    def _add_match_flags(self, match, prob_ratio_secondary):
        ## Add match_flag column, default value to zero
        #idx_flag = match.colnames.index('prob_has_match')
        idx_flag = match.colnames.index('prob_this_match')
        col_flag = Column(name='match_flag', data=[0]*len(match))
        match.add_column(col_flag, index=idx_flag)

        pcat_idcol = 'SRCID_{}'.format(self.pcat.name)
        match = match.group_by(pcat_idcol)
        group_size = np.diff(match.groups.indices)

        ## For each primary source, find the match with maximum p_i
        pi_max = match['prob_this_match'].groups.aggregate(np.max)

        # The previous array has a length equal to the number of groups.
        # We need to rebuild the array having the same length as the
        # original table for using element-wise operations. We repeat
        # each element of sumlr as many times as the size of the
        # corresponding group
        pi_max = np.repeat(pi_max, group_size)

        mask = match['prob_this_match'] == pi_max
        match['match_flag'][mask] = 1

        ## Find secondary matches
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mask2 = match['prob_this_match']/pi_max > prob_ratio_secondary
            mask = np.logical_and(~mask, mask2)

        match['match_flag'][mask] = 2

        return match

    def _sort(self, match):
        # Sort final table as the primary catalogue.
        # We need to pass the column in pcat with the source ids.
        return self._sort_as_pcat(match, 'SRCID_{}'.format(self.pcat.name))

    def _clean_table(self, match):
        # Move chi2Pos column
        match.rename_column('chi2Pos', 'chi2Pos_old')
        
        idx_flag = match.colnames.index('ncat')
        col_flag = Column(name='chi2Pos', data=match['chi2Pos_old'].data)
        match.add_column(col_flag, index=idx_flag)

        # Remove columns
        match.remove_columns(['posRA', 'posDec', 'ePosA', 'ePosB', 'ePosPA',
                              'nPos', 'ePosEpoch', 'chi2Pos_old'])#, 'dist_post_null'])

        for col in match.colnames:
            if col.startswith('proba_'):
                match.remove_column(col)

        # Remove duplicate single primary source entries
        # (it happens when crossmatching more than 3 catalogues)
        mask = match['ncat'] > 1
        for cat in self.scats:
            idcol = 'SRCID_{}'.format(cat.name)
            mask = np.logical_and(mask, np.char.strip(match[idcol]) == b'')

        return match[~mask]

    def _match_rndcat(self, xmatchserver_user=None, **kwargs):
        # Cross-match secondary catalogue with a randomized 
        # version of the primary catalogue
        original_pcat = deepcopy(self.pcat)
        self.pcat = self.pcat.randomise(numrepeat=1)
        self.catalogues[0] = self.pcat

        prob_ratio_secondary = kwargs.pop('prob_ratio_secondary', 0.5)

        # Hide std ouput of xmatch
        with redirect_stdout(open(os.devnull, 'w')):
            match_file = 'tmp_match_rnd.fits'
            match_rnd_raw = self._xmatch(xmatchserver_user, match_file, **kwargs)

        match_rnd = self.final_table(match_rnd_raw, prob_ratio_secondary)

        # Recover original pcat
        self.pcat = original_pcat
        self.catalogues[0] = original_pcat

        return match_rnd

    @staticmethod
    def _reliability(data):
        return data['p_single']


def make_xms_file(catalogues, xms_file, match_file, **kwargs):
    """
    Prepare an X-Match script file (xms) with the data needed for a cross-match.
    """
    completeness = kwargs.pop('completeness', 0.9973)
    method = kwargs.pop('xmatch_method', 'probaN_v1')

    # We assume all catalogues cover the same sky area
    area = catalogues[0].area.to(u.rad**2).value 
    joins = 'M'

    mainId = 'SRCID_{}'.format(catalogues[0].name)
    ids = ','.join('SRCID_{}'.format(cat.name) for cat in catalogues)
    proba_letters = ','.join('{}'.format(cat.name[0]) for cat in catalogues)

    xms_cmd = 'gset proba_letters={}\n'.format(proba_letters)
    for i, cat in enumerate(catalogues):    
        xms_cat = _xms_cat_properties(cat)
        xms_cmd = ''.join([xms_cmd, xms_cat])

        if i > 1:
            joins += 'M'

    # We add the keep=largestnid option because with this output is easier
    # to estimate later proba_null and dist_post_null. After calculating that,
    # we have to split the associations in their corresponding subassociations
    # to obtain an output similar to nway.
    xms_foot = ('\nxmatch {} joins={} completeness={} area={} '
                '? meth=median keep=largestnid mainId={} ids={}\n'
                'save {} fits')
    xms_foot = xms_foot.format(method, joins, completeness,
                               area, mainId, ids, match_file)

    xms_cmd = ''.join([xms_cmd, xms_foot])

    with open(xms_file, 'w') as fp:
        fp.write(unicode(xms_cmd))

def _xms_cat_properties(cat):
    xms_cat = ('\nget FileLoader file=tmp_{0}.fits\n'
               'set pos ra=RA dec=DEC\n'
               'set poserr type={1} {2}\n'
               'set cols SRCID_{0}\n')

    poserr_type = cat.poserr_type.upper()
    poserr_cols = cat.poserr.components.colnames
    poserr_pars = ' '.join('param{:d}={}'.format(i+1, p)
                           for i, p in enumerate(poserr_cols))
    xms_cat = xms_cat.format(cat.name, poserr_type, poserr_pars)

    return xms_cat
