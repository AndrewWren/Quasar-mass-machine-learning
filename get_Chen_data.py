#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from time import sleep
import warnings

from astropy.cosmology import Planck15 as cosmo
from astropy.table import Table
import astropy.units as u
from progressbar import progressbar

DATA_PATH = os.path.join(os.path.dirname (__file__), 'data')
BIGGER_DATA_PATH = os.path.join(DATA_PATH, 'bigger_data')
SPECTRA_PATH = os.path.join(DATA_PATH, 'spectra')
CHEN_DATA_FILE = os.path.join(DATA_PATH, 'apjsab41fet1_mrt.txt')
SDSS_DATA_FILE = os.path.join(BIGGER_DATA_PATH, 'DR16Q_v4.fits')
QUASARS_FILE = os.path.join(BIGGER_DATA_PATH, 'quasars.parquet')
SPECTRA_FILE = os.path.join(BIGGER_DATA_PATH, 'spectra.parquet')
COL_DICT_LIST = [('SDSS', (1, 18)), ('Plate', (20, 24)),
                ('MJD',   (26, 30)),  ('Fiber', (32, 35)),
                ('ZVI',   (37, 42)),
                ('logMBH-Hb',     (91, 94)),
                ('logMBH-Ha-2',   (108, 111)),
                ('logMBH-MgII-2', (113, 117)),
                ('logMBH-CIV-2',  (119, 123))]
Z_MIN = 1.2
Z_MAX = 1.8
INSIST_LOWER = 3.16
INSIST_UPPER = 3.53 
D_SCALE = 4
SCALE = 10**-D_SCALE
RANDOM_STATE = 42


# In[4]:


class Epoch:
    """A class taking data on plate, mjd and fiber which identifies an epoch of 
    SDSS observation."""
    
    def __init__(self, *pmf):
        """pmf: three arguments or a list-like of three arguments - 
        arguments/list-like elements above those three are ignored.
        """
        if len(pmf) == 1:
            pmf = pmf[0]
        self.plate = pmf[0]
        self.mjd   = pmf[1]
        self.fiber = pmf[2]
        
    def __eq__(self, other):
        return ((self.plate == other.plate) & (self.mjd == other.mjd) & 
                (self.fiber == other.fiber))
    
    def __hash__(self):
        """Needed for Epoch to work with some Pandas features - especially
        to create the multiindex for spectra."""
        return hash((self.plate, self.mjd, self.fiber))
    
    def __get_state__(self):
        return (self.plate, self.mjd, self.fiber)
    
    def __set_state__(self, state):
        self.plate = state[0]
        self.mjd   = state[1]
        self.fiber = state[2]
    
    def name(self):
        return(str(self.plate).zfill(4) + '-' + str(self.mjd).zfill(5)
               + '-' + str(self.fiber).zfill(4))
    
    def web_address(self):
        return(str(self.plate).zfill(4) + '/spec-' + self.name() + '.fits\n')
    
    def file_location(self):
        return(os.path.join(str(self.plate).zfill(4),
                            'spec-' + self.name() + '.fits'))
    

def epoch_names(epochs):
    return [epoch.name() for epoch in epochs]


def epoch_addresses(epochs):  
    return [epoch.web_address() for epoch in epochs]


def epoch_locations(epochs):  
    return [epoch.file_location() for epoch in epochs]


# In[5]:


def create_quasars():
    
    def get_quasars_longlist():
        pref_LMBH = ['logMBH-Hb', 'logMBH-Ha-2',
                     'logMBH-MgII-2', 'logMBH-CIV-2']

        def select_LMBH(row):
            for i in pref_LMBH:
                if row.loc[i] != 0.:
                    return row.loc[i]

        col_dict = dict([(elt[0], (elt[1][0]-1, elt[1][1])) for _, elt 
                                             in enumerate(COL_DICT_LIST)])
        quasars  = pd.read_fwf(CHEN_DATA_FILE, skiprows=102, 
                                   colspecs=list(col_dict.values()), 
                                   header=None, names=list(col_dict.keys()))
        quasars['SDSS'] = quasars['SDSS'].apply(lambda x: x.encode())

        quasars.loc[:, 'LMBH'] = quasars.apply(
                                    lambda row: select_LMBH(row), axis=1)
        quasars['CHEN EPOCH'] = quasars.loc[:, 'Plate':'Fiber'].apply(
                                            lambda pmf: Epoch(pmf), axis=1)
        quasars = quasars.drop(['Plate','MJD', 'Fiber'], axis=1)
        quasars = quasars.dropna()
        quasars = quasars.rename(columns={'SDSS': 'OBJECT'})
        quasars = quasars.set_index('OBJECT')

        assert quasars.isnull().sum().sum() == 0, 'Null entries in quasars'

        return quasars


    sdss_table = Table.read(SDSS_DATA_FILE, format='fits')
    sdss_table['SDSS_NAME'].name = 'OBJECT'
    #print(sdss_table)
    sdss_names = sdss_table['OBJECT']
    quasars = get_quasars_longlist()
    quasars_names = quasars.index
    names_both, sdss_both, quasars_both = np.intersect1d(sdss_names,
                                                         quasars_names,
                                                         return_indices=True)
    #print('1\n', quasars.head())
    #[sdss_names.shape, quasars_names.shape, names_both.shape]
    
    print('Converting SDSS Table epoch list to Pandas format')
    epoch_list = list()
    for i, elt in enumerate(sdss_table):
        #if i >= 10:
         #   break
        #print(i, end=' ')
        Chen_epoch = [Epoch(elt['PLATE'], elt['MJD'], elt['FIBERID'])]
        #print(Chen_epoch)
        n_dup_epochs = np.argmax(elt['PLATE_DUPLICATE']==-1)
        if n_dup_epochs > 0:       
            epoch_dup = [Epoch(pmf)
                         for pmf in np.transpose([elt['PLATE_DUPLICATE']
                         [:n_dup_epochs],
                         elt['MJD_DUPLICATE'][:n_dup_epochs],
                         elt['FIBERID_DUPLICATE'][:n_dup_epochs]])]
            epoch_list.append(np.concatenate((Chen_epoch, epoch_dup)))
        else:
            epoch_list.append(Chen_epoch)
        
    #print(epoch_list)
    epoch_prob_list = list()
    sdss_table['LMBH'] = [-1.] * len(sdss_table)
    sdss_table['PROB'] = [False] * len(sdss_table)

    #print('Checking that Chen spectrum appears in SDSS primary or duplicate spectra')
    Chen_epoch_col_no = quasars.columns.get_loc('CHEN EPOCH')
    #print(f'Chen_epoch_col_no={Chen_epoch_col_no}')
    mass_col_no = quasars.columns.get_loc('LMBH')
    for j, (sdss_i, mass_i) in enumerate(zip(sdss_both, quasars_both)):
        #print(f'{j}/{sdss_i}/{mass_i}', end= ' ')
        assert sdss_table[sdss_i]['OBJECT'].encode() == \
            quasars.index[mass_i], (j, sdss_i, mass_i)
        Chen_epoch = quasars.iloc[mass_i, Chen_epoch_col_no]
        if Chen_epoch in epoch_list[sdss_i]:
            sdss_table['LMBH'][sdss_i] = quasars.iloc[mass_i, mass_col_no]
        else:
            epoch_prob_list.append((j, sdss_i, mass_i))
            sdss_table['PROB'][sdss_i] = True

    sdss_table['EPOCH_LIST'] = epoch_list
    names = [name for name in sdss_table.colnames
                     if len(sdss_table[name].shape) <= 1]
    sdss = sdss_table[names].to_pandas(index='OBJECT')
    sdss = sdss[(sdss['ZWARNING'] == 0) & (sdss['LMBH'] > 0.) 
                & (sdss['Z'] >= Z_MIN) & (sdss['Z'] < Z_MAX)
                & (sdss['PROB'] == False)]
    
    valid_object_indices = quasars.index.intersection(sdss.index)
    quasars = quasars.loc[valid_object_indices]
    assert (np.all(np.sort(quasars.index) 
                    == np.sort(sdss.index.get_level_values('OBJECT')))
                    ), 'quasars and sdss have different object sets'
    
    quasars['Z'] = sdss.loc[valid_object_indices, 'Z']
    quasars['EPOCH_LIST'] = sdss.loc[valid_object_indices, 'EPOCH_LIST']
    quasars['N_EPOCHS'] = quasars['EPOCH_LIST'].apply(len)
    
    z_diff = quasars['ZVI'] - quasars['Z']
    quasars = quasars[z_diff<=3 * np.std(z_diff)]
    #print('2\n', quasars.head())
    return quasars



# In[6]:


"""Note that downloading the actual FITS spectra via Python is around 8 times slower than just using the wget in Ubuntu,
so AFTER using the function below go to Ubuntu and issue the following command lines from the main directory you have downloaded this repo to:
cd ./data/spectra; wget -nv -r -nH --cut-dirs=8 -i Chen_spec_list.txt -B https://data.sdss.org/sas/dr16/eboss/spectro/redux/v5_13_0/spectra/lite/

Ignore error messages on Ubuntu if the download works
"""

def write_address_list(quasars):
    epochs = np.concatenate(list(quasars['EPOCH_LIST']))
    print()
    address_list = epoch_addresses(epochs)
    location = os.path.join(SPECTRA_PATH, "Chen_spec_list.txt")

    print(f'Downloading {epochs.shape[0]} spectrum web addresses to {location}'
         )

    with open(location,"w+") as f:
        for address in address_list:
            f.write(address)


# In[7]:


# [10**INSIST_LOWER, 10**INSIST_UPPER]  #Includes the C-IV and Mg-II lines


# In[8]:

def create_spectra(quasars, insist_lower=INSIST_LOWER,
                   insist_upper=INSIST_UPPER):
    """ Input:  sdss, the pandas DataFrame of selected SDSS objects
    
                insist_lower, insist_upper, so only select restframe spectra
                which include those log wavelengths, discarding those that do not.
        
        Output: quasars, a pandas DataFrame, which is a "reduced" version of 
                the input quasars, clipped to match the spectra output (which
                discarded some spectra due to the insists and possibly due
                to there being no spectrum to retrieve.
                
                spectra, a pandas DataFrame of restframe spectra all truncated
                to a common range of restframe log wavelengths.
                The dataframe has two indices, 'OBJECT' and 'EPOCH'.
    """
    
    
    def spec_obj_epoch(file_location, z_subtract, z_factor):   
        filename = os.path.join(SPECTRA_PATH, file_location)

        try:
            with warnings.catch_warnings(): #to avoid:  WARNING: hdu= was not specified but multiple tables are present, reading in first available table (hdu=1) [astropy.io.fits.connect]
                warnings.simplefilter("ignore")
                spec_table = Table.read(filename, format='fits')
            spec = spec_table.to_pandas()

            spec['RESTLOGLAM'] = spec.loc[:,'LOGLAM'].apply(lambda x: 
                                                            x - z_subtract)
            spec['FLUX'] = spec.loc[:,'FLUX'] * z_factor

            return spec['RESTLOGLAM'].to_numpy(), spec['FLUX'].to_numpy()
        except FileNotFoundError:
            return ['No spec'], ['No spec']
    
    
    def finalise_quasars(quasars, spectra):
        print('Finalising quasars to match spectra used')
        valid_object_indices = np.unique(np.array(quasars.index.intersection(
                                    spectra.index.get_level_values('OBJECT'))))
        quasars = quasars.loc[valid_object_indices]
        quasars['EPOCH_LIST'] = [spectra.loc[obj].index.get_level_values
                                 ('EPOCH') for obj in quasars.index]
        quasars['N_EPOCHS'] = quasars['EPOCH_LIST'].apply(len)
        assert quasars['N_EPOCHS'].sum() == spectra.shape[0], f"""Number of 
            epochs in quasars ({quasars['N_EPOCHS'].sum()}) different from
            number in spectra ({spectra.shape[0]})"""
        return quasars
    
    
    print('Creating spectra')
    spectra_list = list()
    indices = list()
    lw_upper = np.inf
    lw_lower = - np.inf
   
    count = 0
    no_spec_count = 0
    insist_fail_count = 0
    print('Going through quasars to generate emitted spectra')
    sleep(0.4)  #Stop the prints affecting the progressbar
    for obj in progressbar(quasars.index):
        z = quasars.loc[obj, 'Z']
        z_subtract = np.log10(1+z)
        z_factor = (cosmo.luminosity_distance(z)/u.Mpc)**2 * 1e-9
        # z_factor is in Units of 1e-8 erg s^-1 A^-1 cm^-2 Mpc^2, where the A^-1 are observed not emitted (the A_em^-1 have cancelled out)
        
        epochs = quasars.loc[obj, 'EPOCH_LIST']
        #if i < 1:
        #print(epoch_pmfs)
        
        for epoch_no, epoch in enumerate(epochs):    
            count += 1
            rest_wavelength, flux = spec_obj_epoch(epoch.file_location(),
                                                   z_subtract, z_factor)
            if rest_wavelength[0] == 'No spec':
                no_spec_count += 1
                continue
            if (rest_wavelength[0] > insist_lower) or (rest_wavelength[-1]
                                                       < insist_upper):
                insist_fail_count += 1
                continue
            
            lw_upper = min(lw_upper, rest_wavelength[-1])
            lw_lower = max(lw_lower, rest_wavelength[0])
            
            spectra_list.append([rest_wavelength, flux])
            indices.append([obj, epoch])
    
    print(f"""count={count}, no_spec_count={no_spec_count}, 
                insist_fail_count={insist_fail_count}""")
    print(f"""len(spectra_list)={len(spectra_list)}, len(indices)=
                    {len(indices)}""")
    
    lw_lower = round(np.ceil(lw_lower/SCALE)*SCALE, D_SCALE)
    lw_upper = round(np.floor(lw_upper/SCALE)*SCALE, D_SCALE)
    x = np.arange(lw_lower, lw_upper, SCALE/2)
 
   
    #print('Point 1')
    print('Transforming emitted spectra to standard wavelength grid')
    sleep(0.4)  #Stop the prints affecting the progressbar
    ys = list()
    for rest_wavelength, flux in progressbar(spectra_list):
        inter = interp1d(rest_wavelength, flux)
        ys.append(inter(x))

    #print('Point 2')
    print('Creating spectra dataframe')
    mi = pd.MultiIndex.from_tuples(indices, names=['OBJECT', 'EPOCH'])
    #print('Point 3')
    spectra = pd.DataFrame(ys, index=mi, columns=x, dtype='float64')
    assert spectra.isnull().sum().sum() == 0,                 "Error: spectra has a null element"

    return finalise_quasars(quasars, spectra), spectra

# In[81]:


def get_Chen_data(creating='Load'):
 
    def save_quasars(quasars):
        quasars_new = quasars.copy().drop(columns=['EPOCH_LIST'])
        index = [idx for idx in quasars_new.index]
        quasars_new.index = index
        quasars_new.index.name = 'OBJECT'            
        quasars_new['CHEN EPOCH'] = (quasars_new['CHEN EPOCH']
                                    .apply(lambda epoch: epoch.name()))
        quasars_new.to_parquet(path=QUASARS_FILE, engine='fastparquet',
                               compression=None)

    def save_spectra(spectra):
        mi = spectra.index
        mi = pd.MultiIndex.from_tuples([(index[0],
                                        index[1].name()) for index in mi],
                                        names=mi.names)#['OBJECT', 'EPOCH'])
        spectra_new = pd.DataFrame(spectra.copy().to_numpy(),
                                   index=mi,
                                   columns=[str(col) for col in
                                            spectra.columns])
        spectra_new.to_parquet(path=SPECTRA_FILE,
                               engine='fastparquet', compression=None)    
        
    def Xy(quasars, spectra):

            
        def objs_train_test_val(quasars, spectra):
            strata_bins = [0.5, 1.5, 2.5, 3.5, 60, np.inf]
            strata = np.digitize(quasars.loc[:, 'N_EPOCHS'], bins=strata_bins)
            obj_full_train, obj_test = train_test_split(quasars.index,
                                        test_size=0.1, stratify=strata, 
                                        random_state=RANDOM_STATE)

            strata = np.digitize(quasars.loc[obj_full_train, 'N_EPOCHS'],
            bins=strata_bins)
            obj_train, obj_val = train_test_split(obj_full_train,
                                                  test_size=0.2/0.9,
                                                  stratify=strata,
                                                  random_state=RANDOM_STATE+1)
            return obj_train, obj_val, obj_test


        def Xy_objs(quasars, spectra, objs):
            index = [(obj, epoch) for obj, epoch in spectra.index
                     if obj in objs]
            X_objs = spectra.loc[index]
            y_objs = pd.Series([quasars.loc[obj, 'LMBH'] for obj 
                                in spectra.index.get_level_values('OBJECT')
                                if obj in objs], index=index, name='LMBH')
            return X_objs, y_objs

        print('Forming X and y for train, val, test')
        obj_train, obj_val, obj_test = objs_train_test_val(quasars, spectra)

        X_train, y_train = Xy_objs(quasars, spectra, obj_train)
        X_val, y_val     = Xy_objs(quasars, spectra, obj_val)
        X_test, y_test   = Xy_objs(quasars, spectra, obj_test)

        print('Completed.')
        return quasars, spectra, X_train, X_val, X_test, y_train, y_val, y_test


    """Begin main get_Chen_data function"""

    if creating == 'Download':
        quasars = create_quasars()
        write_address_list(quasars)
        print("""Note that downloading the actual FITS spectra via Python is 
        around 8 times slower than just using the wget in Ubuntu,
        so AFTER using the function below go to Ubuntu and issue the
        following command line from the main directory to which the repo was
        downloaded:""")
        print('cd ./data/spectra; wget -nv -r -nH --cut-dirs=8 -i ', end='')
        print('Chen_spec_list.txt -B ', end='')
        print('https://data.sdss.org/sas/dr16/eboss/spectro/redux/', end='')
        print('v5_13_0/spectra/lite/')
        print('Ignore error messages on Ubuntu if the download works')
        return ['Null'] * 8 # 'Return' so no error from the function call
    elif creating == 'Create dataframes':
        quasars = create_quasars()
        #pkl.dump([quasars], open(os.path.join(DATA_PATH,
         #                                           "quas.p"),"wb"))
        quasars, spectra = create_spectra(quasars)
        save_quasars(quasars)
        save_spectra(spectra)
        #pkl.dump([quasars, spectra], open(ML_DATA_FILE,"wb"))       
    else:
        print('Loading quasars and spectra')
        quasars = pd.read_parquet(path=QUASARS_FILE)

        """To avoid UserWarning: Non-categorical multi-index is likely brittle
        warnings.warn("Non-categorical multi-index is likely brittle")
        which means might fail with new versions:
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spectra = pd.read_parquet(path=SPECTRA_FILE)
        #quasars, spectra = pkl.load(open(ML_DATA_FILE,"rb"))
        #print('3\n', quasars.head())

    return Xy(quasars, spectra)





