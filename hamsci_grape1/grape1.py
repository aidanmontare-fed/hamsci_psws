"""
grape1.py

Code for analyzing and visualizing HamSCI Grape1 Personal Space Weather Station
Observations.

Contributions by:
    Nathaniel Frissell W2NAF
    Kristina Collins KD8OXT
    Aidan Montare KB3UMD
    David Kazdan AD8Y
    John Gibbons N8OBJ
    Bob Benedict KD8CGH

    July 2022
"""

import os
import sys
import glob
import string
letters = string.ascii_lowercase

import datetime
import pytz

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
tqdm.pandas(dynamic_ncols=True)

from scipy.interpolate import interp1d
from scipy import signal

from . import locator
from . import solar
from . import gen_lib as gl

prm_dict = {}

pkey = 'Freq'
prm_dict[pkey] = {}
prm_dict[pkey]['label'] = 'Doppler Shift [Hz]'

pkey = 'Vpk'
prm_dict[pkey] = {}
prm_dict[pkey]['label'] = 'Peak Voltage [V]'

pkey = 'Power_dB'
prm_dict[pkey] = {}
prm_dict[pkey]['label'] = 'Received Power [dB]'

pkey = 'SLT'
prm_dict[pkey] = {}
prm_dict[pkey]['label'] = 'Solar Mean Time'

class DataInventory(object):
    def __init__(self,data_path='data',suffix='.csv'):
        """
        Create an inventory of availble grape1 data in the data_path.
        Inventory will be dataframe df attached to DataInventory object.

        data_path: location of grape1 data
        suffix:    suffix of data files. Defaults to '.csv'
	"""
        # Load filenames and create a dataframe.
        fpaths = glob.glob(os.path.join(data_path,'*'+suffix))
        bnames = [os.path.basename(fpath) for fpath in fpaths]
        df = pd.DataFrame({'Filename':bnames})

        # Parse filenames into useful information and place into dataframe.
        df2              = df['Filename'].str.split('_', expand=True)
        df2.columns      =['Datetime', 'Node', 'G', 'Grid Square', 'FRQ', 'Frequency']
        df2              = df2.drop(columns=['FRQ']) # no information in this columnn
        df2["Frequency"] = df2["Frequency"].str.strip(suffix)

        # # Parse Grid Squares
        # lat,lon = locator.gridsquare2latlon(df2['Grid Square'])
        # df2['Latitude']  = lat
        # df2['Longitude'] = lon

        df               = pd.concat([df2, df], axis = 1) # concatenate dataframes horizontally
        df['Datetime']   = pd.to_datetime(df['Datetime']) # cast to datetime
        df['Node']       = df['Node'].str.strip("N")                   # Ditch the leading N on the node numbers
        df['Node']       = df['Node'].astype(str).astype(int)          # Cast node number to int
        df               = df[~df['Frequency'].str.contains('G1')]     # discarding files with naming errors

        # Convert frequency abbreviations to numbers:
        df.loc[df['Frequency'] == 'WWV5',    'Frequency'] = 5e6
        df.loc[df['Frequency'] == 'WWV10',   'Frequency'] = 10e6
        df.loc[df['Frequency'] == 'WWV2p5',  'Frequency'] = 2.5e6
        df.loc[df['Frequency'] == 'WWV15',   'Frequency'] = 15e6
        df.loc[df['Frequency'] == 'CHU3',    'Frequency'] = 3330e3
        df.loc[df['Frequency'] == 'CHU7',    'Frequency'] = 7850e3
        df.loc[df['Frequency'] == 'CHU14',   'Frequency'] = 14.67e6
        df.loc[df['Frequency'] == 'Unknown', 'Frequency'] = 0
        
        # Save dataframe to object
        self.df = df
        
        # List of logged nodes during the period of interest, for sorting:
        logged_nodes = df["Node"].unique().tolist()
        logged_nodes.sort()
        
        self.logged_nodes = logged_nodes
    
    def plot_inventory(self,html_out='inventory.html'):
        """
        Generate a plot of the data inventory using plotly.

        html_out: Filename of html-version of data inventory plot.
        """
        # We can create a Data Inventory (Gantt chart) showing when different stations were active. 
        invt         = self.df.copy()
        logged_nodes = self.logged_nodes
        
        invt.set_index('Node')
        invt = invt.drop(columns=['G', 'Grid Square'])
        invt['EndTime'] = invt['Datetime']+ datetime.timedelta(days=1) # create an end
        invt['Filename'].str.strip('FRQ_')

        fig = px.timeline(invt, x_start="Datetime", x_end="EndTime", y="Node", color="Frequency", category_orders={"Node": logged_nodes})
        fig.update_yaxes(type='category')
        fig.update_annotations(text = "Filename", clicktoshow='on')
        fig.show()

        if html_out:
            fig.write_html(html_out, include_plotlyjs="cdn")

class GrapeNodes(object):
    def __init__(self,fpath='nodelist.csv',logged_nodes=None):
        """
        Create an object with information about known Grape1 nodes from a nodelist.csv.

        fpath:        path to nodelist.csv
        logged_nodes: list of nodes we have data for. This can be obtained from DataInventory.logged_nodes.
        """
        # Read in node list:
        nodes_df = pd.read_csv(fpath)
        nodes_df = nodes_df.rename(columns={'Node #': 'Node_Number'})
        nodes_df = nodes_df.set_index('Node_Number')
        
        self.nodes_df = nodes_df

        
        if logged_nodes:
            nodes_df = self.update_status(logged_nodes)
        
    def update_status(self,logged_nodes):
        """
        Updates the nodes_df dataframe with with information regarding which nodes we have data from.

        logged_nodes: list of nodes we have data for. This can be obtained from DataInventory.logged_nodes.
        """
        nodes_df = self.nodes_df
        
        nodes_df['Status'] = np.where((nodes_df.index.isin(logged_nodes)), "Data logged", "No data logged")
        self.nodes_df = nodes_df
        return nodes_df
    
    def status_table(self):
        """
        Returns the nodes_df data frame with nodes that we have data for highlighted in green.
        """
        # Highlight rows in green for stations that we have data from.
        nodes_df = self.nodes_df.copy()
        color    = (nodes_df['Status'] == 'Data logged').map({True: 'background-color: palegreen', False: ''})
        nodes_df = nodes_df.style.apply(lambda s: color)
        return nodes_df
    
    def plot_map(self,color="Status",projection='albers usa',
            width=1200,height=900):
        """
        Use plotly to generate a map of Grape1 nodes.
        """
        nodes_df = self.nodes_df
        # Map nodes:
        fig = px.scatter_geo(nodes_df, "Latitude", "Longitude",
                             color=color, # which column to use to set the color of markers
                             hover_name=nodes_df["Callsign"], # column added to hover information
                             projection = projection,
                             width = width, height = height,
                             )
        fig.show()

class Filter(object):
    def __init__(self,N=6,Tc_min = 3.3333,btype='low',fs=1.):
        """
        Generate a digital filter that can be applied to the data.
        This routine uses the scipy.signal.butter Butterworth filter.

        N:      Filter order.
        Tc_min: Cuttoff in minutes. Scalar value for 'low' and 'high'
                filter btypes; 2-element iterable for 'bandpass' and 
                'bandstop' filter btypes.
        fs:     Sampling frequency of the digital system in samples/sec.
        """
        Wn     = (1./(np.array(Tc_min)*60.))
        
        if np.size(Wn) == 2:
            Wn = Wn[::-1]
            
        #Wn    = (1000, 5000)    # 3 dB Cutoff Frequency in Hz
               # Choose 'bandpass' or 'bandstop'

        b, a = signal.butter(N, Wn, btype, fs=fs)
        
        self.fs = fs
        self.Wn = Wn
        self.a  = a
        self.b  = b
        
    def filter_data(self,data):
        """
        Apply the filter using scipy.signal.filtfilt to data.

        data: Vector of data to be filtered.
        """
        return signal.filtfilt(self.b,self.a,data)
    
    def plotResponse(self):
        """
        Plot the magnitude and phase response of the filter.
        """
        fs = self.fs
        Wn = self.Wn
        a  = self.a
        b  = self.b
        
        w, h = signal.freqz(b, a,worN=2**16)
        f = (fs/2)*(w/(np.pi))        
        
        plt.figure(figsize=(12,8))
        plt.subplot(211)
        plt.plot(f, 20 * np.log10(abs(h)))
        plt.xscale('log')
        plt.title('Butterworth Filter Frequency Response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.grid(which='both', axis='both')
        if np.size(Wn) == 1:
            plt.axvline(Wn, color='green')    # cutoff frequency
        else:            
            plt.axvline(Wn[0], color='green') # cutoff frequency
            plt.axvline(Wn[1], color='green') # cutoff frequency

        plt.ylim(-30,0)

        plt.subplot(212)
        plt.plot(f, np.unwrap(np.angle(h)))
        plt.xscale('log')
        plt.title('Butterworth Filter Frequency Response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [rad]')
        plt.grid(which='both', axis='both')
        if np.size(Wn) == 1:
            plt.axvline(Wn, color='green')    # cutoff frequency
        else:            
            plt.axvline(Wn[0], color='green') # cutoff frequency
            plt.axvline(Wn[1], color='green') # cutoff frequency

        plt.tight_layout()

class Grape1Data(object):
    def __init__(self,node=None,freq=None,sTime=None,eTime=None,data_path='data',
                 lat=None,lon=None,call_sign=None,
                 solar_lat=None,solar_lon=None,
                 inventory=None,grape_nodes=None,
                 data=None,meta=None):

        if data is None and meta is None:
            self.__load_raw(node,freq,sTime,eTime=eTime,data_path=data_path,
                     lat=lat,lon=lon,call_sign=call_sign,solar_lat=solar_lat,solar_lon=solar_lon,
                     inventory=inventory,grape_nodes=grape_nodes)
        else:
            self.data = data
            self.meta = meta
        
    def __load_raw(self,node,freq,sTime,eTime,data_path,
                 lat,lon,call_sign,solar_lat,solar_lon,inventory,grape_nodes):
        
        if inventory is None:
            inventory = DataInventory(data_path=data_path)
        
        # Make temporary copy of dataframe.
        dft   = inventory.df.copy()

        # Select rows with selected frequency
        tf    = dft['Frequency'] == freq
        dft   = dft[tf].copy()
        
        # Select rows matching node number.
        tf    = dft['Node'] == node
        dft   = dft[tf].copy()

        if sTime is None:
            sTime = min(dft['Datetime'])

        if eTime is None:
            eTime = max(dft['Datetime'])        
        
        # Select rows matching time range.
        tf    = np.logical_and(dft['Datetime'] >= sTime, dft['Datetime'] < eTime)
        dft   = dft[tf].copy()
        dft   = dft.sort_values('Datetime')

        # Load data from every data file available for node/frequency/date range.
        df_raw = []
        for rinx,row in tqdm(dft.iterrows(),total=len(dft),dynamic_ncols=True,desc='Loading Raw Data'):
            fname = row['Filename']
            fpath = os.path.join(data_path,fname)
#            print(' --> {!s}'.format(fname))

            df_load = pd.read_csv(fpath, comment = '#', parse_dates=[0])
            if len(df_load) == 0: continue

            # Remove the center frequency offset from the frequency column.
            df_load['Freq'] = df_load['Freq']-freq
            df_raw.append(df_load)

        df_raw  = pd.concat(df_raw,ignore_index=True)
        df_raw  = df_raw.sort_values('UTC')
     
        # Generate a label for each Node
        if (call_sign is None) and (grape_nodes is not None):
            call_sign = grape_nodes.nodes_df.loc[node,'Callsign']
        else:
            call_sign = ''
        
        label     = '{:0.1f} MHz N{:07d} {!s}'.format(freq/1e6,node,call_sign)

        # Get latitude and longitude
        if (lat is None) and (grape_nodes is not None):
            lat = grape_nodes.nodes_df.loc[node,'Latitude']
        
        if (lon is None) and (grape_nodes is not None):
            lon = grape_nodes.nodes_df.loc[node,'Longitude']

        if solar_lat is None:
            solar_lat = lat

        if solar_lon is None:
            solar_lon = lon

        # Store data into dictionaries.
        data = {}
        data['raw']          = {}
        data['raw']['df']    = df_raw
        data['raw']['label'] = 'Raw Data'
        self.data = data
        
        meta = {}
        meta['label']  = label
        meta['lat']    = lat
        meta['lon']    = lon
        meta['solar_lat']    = solar_lat
        meta['solar_lon']    = solar_lon

        self.meta      = meta

    def process_data(self,profile='standard'):
        tic_0 = datetime.datetime.now()
        print('Processing data using "{!s}" profile...'.format(profile))
        print('')
        if profile == 'standard':
            print('Computing Solar Local Time on raw data...')
            tic = datetime.datetime.now()
            self.calculate_solar_time('raw')
            toc = datetime.datetime.now()
            print('  Solar Time Computation Time: {!s}'.format(toc-tic))
        
            resample_rate = datetime.timedelta(seconds=1)
            filter_order  = 6
            Tc_min        = 3.3333
            btype         = 'low'

            print('Resampling data with {!s} second cadence...'.format(resample_rate.total_seconds()))
            tic = datetime.datetime.now()
            self.resample_data(resample_rate=resample_rate,
                              data_set_in='raw',data_set_out='resampled')
            toc = datetime.datetime.now()
            print('  Resampling Time: {!s}'.format(toc-tic))

            print('Computing Solar Local Time on resampled...')
            tic = datetime.datetime.now()
            self.calculate_solar_time('resampled')
            toc = datetime.datetime.now()
            print('  Solar Time Computation Time: {!s}'.format(toc-tic))
            
            # Convert Vpk to Power_dB
            print('dB Conversion')
            tic = datetime.datetime.now()
            self.data['resampled']['df']['Power_dB'] = 20*np.log10( self.data['resampled']['df']['Vpk'])
            toc = datetime.datetime.now()
            print('  dB Conversion Time: {!s}'.format(toc-tic))
            
            print('Filtering data with {!s} minute low-pass Butterworth filter...'.format(Tc_min))
            tic = datetime.datetime.now()
            self.filter_data(N=filter_order,Tc_min=Tc_min,btype=btype)
            toc = datetime.datetime.now()
            print('  Filtering Time: {!s}'.format(toc-tic))

        elif profile == '5min_mean':
            resample_rate = datetime.timedelta(minutes=5)
            print('Resampling data with {!s} minute cadence...'.format(resample_rate.total_seconds()/60.))
            tic = datetime.datetime.now()
            self.resample_data(resample_rate=resample_rate,method='mean',
                              data_set_in='raw',data_set_out='resampled')
            toc = datetime.datetime.now()
            print('  Resampling Time: {!s}'.format(toc-tic))

            print('Computing Solar Local Time on resampled...')
            tic = datetime.datetime.now()
            self.calculate_solar_time('resampled')
            toc = datetime.datetime.now()
            print('  Solar Time Computation Time: {!s}'.format(toc-tic))

            # Convert Vpk to Power_dB
            print('dB Conversion')
            tic = datetime.datetime.now()
            self.data['resampled']['df']['Power_dB'] = 20*np.log10( self.data['resampled']['df']['Vpk'])
            toc = datetime.datetime.now()
            print('  dB Conversion Time: {!s}'.format(toc-tic))

#            print('Compute Time-Date-Parameter (TDP) Matrix')
#            tic = datetime.datetime.now()
#            self.calculate_timeDateParameter_matrix('resampled','Freq')
#            toc = datetime.datetime.now()
#            print('  dB Conversion Time: {!s}'.format(toc-tic))
            
        toc_0 = datetime.datetime.now()
        print('')
        print('Total Processing Time: {!s}'.format(toc_0-tic_0))
    
    def resample_data(self,resample_rate,on='UTC',method='linear_interp',
                          data_set_in='raw',data_set_out='resampled'):
        
        df   = self.data[data_set_in]['df'].copy()
        
        # Create the list of datetimes that we want to resample to.
        # Find the start and end times of the array.
        sTime = df['UTC'].min()
        eTime = df['UTC'].max()

        tzinfo= sTime.tzinfo

        # Break
        sYr  = sTime.year
        sMon = sTime.month
        sDy  = sTime.day
        sHr  = sTime.hour
        sMin = sTime.minute
        sSec = sTime.second
        resample_sTime = datetime.datetime(sYr,sMon,sDy,sHr,sMin,sSec,tzinfo=tzinfo)

        eYr  = eTime.year
        eMon = eTime.month
        eDy  = eTime.day
        eHr  = eTime.hour
        eMin = eTime.minute
        eSec = eTime.second
        resample_eTime = datetime.datetime(eYr,eMon,eDy,eHr,eMin,eSec,tzinfo=tzinfo)

        # Remove SLT column if it exists because it cannot be resampled.
        if 'SLT' in df.keys():
            df = df.drop('SLT',axis=1)

        cols        = df.keys()
        df          = df.set_index(on) # Need to make UTC column index for interpolation to work.

        rs_df       = df.resample(resample_rate,origin=resample_sTime)
        if method == 'mean': 
            rs_df = rs_df.mean()
        else:
            rs_df = rs_df.interpolate(method='linear')

        rs_df       = rs_df.copy()
        rs_df[on]   = rs_df.index
        rs_df.index = np.arange(len(rs_df)) 

        # Put Columns back in original order.
        rs_df       = rs_df[cols].copy()
        
        tmp          = {}
        tmp['df']    = rs_df
        tmp['label'] = 'Resampled Data (dt = {!s} s)'.format(resample_rate.total_seconds())
        tmp['Ts']    = resample_rate.total_seconds()
        self.data[data_set_out] = tmp

    def filter_data(self,N,Tc_min,btype,
                    data_set_in='resampled',data_set_out='filtered',
                    params=['Freq','Vpk','Power_dB']):
        
        df     = self.data[data_set_in]['df'].copy()

        # Get sample rate of data set.
        Ts     = self.data[data_set_in].get('Ts')
        if Ts is None:
            Ts_arr = np.unique(np.diff(df['UTC']))
            if len(Ts_arr) != 1:
                raise Exception("{!s} is not evenly sampled. Cannot apply filter.".format(data_set_in))
            Ts   = (Ts_arr[0]).total_seconds()

        # Convert sample rate to sampling frequency.
        fs   = 1./Ts

        filt = Filter(N=N,Tc_min=Tc_min,btype=btype,fs=fs)
        
        for param in params:
            if param not in df.keys():
                continue
            df[param] = filt.filter_data(df[param])
        
        tmp          = {}
        tmp['df']    = df
        tmp['label'] = 'Butterworth Filtered Data\n(N={!s}, Tc={!s} min, Type: {!s})'.format(N, Tc_min, btype)
        self.data[data_set_out] = tmp                            

    def calculate_solar_time(self,data_set,solar_lon=None):
        if solar_lon is None:
            solar_lon = self.meta.get('solar_lon')
        else:
            self.meta['solar_lon'] = solar_lon

        df = self.data.get(data_set)['df']
        df['SLT'] = df['UTC'].progress_apply(solar.solar_time,lon=solar_lon)

        # Set columns so UTC and SLT lead.
        keys = list(df.keys())
        keys.remove('UTC')
        keys.remove('SLT')
        keys = ['UTC','SLT'] + keys
        df   = df[keys]
        self.data[data_set]['df'] = df

    def show_datasets(self):
        keys        = []
        datasets    = []
        for key,dct in self.data.items():
            keys.append(key)

            tmp  = {}
            tmp['label'] = dct.get('label')

            datasets.append(tmp)
        
        datasets = pd.DataFrame(datasets,index=keys)
        return datasets
        
    def plot_timeSeries(self,data_sets=['raw'],
                        sTime=None,eTime=None,
                        xkey='UTC',params=['Freq','Vpk','Power_dB'],
                        ylims = {},
                        plot_kws = [{'ls':'','marker':'.'},{}],
                        overlayTerminator=True,
                        solar_lat=None,solar_lon=None,
                        fig_width=15,panel_height=6):
        
        data_sets = gl.get_iterable(data_sets)
        
        df      = self.data[data_sets[0]]['df']
        if sTime is None:
            sTime = min(df[xkey])
            
        if eTime is None:
            eTime = max(df[xkey])
        
        # Make sure we have the requested parameters.
        params = gl.get_iterable(params)
        params_good = []
        for param in params:
            if param in df.keys():
                params_good.append(param)
        params = params_good
        
        # Start plotting
        ncols   = 1
        nrows   = len(params)
        figsize = (fig_width, nrows*panel_height)
        
        fig = plt.figure(figsize=figsize)       
        axs = []
        for plt_inx,param in enumerate(params):
            ax  = fig.add_subplot(nrows,ncols,plt_inx+1)
            axs.append(ax)

            if overlayTerminator:
                solar_lat = self.meta.get('solar_lat',solar_lat)
                solar_lon = self.meta.get('solar_lon',solar_lon)
                if (solar_lat is not None) and (solar_lon is not None):
                    solar.add_terminator(sTime,eTime,solar_lat,solar_lon,ax,xkey=xkey)
                else:
                    print('Error: Need to provide solar_lat and solar_lon to overlay solar terminator.')

            ax.set_xlim(sTime,eTime)
            if plt_inx != nrows-1:
                ax.set_xticklabels('')
            else:
                fig.autofmt_xdate()
                xprmd  = prm_dict.get(xkey,{})
                xlabel = xprmd.get('label',xkey)
                ax.set_xlabel(xlabel)

            yprmd  = prm_dict.get(param,{})
            ylabel = yprmd.get('label',yprmd)
            ax.set_ylabel(ylabel)
            ylim = ylims.get(param,None)
            if ylim is not None:
                ax.set_ylim(ylim)

            if plt_inx == 0:
                ax.set_title(self.meta.get('label',''))
            ax.set_title('({!s})'.format(letters[plt_inx]),loc='left')
           
        for ds_inx,data_set in enumerate(data_sets):
            for plt_inx,param in enumerate(params):
                ax = axs[plt_inx]
                
                df  = self.data[data_set]['df']
                xx  = df[xkey]
                yy  = df[param]
                
                label   = self.data[data_set].get('label','')
                plot_kw = plot_kws[ds_inx]
                ax.plot(xx,yy,label=label,**plot_kw)

        for plt_inx,param in enumerate(params):
            ax = axs[plt_inx]
            ax.legend(loc='upper right',fontsize='small')
        
        fig.tight_layout()
        
        return {'fig':fig}
