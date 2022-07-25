#!/usr/bin/env python3
import os
import datetime
import pytz

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import pickle

from hamsci_grape1 import gen_lib as gl
from hamsci_grape1 import grape1
prm_dict = grape1.prm_dict


mpl.rcParams['font.size']        = 16
mpl.rcParams['font.weight']      = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.grid']        = True
mpl.rcParams['grid.linestyle']   = ':'
mpl.rcParams['figure.figsize']   = np.array([15, 8])
mpl.rcParams['axes.xmargin']     = 0

if __name__ == '__main__':
    path  = 'output'
    gl.make_dir(path,clear=True)

    node            = 7
    freq            = 10e6 # MHz

#    sTime           = datetime.datetime(2021,10,28, tzinfo=pytz.UTC)
#    eTime           = sTime + datetime.timedelta(days=3)

    sTime           = None
    eTime           = None

    cache_file      = 'gd.p'
    recalc_cache    = False

    inventory = grape1.DataInventory()
    nodes     = grape1.GrapeNodes(logged_nodes=inventory.logged_nodes)

    if not os.path.exists(cache_file) or recalc_cache:
        print('Loading fresh data...')
        gd = grape1.Grape1Data(node,freq,sTime=sTime,eTime=eTime,
                        inventory=inventory,grape_nodes=nodes)
        gd.process_data('5min_mean')

        data_meta = {'data':gd.data,'meta':gd.meta}

        with open(cache_file,'wb') as cfl:
            pickle.dump(data_meta,cfl)
    else:
        print('Using cache file: {!s}'.format(cache_file))
        tic = datetime.datetime.now() 
        with open(cache_file,'rb') as cfl:
            data_meta= pickle.load(cfl)
        gd = grape1.Grape1Data(**data_meta)
        toc = datetime.datetime.now()
        print(' Cache Loading Time: {!s}'.format(toc-tic))

    df = gd.data['resampled']['df']

    xkeys  = ['SLT','UTC']
    params = ['Freq','Power_dB']

    for xkey in xkeys:
        for param in params:
            print('Processing: {!s} - {!s}'.format(xkey,param))

            print('Compute Time-Date-Parameter (TDP) Array')
            tic = datetime.datetime.now()
            tdp = gd.calculate_timeDateParameter_array('resampled',param,xkey=xkey)
            toc = datetime.datetime.now()
            print('  Time-Date-Parameter Time: {!s}'.format(toc-tic))

            fig = plt.figure(figsize=(15,10))
            ax  = fig.add_subplot(111)

            xr_xkey = '{!s}_Date'.format(xkey)
            xprmd   = prm_dict.get(xr_xkey)
            xlabel  = xprmd.get('label')

            xr_ykey = '{!s}_Hour'.format(xkey)
            yprmd   = prm_dict.get(xr_ykey)
            ylabel  = yprmd.get('label')

            prmd = prm_dict.get(param)
            vmin = prmd.get('vmin')
            vmax = prmd.get('vmax')
            cmap = prmd.get('cmap')
            plbl = prmd.get('label')

            cbar_kwargs = {}
            cbar_kwargs['label'] = plbl

            ret  = tdp.plot.pcolormesh(xr_xkey,xr_ykey,vmin=vmin,vmax=vmax,cmap=cmap,
                    cbar_kwargs=cbar_kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            fig.tight_layout()
            fname = 'tdp_{!s}_{!s}.png'.format(param,xkey)
            fpath = os.path.join(path,fname)
            fig.savefig(fpath,bbox_inches='tight')

    import ipdb; ipdb.set_trace()
