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

    xkeys  = ['LMT','UTC']
    for xkey in xkeys:
        print('Plotting: {!s}'.format(xkey))
        ret     = gd.plot_timeDateParameter_array('resampled',xkey=xkey)
        fig     = ret['fig']
        fname   = 'tdp_{!s}.png'.format(xkey)
        fpath   = os.path.join(path,fname)
        fig.savefig(fpath,bbox_inches='tight')

    import ipdb; ipdb.set_trace()
