#!/usr/bin/env python3
import os
import datetime
import pytz

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import pickle

from hamsci_psws import gen_lib as gl
from hamsci_psws import grape1
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
    path  = os.path.join('output','multiplot')
    gl.make_dir(path,clear=True)

    node            = 7
    freq            = 10e6 # MHz

    sTime           = datetime.datetime(2021,10,28, tzinfo=pytz.UTC)
    eTime           = sTime + datetime.timedelta(days=1)

    inventory       = grape1.DataInventory()
    inventory.filter(freq=freq,sTime=sTime,eTime=eTime)
    grape_nodes     = grape1.GrapeNodes(logged_nodes=inventory.logged_nodes)

    node_nrs    = inventory.get_nodes()

    gds = []
    for node in node_nrs:
        gd = grape1.Grape1Data(node,freq,sTime,eTime,inventory=inventory,grape_nodes=grape_nodes)
        gd.process_data()
        gds.append(gd)

    mp          = grape1.GrapeMultiplot(gds)

    # WWV Coordinates
    solar_lat   =   40.6683
    solar_lon   = -105.0384

    color_dct   = {'ckey':'lon'}
    xkeys       = ['LMT','UTC']
    for xkey in xkeys:
        print('Plotting: {!s}'.format(xkey))

        ret     = mp.multiplot('filtered',color_dct=color_dct,xkey=xkey,
                solar_lat=solar_lat,solar_lon=solar_lon)
        fig     = ret['fig']
        fname   = 'multiplot_{!s}.png'.format(xkey)
        fpath   = os.path.join(path,fname)
        fig.savefig(fpath,bbox_inches='tight')
    import ipdb; ipdb.set_trace()
