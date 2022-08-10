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

mpl.rcParams['font.size']        = 18
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

    sTime           = datetime.datetime(2021,10,28,14, tzinfo=pytz.UTC)
    eTime           = datetime.datetime(2021,10,28,18, tzinfo=pytz.UTC)

    inventory       = grape1.DataInventory()
    inventory.filter(freq=freq,sTime=sTime,eTime=eTime)
    grape_nodes     = grape1.GrapeNodes(logged_nodes=inventory.logged_nodes)

    node_nrs    = inventory.get_nodes()

    gds = []
    for node in node_nrs:
#        if node not in [13]:
#            continue

        # Aidan Montare's station not working right, so skip it.
        if node in [9,10]:
            continue

        gd = grape1.Grape1Data(node,freq,sTime,eTime,inventory=inventory,grape_nodes=grape_nodes)
        gd.process_data()
        gds.append(gd)

        df = gd.data['filtered']['df']

#        import ipdb; ipdb.set_trace()

    mp          = grape1.GrapeMultiplot(gds)

    # WWV Coordinates
    solar_lat   =   40.6683
    solar_lon   = -105.0384

    color_dct   = {'ckey':'lon'}
#    xkeys       = ['LMT','UTC']
    xkeys       = ['UTC']

    events      = []
    evt = {}
    evt['datetime'] = datetime.datetime(2021,10,28,15,35)
    evt['label']    = 'X1'
    events.append(evt)

    evt = {}
    evt['datetime'] = datetime.datetime(2021,10,28,17,38)
    evt['label']    = 'C4.9'
    events.append(evt)

    for xkey in xkeys:
        print('Plotting: {!s}'.format(xkey))

        if xkey == 'UTC':
            events_plot = events
        else:
            events_plot = None

        ret     = mp.multiplot('filtered',color_dct=color_dct,xkey=xkey,panel_height=6,
                events=events_plot,solar_lat=solar_lat,solar_lon=solar_lon,plot_GOES=True)
        fig     = ret['fig']
        fname   = 'multiplot_{!s}.png'.format(xkey)
        fpath   = os.path.join(path,fname)
        fig.savefig(fpath,bbox_inches='tight')
    import ipdb; ipdb.set_trace()
