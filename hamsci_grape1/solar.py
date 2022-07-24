import datetime
import pytz
from suntime import Sun

def solar_time(datetime_utc,lon):
    slt = datetime_utc + datetime.timedelta(hours=(lon/15.))
    slt = slt.replace(tzinfo=None)
    return slt

def utc_time(datetime_slt,lon):
    utc = datetime_slt - datetime.timedelta(hours=(lon/15.))
    utc = utc.replace(tzinfo=pytz.UTC)
    return utc

def add_terminator(sTime,eTime,lat,lon,ax,color='0.7',alpha=0.3,xkey='UTC',**kw_args):
    """
    Shade the nighttime region on a time series plot.

    sTime:   UTC start time of time series plot in datetime.datetime format
    eTime:   UTC end time of time series plot in datetime.datetime format
    lat:     latitude for solar terminator calculation
    lon:     longitude for solar terminator calculation
    ax:      matplotlib axis object to apply shading to.
    color:   color of nighttime shading
    alpha:   alpha of nighttime shading
    kw_args: additional keywords passed to ax.axvspan
    """
    if xkey == 'SLT':
        sTime = utc_time(sTime,lon)
        eTime = utc_time(eTime,lon)

    sun   = Sun(float(lat), float(lon))
    sDate = datetime.datetime(sTime.year,sTime.month,sTime.day)
    eDate = datetime.datetime(eTime.year,eTime.month,eTime.day)

    dates = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1] + datetime.timedelta(days=1))

    sunSetRise = []
    for date in dates:    
        ss = sun.get_sunset_time(date - datetime.timedelta(days=1))
        sr = sun.get_sunrise_time(date)
        sunSetRise.append( (sr, ss) )

    for ss,sr in sunSetRise:                
        if xkey == 'SLT':
            ss = solar_time(ss,lon)
            sr = solar_time(sr,lon)
        ax.axvspan(ss,sr,color=color,alpha=alpha,**kw_args) 

