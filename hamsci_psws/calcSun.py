# -*- coding: utf-8 -*-
"""The calcSun module

This subpackage contains methods for various solar calculations

Note
----
Source: http://www.esrl.noaa.gov/gmd/grad/solcalc/
Translated to Python by Sebastien de Larquier

"""

from __future__ import annotations

import datetime
import numpy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def calcTimeJulianCent( jd : float ) -> float:
    """Convert Julian Day to centuries since J2000.0.
    """
    T = (jd - 2451545.0)/36525.0
    return T


def calcGeomMeanLongSun( t : float ) -> float:
    """Calculate the Geometric Mean Longitude of the Sun (in degrees)
    """
    L0 = 280.46646 + t * ( 36000.76983 + t*0.0003032 )
    while L0 > 360.0:
        L0 -= 360.0
    while L0 < 0.0:
        L0 += 360.0
    return L0 # in degrees


def calcGeomMeanAnomalySun( t : float ) -> float:
    """Calculate the Geometric Mean Anomaly of the Sun (in degrees)
    """
    M = 357.52911 + t * ( 35999.05029 - 0.0001537 * t)
    return M # in degrees


def calcEccentricityEarthOrbit( t : float ) -> float:
    """Calculate the eccentricity of earth's orbit (unitless)
    """
    e = 0.016708634 - t * ( 0.000042037 + 0.0000001267 * t)
    return e # unitless


def calcSunEqOfCenter( t : float ) -> float:
    """Calculate the equation of center for the sun (in degrees)
    """
    mrad = numpy.radians(calcGeomMeanAnomalySun(t))
    sinm = numpy.sin(mrad)
    sin2m = numpy.sin(mrad+mrad)
    sin3m = numpy.sin(mrad+mrad+mrad)
    C = sinm * (1.914602 - t * (0.004817 + 0.000014 * t)) + sin2m * (0.019993 - 0.000101 * t) + sin3m * 0.000289
    return C # in degrees


def calcSunTrueLong( t : float ) -> float:
    """Calculate the true longitude of the sun (in degrees)
    """
    l0 = calcGeomMeanLongSun(t)
    c = calcSunEqOfCenter(t)
    O = l0 + c
    return O # in degrees


def calcSunTrueAnomaly( t : float) -> float:
    """Calculate the true anamoly of the sun (in degrees)
    """
    m = calcGeomMeanAnomalySun(t)
    c = calcSunEqOfCenter(t)
    v = m + c
    return v # in degrees


def calcSunRadVector( t : float) -> float:
    """Calculate the distance to the sun in AU (in degrees)
    """
    v = calcSunTrueAnomaly(t)
    e = calcEccentricityEarthOrbit(t)
    R = (1.000001018 * (1. - e * e)) / ( 1. + e * numpy.cos( numpy.radians(v) ) )
    return R # n AUs


def calcSunApparentLong( t : float) -> float:
    """Calculate the apparent longitude of the sun (in degrees)
    """
    o = calcSunTrueLong(t)
    omega = 125.04 - 1934.136 * t
    SunLong = o - 0.00569 - 0.00478 * numpy.sin(numpy.radians(omega))
    return SunLong # in degrees


def calcMeanObliquityOfEcliptic( t : float ) -> float:
    """Calculate the mean obliquity of the ecliptic (in degrees)
    """
    seconds = 21.448 - t*(46.8150 + t*(0.00059 - t*(0.001813)))
    e0 = 23.0 + (26.0 + (seconds/60.0))/60.0
    return e0 # in degrees


def calcObliquityCorrection( t : float) -> float:
    """Calculate the corrected obliquity of the ecliptic (in degrees)
    """
    e0 = calcMeanObliquityOfEcliptic(t)
    omega = 125.04 - 1934.136 * t
    e = e0 + 0.00256 * numpy.cos(numpy.radians(omega))
    return e # in degrees


def calcSunRtAscension( t : float) -> float:
    """Calculate the right ascension of the sun (in degrees)
    """
    e = calcObliquityCorrection(t)
    SunLong = calcSunApparentLong(t)
    tananum = ( numpy.cos(numpy.radians(e)) * numpy.sin(numpy.radians(SunLong)) )
    tanadenom = numpy.cos(numpy.radians(SunLong))
    alpha = numpy.degrees(numpy.arctan2(tananum, tanadenom))
    return alpha # in degrees


def calcSunDeclination( t : float) -> float:
    """Calculate the declination of the sun (in degrees)
    """
    e = calcObliquityCorrection(t)
    SunLong = calcSunApparentLong(t)
    sint = numpy.sin(numpy.radians(e)) * numpy.sin(numpy.radians(SunLong))
    theta = numpy.degrees(numpy.arcsin(sint))
    return theta # in degrees


def calcEquationOfTime( t : float) -> float:
    """Calculate the difference between true solar time and mean solar time (output: equation of time in minutes of time)  
    """
    epsilon = calcObliquityCorrection(t)
    l0 = calcGeomMeanLongSun(t)
    e = calcEccentricityEarthOrbit(t)
    m = calcGeomMeanAnomalySun(t)
    y = numpy.tan(numpy.radians(epsilon/2.0))
    y *= y

    sin2l0 = numpy.sin(numpy.radians(2.0 * l0))
    sinm   = numpy.sin(numpy.radians(m))
    cos2l0 = numpy.cos(numpy.radians(2.0 * l0))
    sin4l0 = numpy.sin(numpy.radians(4.0 * l0))
    sin2m  = numpy.sin(numpy.radians(2.0 * m))

    Etime = y * sin2l0 - 2.0 * e * sinm + 4.0 * e * y * sinm * cos2l0 - 0.5 * y * y * sin4l0 - 1.25 * e * e * sin2m
    return numpy.degrees(Etime*4.0) # in minutes of time


def calcHourAngleSunrise( lat : float, solarDec : float) -> float:
    """Calculate the hour angle of the sun at sunrise for the latitude (in radians)
    """
    latRad = numpy.radians(lat)
    sdRad  = numpy.radians(solarDec)
    HAarg = numpy.cos(numpy.radians(90.833)) / ( numpy.cos(latRad)*numpy.cos(sdRad) ) - numpy.tan(latRad) * numpy.tan(sdRad)
    HA = numpy.arccos(HAarg)
    return HA # in radians (for sunset, use -HA)


def calcAzEl( t: float, localtime: float, latitude: float, longitude: float, zone: float ) -> tuple[float, float]:
    """Calculate sun azimuth and zenith angle
    """
    eqTime = calcEquationOfTime(t)
    theta  = calcSunDeclination(t)

    solarTimeFix = eqTime + 4.0 * longitude - 60.0 * zone
    earthRadVec = calcSunRadVector(t)

    trueSolarTime = localtime + solarTimeFix
    while trueSolarTime > 1440:
        trueSolarTime -= 1440.

    hourAngle = trueSolarTime / 4.0 - 180.0
    if hourAngle < -180.: 
        hourAngle += 360.0

    haRad = numpy.radians(hourAngle)
    csz = numpy.sin(numpy.radians(latitude)) * numpy.sin(numpy.radians(theta)) + numpy.cos(numpy.radians(latitude)) * numpy.cos(numpy.radians(theta)) * numpy.cos(haRad)
    if csz > 1.0: 
        csz = 1.0 
    elif csz < -1.0: 
        csz = -1.0
    zenith = numpy.degrees(numpy.arccos(csz))
    azDenom = numpy.cos(numpy.radians(latitude)) * numpy.sin(numpy.radians(zenith))
    if abs(azDenom) > 0.001: 
        azRad = (( numpy.sin(numpy.radians(latitude)) * numpy.cos(numpy.radians(zenith)) ) - numpy.sin(numpy.radians(theta))) / azDenom
        if abs(azRad) > 1.0: 
            if azRad < 0.: 
                azRad = -1.0 
            else:
                azRad = 1.0
        
        azimuth = 180.0 - numpy.degrees(numpy.arccos(azRad))
        if hourAngle > 0.0: 
            azimuth = -azimuth
    else:
        if latitude > 0.0: 
            azimuth = 180.0 
        else:
            azimuth = 0.0
    if azimuth < 0.0: 
        azimuth += 360.0
    exoatmElevation = 90.0 - zenith

    # Atmospheric Refraction correction
    if exoatmElevation > 85.0: 
        refractionCorrection = 0.0
    else:
        te = numpy.tan(numpy.radians(exoatmElevation))
        if exoatmElevation > 5.0: 
            refractionCorrection = 58.1 / te - 0.07 / (te*te*te) + 0.000086 / (te*te*te*te*te) 
        elif exoatmElevation > -0.575: 
            refractionCorrection = 1735.0 + exoatmElevation * (-518.2 + exoatmElevation * (103.4 + exoatmElevation * (-12.79 + exoatmElevation * 0.711) ) ) 
        else:
            refractionCorrection = -20.774 / te
        refractionCorrection = refractionCorrection / 3600.0

    solarZen = zenith - refractionCorrection
    
    return azimuth, solarZen


def calcSolNoonUTC( jd : float, longitude : float ) -> float:
    """Calculate time of solar noon the given day at the given location on earth (in minute since 0 UTC)
    """
    tnoon = calcTimeJulianCent(jd)
    eqTime = calcEquationOfTime(tnoon)
    solNoonUTC = 720.0 - (longitude * 4.) - eqTime # in minutes
    return solNoonUTC


def calcSolNoon( jd: float, longitude: float, timezone: float, dst: bool) -> float:
    """Calculate time of solar noon the given day at the given location on earth (in minute)
    """
    timeUTC    = calcSolNoonUTC(jd, longitude)
    newTimeUTC = calcSolNoonUTC(jd + timeUTC/1440.0, longitude)
    solNoonLocal = newTimeUTC + (timezone*60.0) # in minutes
    if dst: 
        solNoonLocal += 60.0
    return solNoonLocal


def calcSunRiseSetUTC( jd : float, latitude: float, longitude : float) -> tuple[float, float]:
    """Calculate sunrise/sunset the given day at the given location on earth (in minute since 0 UTC)
    """
    t = calcTimeJulianCent(jd)
    eqTime = calcEquationOfTime(t)
    solarDec = calcSunDeclination(t)
    hourAngle = calcHourAngleSunrise(latitude, solarDec)
    # Rise time
    delta = longitude + numpy.degrees(hourAngle)
    riseTimeUTC = 720. - (4.0 * delta) - eqTime # in minutes
    # Set time
    hourAngle = -hourAngle
    delta = longitude + numpy.degrees(hourAngle)
    setTimeUTC = 720. - (4.0 * delta) - eqTime # in minutes
    return riseTimeUTC, setTimeUTC


def calcSunRiseSet( jd: float, latitude: float, longitude: float, timezone: float, dst: float ) -> tuple[float, float]:
    """Calculate sunrise/sunset the given day at the given location on earth (in minutes)
    """
    rtimeUTC, stimeUTC = calcSunRiseSetUTC(jd, latitude, longitude)
    # calculate local sunrise time (in minutes)
    rnewTimeUTC, snewTimeUTC = calcSunRiseSetUTC(jd + rtimeUTC/1440.0, latitude, longitude)
    rtimeLocal = rnewTimeUTC + (timezone * 60.0)
    rtimeLocal += 60.0 if dst else 0.0
    if rtimeLocal < 0.0 or rtimeLocal >= 1440.0: 
        jday = jd
        increment = 1. if rtimeLocal < 0. else -1.
        while rtimeLocal < 0.0 or rtimeLocal >= 1440.0:
            rtimeLocal += increment * 1440.0
            jday -= increment
    # calculate local sunset time (in minutes)
    rnewTimeUTC, snewTimeUTC = calcSunRiseSetUTC(jd + stimeUTC/1440.0, latitude, longitude)
    stimeLocal = snewTimeUTC + (timezone * 60.0)
    stimeLocal += 60.0 if dst else 0.0
    if stimeLocal < 0.0 or stimeLocal >= 1440.0: 
        jday = jd
        increment = 1. if stimeLocal < 0. else -1.
        while stimeLocal < 0.0 or stimeLocal >= 1440.0:
            stimeLocal += increment * 1440.0
            jday -= increment
    # return
    return rtimeLocal, stimeLocal


def calcTerminator( date: datetime.datetime, latitudes: Sequence[float], longitudes: Sequence[float] ,nlats: int = 50,nlons: int = 50 ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Calculate terminator position and solar zenith angle for a given julian date-time 
    within latitude/longitude limits
    
    Note that for plotting only, basemap has a built-in terminator.
    """
    jd = getJD(date)
    t = calcTimeJulianCent(jd)
    ut = ( jd - (int(jd - 0.5) + 0.5) )*1440.
    zen  = numpy.zeros((nlats,nlons))
    lats = numpy.linspace(latitudes[0],  latitudes[1],  num=nlats)
    lons = numpy.linspace(longitudes[0], longitudes[1], num=nlons)
    term = []
    for ilat in range(1,nlats+1):
        for ilon in range(nlons):
            az,el = calcAzEl(t, ut, lats[-ilat], lons[ilon], 0.) 
            zen[-ilat,ilon] = el
        a = (90 - zen[-ilat,:])
        mins = numpy.r_[False, a[1:]*a[:-1] <= 0] | \
            numpy.r_[a[1:]*a[:-1] <= 0, False] 
        zmin = mins & numpy.r_[False, a[1:] < a[:-1]]
        if True in zmin:
            ll = numpy.interp(0, a[zmin][-1::-1], lons[zmin][-1::-1])
            term.append([lats[-ilat], ll])
        zmin = mins & numpy.r_[a[:-1] < a[1:], False]
        if True in zmin:
            ll = numpy.interp(0, a[zmin], lons[zmin])
            term.insert(0, [lats[-ilat], ll])
    return lats, lons, zen, numpy.array(term)


def getJD(date: datetime.datetime) -> float:
    """Calculate the julian date from a python datetime object.
    """

    from dateutil.relativedelta import relativedelta
    
    if date.month < 2: 
        date.replace(year=date.year-1)
        date += relativedelta(month=12)

    A = numpy.floor(date.year/100.)
    B = 2. - A + numpy.floor(A/4.)
    jd = numpy.floor(365.25*(date.year + 4716.)) + numpy.floor(30.6001*(date.month+1)) + date.day + B - 1524.5
    jd = jd + date.hour/24.0 + date.minute/1440.0 + date.second/86400.0
    return jd
