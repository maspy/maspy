#!/usr/bin/env python
from collections import defaultdict
import tempfile
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from obspy import UTCDateTime, Stream
from obspy.core import AttribDict
from obspy.geodetics import locations2degrees,kilometer2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import getTravelTimes
import scipy.interpolate as spi
import scipy as sp
import matplotlib.cm as cm
from obspy.signal.util import util_geo_km,next_pow_2,utlLonLat
import ctypes as C
from obspy.core import Stream
import math
import warnings
from scipy.integrate import cumtrapz
from obspy.core import Stream
from obspy.signal.headers import clibsignal
from obspy.signal.invsim import cosine_taper
from obspy.taup import TauPyModel
from obspy.taup import getTravelTimes
from mpl_toolkits.basemap import Basemap
from matplotlib.image import NonUniformImage
from matplotlib import cm
from matplotlib.pyplot import figure
import shutil

KM_PER_DEG = 111.1949
os.system('clear')  # clear screen
model =  TauPyModel(model="ak135e")

def vespagram(stream, ev, inv, method, scale, nthroot=4, static3D=False, vel_corr=4.8, 
    sl=(0.0, 10.0, 0.1), plot_trace=True, phase_shift=0., phase=['PP'], plot_max_beam_trace=False, 
    save_fig=False, plot_circle_path=False, plot_stations=False,vespa_iter=0, component ='Z',
    static_correction=False, vespagram_show=True):
    """
    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param inventory: Station metadata for waveforms
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param method: Method used for the array analysis
        (one of "DLS": Delay and Sum, "PWS": Phase Weighted Stack).
    :type method: str
    :param frqlow: Low corner of frequency range for array analysis
    :type frqlow: float
    :param frqhigh: High corner of frequency range for array analysis
    :type frqhigh: float
    :param baz: pre-defined (theoretical or calculated) backazimuth used for calculation
    :type baz_plot: float
    :param scale: scale for plotting
    :type scale: float
    :param nthroot: estimating the nthroot for calculation of the beam
    :type nthroot: int
    :param filter: Whether to bandpass data to selected frequency range
    :type filter: bool
    :param static3D: static correction of topography using `vel_corr` as
        velocity (slow!)
    :type static3D: bool
    :param vel_corr: Correction velocity for static topography correction in
        km/s.
    :type vel_corr: float
    :param sl: Min/Max and stepwidth slowness for analysis
    :type sl: (float, float,float)
    :param plot_trace: if True plot the vespagram as wiggle plot, if False as density map
    :phase_shift: time shifting for ploting theoretical taup-traveltimes phases
    """

    print("BEGIN OF VESPAGRAM ANALYSIS ...")

    print (stream)

    # choose the maximun of the starting times
    starttime = max([tr.stats.starttime for tr in stream])
    # print 'starttime',starttime
    # choose the ninumun of the ending times
    endtime = min([tr.stats.endtime for tr in stream])

    # check that starttime is the real start time
    for i, tr in enumerate(stream):
        if tr.stats.starttime > starttime:
            msg = "Specified stime %s is smaller than starttime %s in stream"
            raise ValueError(msg % (starttime, tr.stats.starttime))
        if tr.stats.endtime < endtime:
            msg = "Specified etime %s is bigger than endtime %s in stream"
            raise ValueError(msg % (endtime, tr.stats.endtime))

    # keep only the shortest window lenght of the whole seismograms
    #stream.trim(starttime,endtime)
    # remove the trend
    #stream.detrend('linear')

    # print in the screen
    # print(starttime)
    # print(endtime)
    # closeInput = raw_input("Press ENTER to exit")
    
    org = ev.preferred_origin() or ev.origins[0]
    ev_lat = org.latitude
    ev_lon = org.longitude
    ev_depth = org.depth/1000.  # in km
    ev_otime = org.time

    # print(org)
    print 'ev_lat',ev_lat
    print 'ev_lon',ev_lon
    print 'ev_depth',ev_depth
    # print(ev_otime)
    #closeInput = raw_input("Press ENTER to exit")

    sll, slm, sls = sl
    # print sl
    # print sll
    # print slm
    # print sls
    # closeInput = raw_input("Press ENTER to exit")
    
    sll /= KM_PER_DEG
    slm /= KM_PER_DEG
    sls /= KM_PER_DEG
    center_lon = 0.
    center_lat = 0.
    center_elv = 0.
    seismo = stream
    #print len(seismo)
    #print(seismo)
    #seismo.attach_response(inv)
    #seismo.merge()
    sz = Stream()
    i = 0.
    net = len(inv.networks) # number of networks in the inventory file
    max_sampling_rate = 0.
    min_sampling_rate = 0.
    sampling_rate_count = 0. #  counter to determine if resampling is needed
    for tr in seismo:
        found = False
        for i_network in range(0,net): 
            if found: 
                break
            for station in inv[i_network].stations:
                if tr.stats.station == station.code:
                    # print 'sampling rate',tr.stats.sampling_rate
                    # print 'station.code',station.code
                    # print 'tr.stats.station',tr.stats.station
                    # print 'Network',inv.networks[i_network].code
                    # print 'latitude',station.latitude
                    # print 'longitude',station.longitude
                    # print '-----------------------------'
                    tr.stats.coordinates = \
                        AttribDict({'latitude': station.latitude,
                                    'longitude': station.longitude,
                                    'elevation': station.elevation})
                    center_lon += station.longitude
                    center_lat += station.latitude
                    center_elv += station.elevation
                    i += 1
                    found = True # exit the loop
                    if tr.stats.sampling_rate > max_sampling_rate:
                        sampling_rate_count += 1.
                        max_sampling_rate = tr.stats.sampling_rate
                    if tr.stats.sampling_rate < max_sampling_rate:
                        sampling_rate_count += 1.
                        min_sampling_rate = tr.stats.sampling_rate
                    if found:
                        break
        sz.append(tr)

    for network in inv:
        array_name = network.code
    array_name = array_name.encode('utf8')
    
    if sampling_rate_count > 1.: # resample the data
        seismo.resample(min_sampling_rate)
    
    # print '-----------------------------'
    # print'number of stations found',i
    # print'max smapling rate',max_sampling_rate
    # print'min_sampling_rate',min_sampling_rate
    # print'number of station in file',len(seismo)
    # print'sampling_rate_count',sampling_rate_count
    # print(len(sz))
    # print(sz)
    # sz.plot()

    # write the stations used to a file
    WRITE_INSTASEIS = False
    WRITE_STEPHANIE_FILE = False
    if WRITE_INSTASEIS:
        # Open a file
        fo = open("/Users/admin/Desktop/stations.txt", "wb")
        for tr in seismo:
          fo.write(tr.stats.station)
          fo.write(' ')
          fo.write(str(tr.stats.coordinates.latitude))
          fo.write(' ')
          fo.write(str(tr.stats.coordinates.longitude))
          fo.write('\n')
          #print tr.stats.station
          #print tr.stats.coordinates.latitude
          #print tr.stats.coordinates.longitude
    if WRITE_STEPHANIE_FILE:
        myfile = open('stephanie_files/info.txt', 'w')
        i = 0
        for tr in seismo: 
          a = tr.data
          station_name = tr.stats.station
          dt = tr.stats.sampling_rate
          npts = tr.stats.npts
          stat_lat = tr.stats.coordinates.latitude
          stat_lon = tr.stats.coordinates.longitude
          dist_km, baz, az2 = gps2dist_azimuth(stat_lat,stat_lon,ev_lat,ev_lon)
          dist_deg = dist_km / (KM_PER_DEG*1000)
          variables = [station_name,stat_lat,stat_lon,dt,npts,dist_deg]
          print variables
          myfile.write("%s\n" % variables)
          name = 'stephanie_files/schtefanie_' + `i` + '.txt'
          np.savetxt(name,a,delimiter=' ')
          i += 1

        # Close opend file
        fo.close()

    if i == 0.:
        msg = 'Stations can not be found!'
        raise ValueError(msg)

    if i < len(seismo):
        print'Number of stations: ',len(seismo)
        print'Number of stations found: ',i
        msg = 'Not all stations can be found!'
        raise ValueError(msg)

    if i > len(seismo):
        print'Number of stations: ',len(seismo)
        print'Number of stations found: ',i
        msg = 'There are station duplicates!'
        raise ValueError(msg)

    #sz.plot()
    #stream.plot()

    center_lon /= float(i)
    center_lat /= float(i)
    center_elv /= float(i)
    
    print 'center_lon',center_lon
    print 'center_lat',center_lat
    #closeInput = raw_input("Press ENTER to exit")

    # calculate the back azimuth
    great_circle_dist, baz, az2 = gps2dist_azimuth(center_lat,center_lon,ev_lat,ev_lon)
    great_circle_dist /=  (KM_PER_DEG*1000)
    # print 'great_circle_dist',great_circle_dist
    print'back-azimuth used is: ', baz
    # print("az2")
    # print(az2)

    if plot_circle_path:
       plot_great_circle_path(ev_lon,ev_lat,ev_depth,center_lon,center_lat,baz,great_circle_dist)
       
    if plot_stations:
       plot_array_stations(sz,center_lon,center_lat,array_name)

    # print(center_lon)
    # print(center_lat)
    # print(center_elv)
    
    #closeInput = raw_input("Press ENTER to exit")

    # trim it again?!?!
    stt = starttime
    e = endtime
    nut = 0.
    max_amp = 0.
    # sz.trim(stt, e)
    # sz.detrend('simple')
    # print sz

    # compute the number of traces in the vespagram
    nbeam = int((slm - sll)/sls + 0.5) + 1
    # print("nbeam")
    # print(nbeam)

    # arguments to compute the vespagram
    kwargs = dict(
        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll=sll, slm=slm, sls=sls, baz=baz, stime=starttime, etime=endtime, source_depth=ev_depth,
        distance=great_circle_dist, static_correction=static_correction, phase=phase, method=method,
        nthroot=nthroot, correct_3dplane=False, static_3D=static3D, vel_cor=vel_corr)
    
    # date to compute total time in routine
    start = UTCDateTime()

    # compute the vespagram
    slow, beams, max_beam, beam_max, mini, maxi = vespagram_baz(sz, **kwargs)
    # print 'mini',mini
    # print 'maxi',maxi
    # print 'starttime',starttime
    # print slow
    # print total time in routine
    print "Total time in routine: %f\n" % (UTCDateTime() - start)
    
    # Plot the seismograms
    # sampling rate
    df = sz[0].stats.sampling_rate
    # print'df',df
    npts = len(beams[0])
    #print("npts")
    #print(npts)
    # time vector
    T = np.arange(0, npts/df, 1/df)
    # reconvert slowness to degrees
    sll *= KM_PER_DEG
    slm *= KM_PER_DEG
    sls *= KM_PER_DEG
    # slowness vector
    slow = np.arange(sll, slm, sls)
    max_amp = np.max(beams[:, :])
    #min_amp = np.min(beams[:, :])
    scale *= sls
    
    # initialize the figure
    fig = plt.figure(num=1,figsize=(13,6),dpi=100)
    
#    print("sl")
#    print(sl)
#    print("sll")
#    print(sll)
#    print("slm")
#    print(slm)
#    print("sls")
#    print(sls)

    # get taup points for ploting the phases 
    phase_name_info,phase_slowness_info,phase_time_info = get_taupy_points(center_lat,center_lon,ev_lat,ev_lon,ev_depth, \
                                                                starttime,endtime,mini,maxi,ev_otime,phase_shift,sll,slm)

    # print(phase_name_info)
    # print(phase_slowness_info)
    # print(phase_time_info)

    schtefanie = False
    if schtefanie:
        trace1 = sz[0]
        plt.figure(num=2,figsize=(24,10),dpi=100)
        T_seis = np.linspace(0, 1./df, len(trace1))
        print len(trace1)
        print len(T_seis)
        plt.plot(T_seis,trace1)
        plt.scatter(phase_time_info,np.zeros(len(phase_time_info)),s=2000,marker=u'|',lw=2,color='g')

    if plot_trace:
        ax1 = fig.add_axes([0.1, 0.15, 0.75, 0.75])
        for i in xrange(nbeam):
            if plot_max_beam_trace:
              if i == max_beam:
                ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'r',zorder=1)
              else:
                ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'k',zorder=-1)
            else:    
              ax1.plot(T, sll + scale*beams[i]/max_amp + i*sls, 'k',zorder=-1)
        ax1.set_xlabel('Time (s)',size=20)
        ax1.set_ylabel('slowness (s/deg)',size=20)
        ax1.set_xlim(T[0], T[-1])
        ax1.xaxis.set_tick_params(labelsize=22)
        ax1.yaxis.set_tick_params(labelsize=22)
        data_minmax = ax1.yaxis.get_data_interval()
        minmax = [min(slow[0], data_minmax[0]), max(slow[-1], data_minmax[1])]
        ax1.set_ylim(*minmax)
        # plot the phase info
        ax1.scatter(phase_time_info,phase_slowness_info,s=2000,marker=u'|',lw=2,color='g')
        for i, txt in enumerate(phase_name_info):
            ax1.annotate(txt,(phase_time_info[i],phase_slowness_info[i]),fontsize=24,color='r')
    #####
    else:
        #step = (max_amp - min_amp)/100.
        #level = np.arange(min_amp, max_amp, step)
        #beams = beams.transpose()
        cmap = cm.hot_r
        #cmap = cm.rainbow

        ax1 = fig.add_axes([0.1, 0.1, 0.7, 0.7])
        #ax1.contour(slow,T,beams,level)
        #extent = (slow[0], slow[-1], \
        #               T[0], T[-1])
        extent = (T[0], T[-1], slow[0] - sls * 0.5, slow[-1] + sls * 0.5)

        ax1.set_ylabel('slowness (s/deg)')
        ax1.set_xlabel('T (s)')
        beams = np.flipud(beams)
        ax1.imshow(beams, cmap=cmap, interpolation="nearest",extent=extent, aspect='auto')
        # plot the phase info
        ax1.scatter(phase_time_info,phase_slowness_info,s=2000,marker=u'|',lw=2,color='g')
        for i, txt in enumerate(phase_name_info):
            ax1.annotate(txt,(phase_time_info[i],phase_slowness_info[i]),fontsize=17,color='r')

    ####
    result = "BAZ: %.2f Time %s" % (baz, stt)
    # ax1.set_title(result)
    ax1.set_title('VESPAGRAM',size=24)

    if vespagram_show:
        plt.show()
    
    # save the figure
    save_fig = False
    if save_fig:
        print 'I am printing the figure in a file!'
        file_name = 'vespagram_' + component + '.png'
        plt.savefig(file_name)
        #plt.clf() # clear the figure
        #plt.close() # close the figure
    #else:
        #plt.show()
    
    #return
    return center_lon, center_lat, phase_name_info,phase_slowness_info,phase_time_info    
    # return slow, beams, max_beam, beam_max

def vespagram_baz(stream, sll, slm, sls, baz, stime, etime, source_depth, distance, static_correction,
        phase, verbose=False, coordsys='lonlat', timestamp='mlabday', method="DLS", nthroot=1,
        store=None, correct_3dplane=False, static_3D=False, vel_cor=4.):

    """
    Estimating the azimuth or slowness vespagram

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type sll: Float
    :param sll: slowness  min (lower)
    :type slm: Float
    :param slm: slowness max
    :type sls: Float
    :param sls: slowness step
    :type baz: Float
    :param baz: given backazimuth
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :type store: function
    :param store: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.
    :return: numpy.ndarray of beams with different slownesses
    """

    # compare th original trace with the traces used for computing the vespagram
    #stream.plot()
    #closeInput = raw_input("Press ENTER to exit")

    # check that sampling rates do not vary
    # sampling rate
    fs = stream[0].stats.sampling_rate

    # number of stations
    nstat = len(stream)
    # print("nstat")
    # print(nstat)

    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    # maximum lenght of the seismogram that can be used
    ndat = int((etime - stime)*fs)
    #print ndat
    
    # number of beam traces
    nbeams = int(((slm - sll) / sls + 0.5) + 1)
    #print("nbeams")
    #print nbeams
    # closeInput = raw_input("Press ENTER to exit")

    geometry = get_geometry(stream,coordsys=coordsys,verbose=verbose)
    #stream.plot()
    #print("geometry:")
    # print(geometry)
    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift_baz(geometry, sll, slm, sls, baz, source_depth,
         distance, phase, vel_cor=vel_cor, static_3D=static_3D, model=model, static_correction=static_correction)

    # calculate the overlaping lenght of the traces
    mini = np.min(time_shift_table[:,:])
    maxi = np.max(time_shift_table[:,:])
    # print("mini")
    # print(mini)
    # print("maxi")
    # print(maxi)
    #print("stime")
    #print(stime)
    #print("etime")
    #print(etime)
    spoint, _epoint = get_spoint(stream, (stime - mini), (etime - maxi))
    
    # print("spoint")
    # print(spoint)
    # print("epoint")
    # print(_epoint)

    # recalculate the maximum possible trace length
    ndat = int(((etime - maxi) - (stime - mini)) * fs) + 1
    seconds = (etime - maxi) - (stime - mini)
    # print("ndat recalculated")
    # print(ndat)
    # print("total second of the vespagram")
    # print(seconds)
    number_total_points = len(stream[0]) 
    biggest_left = int(np.abs(mini) * fs) + 1
    biggest_right = number_total_points - int(np.abs(maxi) * fs) - 1
    efective_trace_lenght = biggest_right - biggest_left 

    # vespagram matrix
    beams = np.zeros((nbeams, ndat), dtype='f8')
    
    # initialize variabes
    max_beam = 0.
    slow = 0.
    beam_max = 0.   
    
    # print("efective_trace_lenght")
    # print(efective_trace_lenght)
    # print("number total efective points")
    # print(number_total_points)
    # print("biggest_left")
    # print(biggest_left)
    # print("ndat")
    # print(ndat)
    # print("sampling_rate")
    # print(fs)
    # print("biggest_right")
    # print(biggest_right)
    # print("stime")
    # print(stime)
    # print("mini")
    # print(mini)
    # print("maxi")
    # print(maxi)

    for x in xrange(nbeams):
        singlet = 0.
        if method == 'DLS':
            for i in xrange(nstat):
                # check the nthroot used
                #print("nthroot", nthroot)
                
                # correct way to do it!
                starting_point =  biggest_left + int(time_shift_table[i, x] * fs)
                ending_point = starting_point + ndat
                
                # original implementation
                s = spoint[i] + int(time_shift_table[i, x]*fs + 0.5)
                shifted = stream[i].data[s: s + ndat]
                
                # our implementation
                #shifted = stream[i].data[starting_point : ending_point]
                #print(shifted)
                
                singlet += 1. / nstat * np.sum(shifted * shifted)
                
                # compute the vespagram
                beams[x] += 1. / nstat * np.power(np.abs(shifted), 1. / nthroot) * shifted / np.abs(shifted)
            
            beams[x] = np.power(np.abs(beams[x]), nthroot) * beams[x] / np.abs(beams[x])
            bs = np.sum(beams[x]*beams[x])
            bs /= singlet
            #bs = np.abs(np.max(beams[x]))
            if bs > max_beam:
                max_beam = bs
                beam_max = x
                slow = np.abs(sll + x * sls)
                if (slow) < 1e-8:
                    slow = 1e-8
                    
        if method == 'PWS':
           stack = np.zeros(ndat, dtype='c8')
           phase = np.zeros(ndat, dtype='f8')
           coh = np.zeros(ndat, dtype='f8')
           for i in xrange(nstat):
               s = spoint[i] + int(time_shift_table[i, x] * fs +0.5)
               try:
                  shifted = sp.signal.hilbert(stream[i].data[s : s + ndat])
               except IndexError:
                  break
               phase = np.arctan2(shifted.imag, shifted.real)
               stack.real += np.cos(phase)
               stack.imag += np.sin(phase)
           coh = 1. / nstat * np.abs(stack)
           for i in xrange(nstat):
               s = spoint[i]+int(time_shift_table[i, x] * fs + 0.5)
               shifted = stream[i].data[s: s + ndat]
               singlet += 1. / nstat * np.sum(shifted * shifted)
               beams[x] += 1. / nstat * shifted * np.power(coh, nthroot)
           bs = np.sum(beams[x]*beams[x])
           bs = bs / singlet
           if bs > max_beam:
              max_beam = bs
              beam_max = x
              slow = np.abs(sll + x * sls)
              if (slow) < 1e-8:
                  slow = 1e-8


    return(slow, beams, beam_max, max_beam, mini, maxi)

def get_spoint(stream, stime, etime):
    """
    Calculates start and end offsets relative to stime and etime for each
    trace in stream in samples.

    :type stime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param stime: Start time
    :type etime: :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param etime: End time
    :returns: start and end sample offset arrays
    """
    spoint = np.empty(len(stream), dtype=np.int32, order="C")
    epoint = np.empty(len(stream), dtype=np.int32, order="C")
    for i, tr in enumerate(stream):
        if tr.stats.starttime > stime:
            msg = "Specified stime %s is smaller than starttime %s in stream"
            raise ValueError(msg % (stime, tr.stats.starttime))
        if tr.stats.endtime < etime:
            msg = "Specified etime %s is bigger than endtime %s in stream"
            raise ValueError(msg % (etime, tr.stats.endtime))
        # now we have to adjust to the beginning of real start time
        spoint[i] = int((stime - tr.stats.starttime) *
                        tr.stats.sampling_rate + .5)
        epoint[i] = int((tr.stats.endtime - etime) *
                        tr.stats.sampling_rate + .5)
    return spoint, epoint

def get_timeshift_baz(geometry, sll, slm, sls, baze, source_depth, distance, phase, 
        vel_cor=4., static_3D=False, model=model, static_correction=False):
    """
    Returns timeshift table for given array geometry and a pre-definded
    backazimut

    :param geometry: Nested list containing the arrays geometry, as returned by
            get_group_geometry
    :param sll_x: slowness x min (lower)
    :param slm_y: slowness x max (lower)
    :param sl_s: slowness step
    :param baze:  backazimuth applied
    :param vel_cor: correction velocity (upper layer) in km/s
    :param static_3D: a correction of the station height is applied using
        vel_cor the correction is done according to the formula:
        t = rxy*s - rz*cos(inc)/vel_cor
        where inc is defined by inv = asin(vel_cor*slow)
    """
    nstat = len(geometry)  # last index are center coordinates
    baz = math.pi * baze / 180. # BAZ converted from degrees to radiants
    nbeams = int((slm - sll) / sls + 0.5) + 1

    # check the correct values
    # print(slm*KM_PER_DEG)
    # print(sll*KM_PER_DEG)
    # print(nbeams)

    # time shift table is given by the number of staions and number of beam traces
    time_shift_tbl = np.empty((nstat, nbeams), dtype="float32")

    arrivals = model.get_travel_times(source_depth, distance, phase_list = phase)
    inc_ang = arrivals[0].incident_angle
    inc_ang_rad = inc_ang * np.pi/180

    for k in xrange(nbeams):
        sx = sll + k * sls
        #print(sx)
        if vel_cor*sx < 1.:
            # print("Im in velocity correction - timeshift!!!")
            inc = np.arcsin(vel_cor*sx)
        else:
            inc = np.pi/2.

        # time shift table matrix    
        #time_shift_tbl[:, k] = - sx * (geometry[:, 0] * math.sin(baz) + geometry[:, 1] * math.cos(baz))

        #if static_3D:
           #time_shift_tbl[:, k] += geometry[:, 2] * np.cos(inc) / vel_cor
		
        if static_correction:
            time_shift_tbl[:, k] = - sx * (geometry[:, 0] * math.sin(baz) + geometry[:, 1] * math.cos(baz))  \
                                   + sx * geometry[:, 2]/1000 * 1./np.tan(inc_ang_rad)
            # print(1./np.tan(inc_ang_rad))
            # print(time_shift_tbl)

        else:
            # time shift table matrix    
            time_shift_tbl[:, k] = - sx * (geometry[:, 0] * math.sin(baz) + geometry[:, 1] * math.cos(baz))	
						
            # print(time_shift_tbl)

            if static_3D:
	            time_shift_tbl[:, k] += geometry[:, 2] * np.cos(inc) / vel_cor
    
    return time_shift_tbl

def get_geometry(stream,coordsys='lonlat',return_center=True,verbose=False):
    """
    Method to calculate the array geometry and the center coordinates in km

    :param stream: Stream object, the trace.stats dict like class must
        contain an :class:`~obspy.core.util.attribdict.AttribDict` with
        'latitude', 'longitude' (in degrees) and 'elevation' (in km), or 'x',
        'y', 'elevation' (in km) items/attributes. See param ``coordsys``
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :param return_center: Returns the center coordinates as extra tuple
    :return: Returns the geometry of the stations as 2d :class:`numpy.ndarray`
            The first dimension are the station indexes with the same order
            as the traces in the stream object. The second index are the
            values of [lat, lon, elev] in km
            last index contains center [lat, lon, elev] in degrees and km if
            return_center is true
    """
    nstat = len(stream)
    center_lat = 0.
    center_lon = 0.
    center_h = 0.
    geometry = np.empty((nstat, 3))

    if isinstance(stream, Stream):
        for i, tr in enumerate(stream):
            if coordsys == 'lonlat':
                geometry[i, 0] = tr.stats.coordinates.longitude
                geometry[i, 1] = tr.stats.coordinates.latitude
                geometry[i, 2] = tr.stats.coordinates.elevation
                #print("latitude",geometry[i, 1])
                #print("longitude",geometry[i, 0])
            elif coordsys == 'xy':
                geometry[i, 0] = tr.stats.coordinates.x
                geometry[i, 1] = tr.stats.coordinates.y
                geometry[i, 2] = tr.stats.coordinates.elevation
    elif isinstance(stream, np.ndarray):
        print("Im not here!!")
        geometry = stream.copy()
    else:
        raise TypeError('only Stream or numpy.ndarray allowed')

    if verbose:
        print("coordsys = " + coordsys)

    if coordsys == 'lonlat':
        center_lon = geometry[:, 0].mean()
        center_lat = geometry[:, 1].mean()
        # print(center_lon)
        # print(center_lat)
        center_h = geometry[:, 2].mean()
        
        for i in np.arange(nstat):
            x, y = util_geo_km(center_lon, center_lat, geometry[i, 0],geometry[i, 1])  # original: utlGeoKm
            geometry[i, 0] = x
            geometry[i, 1] = y
            geometry[i, 2] -= center_h
    elif coordsys == 'xy':
        geometry[:, 0] -= geometry[:, 0].mean()
        geometry[:, 1] -= geometry[:, 1].mean()
        geometry[:, 2] -= geometry[:, 2].mean()
    else:
        raise ValueError("Coordsys must be one of 'lonlat', 'xy'")
    
    if return_center:
        return np.c_[geometry.T,np.array((center_lon, center_lat, center_h))].T
    else:
        return geometry


def get_taupy_points(center_lat,center_lon,ev_lat,ev_lon,ev_depth,
                     stime,etime,mini,maxi,ev_otime,phase_shift,sll,slm):
  
  distance = locations2degrees(center_lat,center_lon,ev_lat,ev_lon)
  #print(distance)

  earthmodel =  TauPyModel(model="ak135e")
  #earthmodel =  TauPyModel(model="prem")
  #earthmodel =  TauPyModel(model="pwdk")
  arrivals = earthmodel.get_pierce_points(ev_depth,distance,phase_list=['P', 'PcP', 'PKiKP', 'Pdiff', 'PP', 'P^410P','P^660P','PKP'])
  
  # compute the vespagram window
  start_vespa = stime - mini
  end_vespa = etime - maxi

  # compare the arrival times with the time window
  count = 0
  k = 0
  phase_name_info = []
  phase_slowness_info = []
  phase_time_info = []

  for i_elem in arrivals:
    #print(i_elem)
    dummy_phase = arrivals[count]
    # print('dummy_phase_name',dummy_phase.name) #  print phases and traveltimes in the screen!
    # phase time in seconds
    taup_phase_time = dummy_phase.time
    # print('taup_phase_time',taup_phase_time)
    # slowness of the phase
    taup_phase_slowness = dummy_phase.ray_param_sec_degree
    # compute the UTC travel phase time
    # print 'ev_otime',ev_otime
    # print 'taup_phase_time',taup_phase_time
    # print 'phase_shift',phase_shift
    taup_phase_time2 = ev_otime + taup_phase_time + phase_shift
    # print 'taup_phase_time2',taup_phase_time2
    # print('phase_name,time',dummy_phase.name,taup_phase_time)
    # print('phase_name,corrected time',dummy_phase.name,taup_phase_time2)
    # print(start_vespa)
    # print(end_vespa)
    # print(taup_phase_time2)

    if start_vespa <= taup_phase_time2 <= end_vespa: # time window    
      if sll <= taup_phase_slowness <= slm: # slowness window
      
        # seconds inside the vespagram
        # print 'start_vespa',start_vespa
        # print 'taup_phase_time2',taup_phase_time2
        taup_mark = taup_phase_time2 - start_vespa
        # taup_mark = taup_phase_time2 - stime #  uncomment this one for the seismogram!
        # print 'start_vespa',start_vespa
        # print 'taup_mark', taup_mark
        # store the information
        phase_name_info.append(dummy_phase.name)
        phase_slowness_info.append(dummy_phase.ray_param_sec_degree)
        phase_time_info.append(taup_mark)
        #print(phases_info[k])
        k += 1

    count += 1  
    
  #print(phase_name_info)
  
  phase_slowness_info = np.array(phase_slowness_info)
  phase_time_info = np.array(phase_time_info)

  return phase_name_info, phase_slowness_info, phase_time_info


def plot_great_circle_path(ev_lon,ev_lat,ev_depth,center_lon,center_lat,baz,great_circle_dist):

  plt.figure(num=3,figsize=(17,10),dpi=100) # define plot size in inches (width, height) & resolution(DPI)

  distance = locations2degrees(center_lat,center_lon,ev_lat,ev_lon)
  #print(distance)
  earthmodel =  TauPyModel(model='ak135e')
  arrivals = earthmodel.get_pierce_points(ev_depth,distance,phase_list=['PP'])
  #print(arrivals)
  arrival = arrivals[0]
  pierce_info = arrival.pierce
  #print(pierce_info)
  max_index = 0.
  count = 0.
  max_val = 0.
  for i_index in pierce_info:
    #print(i_index)
    count += 1
    if i_index[3] > max_val: 
      max_val = i_index[3] 
      max_index = count - 1
      #print(max_index)
  
  #print(max_index)
  bounce_vect = pierce_info[max_index]
  bounce_dist = bounce_vect[2] / 0.017455053237912375 # convert from radians to degrees
  # print("bounce_dist")
  # print(bounce_dist)

  # print("ev_lat")
  # print(ev_lat)
  # print("ev_lon")
  # print(ev_lon)
  # print("center_lon")
  # print(center_lon)
  # print("center_lat")
  # print(center_lat)
  # print("backazimuth")
  # print(baz)

  # bounce point approximation
  bounce_lat_appx, bounce_lon_appx = midpoint(ev_lat,ev_lon,center_lat,center_lon)

  # putting everything into a vector
  lons = [ev_lon, center_lon]
  lats = [ev_lat, center_lat]

  # trick - the basemap functions does not like the arguments that math gives
  resolution = 0.0001
  bounce_lon_appx = np.round(bounce_lon_appx/resolution)*resolution
  bounce_lat_appx = np.round(bounce_lat_appx/resolution)*resolution

  # print(bounce_lon_appx)
  # print(bounce_lat_appx)

  # plot results
  map = Basemap(projection='hammer',lon_0=bounce_lon_appx,lat_0=bounce_lat_appx,resolution='c')
  map.drawcoastlines()
  # map.fillcontinents()
  # map.drawmapboundary()
  map.fillcontinents(color='#cc9966',lake_color='#99ffff')
  map.drawmapboundary(fill_color='#99ffff')
  msg = "Great circle path distance: %.2f degrees" % great_circle_dist
  plt.title(msg,fontsize=18) 

  # draw great circle path
  map.drawgreatcircle(ev_lon,ev_lat,center_lon,center_lat,linewidth=3,color='g')

  # plot event
  x, y = map(ev_lon,ev_lat)
  map.scatter(x, y, 200, marker='*',color='k',zorder=10)
  
  # plot receiver
  x, y = map(center_lon,center_lat)
  map.scatter(x, y, 100, marker='^',color='k',zorder=10)
  
  # plot the bounce point approximated
  x, y = map(bounce_lon_appx,bounce_lat_appx)
  map.scatter(x, y, 100, marker='D',color='k',zorder=10)


def midpoint(lat1,lon1,lat2,lon2):
  
  # compute the mid point between two coordinates in degrees
  assert -90 <= lat1 <= 90
  assert -90 <= lat2 <= 90
  assert -180 <= lon1 <= 180
  assert -180 <= lon2 <= 180
  lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))

  dlon = lon2 - lon1
  dx = math.cos(lat2) * math.cos(dlon)
  dy = math.cos(lat2) * math.sin(dlon)
  lat3 = math.atan2(math.sin(lat1) + math.sin(lat2), math.sqrt((math.cos(lat1) + dx) * (math.cos(lat1) + dx) + dy * dy))
  lon3 = lon1 + math.atan2(dy, math.cos(lat1) + dx)
  
  return(math.degrees(lat3), math.degrees(lon3))

def align_phases(stream, event, inventory, phase_name, method="simple"):
    """
    Aligns the waveforms with the theoretical travel times for some phase. The
    theoretical travel times are calculated with obspy.taup.

    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param event: The event for which to calculate phases.
    :type event: :class:`obspy.core.event.Event`
    :param inventory: Station metadata.
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param phase_name: The name of the phase you want to align. Must be
        contained in all traces. Otherwise the behaviour is undefined.
    :type phase_name: str
    :param method: Method is either `simple` or `fft`. Simple will just shift
     the starttime of Trace, while 'fft' will do the shift in the frequency
     domain. Defaults to `simple`.
    :type method: str
    """
    method = method.lower()
    if method not in ['simple', 'fft']:
        msg = "method must be 'simple' or 'fft'"
        raise ValueError(msg)

    stream = stream.copy()
    attach_coordinates_to_traces(stream, inventory, event)

    stream.traces = sorted(stream.traces, key=lambda x: x.stats.distance)[::-1]

    tr_1 = stream[-1]
    tt_1 = getTravelTimes(tr_1.stats.distance,
                          event.origins[0].depth / 1000.0,
                          "ak135e")
    
    cont = 0.
    for tt in tt_1:
        if tt["phase_name"] != phase_name:
            continue
        if tt["phase_name"] == phase_name:
            cont = 1.
        tt_1 = tt["time"]
        break
   
    if cont == 0:    
      msg = "The selected phase is not present in your seismograms!!!"
      raise ValueError(msg)
        
    for tr in stream:
        tt = getTravelTimes(tr.stats.distance,
                            event.origins[0].depth / 1000.0,
                            "ak135e")
        for t in tt:
            if t["phase_name"] != phase_name:
                continue
            tt = t["time"]
            break
        if method == "simple":
            tr.stats.starttime -= (tt - tt_1)
        else:
            shifttrace_freq(Stream(traces=[tr]), [- ((tt - tt_1))])
    return stream

def shifttrace_freq(stream, t_shift):
    if isinstance(stream, Stream):
        for i, tr in enumerate(stream):
            ndat = tr.stats.npts
            samp = tr.stats.sampling_rate
            nfft = next_pow_2(ndat)
            nfft *= 2
            tr1 = np.fft.rfft(tr.data, nfft)
            for k in xrange(0, nfft / 2):
                tr1[k] *= np.complex(
                    np.cos((t_shift[i] * samp) * (k / float(nfft))
                           * 2. * np.pi),
                    -np.sin((t_shift[i] * samp) *
                            (k / float(nfft)) * 2. * np.pi))

            tr1 = np.fft.irfft(tr1, nfft)
            tr.data = tr1[0:ndat]

def attach_coordinates_to_traces(stream, inventory, event=None):
    """
    Function to add coordinates to traces.

    It extracts coordinates from a :class:`obspy.station.inventory.Inventory`
    object and writes them to each trace's stats attribute. If an event is
    given, the distance in degree will also be attached.

    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param inventory: Station metadata for waveforms
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param event: If the event is given, the event distance in degree will also
     be attached to the traces.
    :type event: :class:`obspy.core.event.Event`
    """
    # Get the coordinates for all stations
    coords = {}
    for network in inventory:
        for station in network:
            # coords["%s.%s" % (network.code, station.code)] = \
            coords[".%s" % (station.code)] = \
                {"latitude": station.latitude,
                 "longitude": station.longitude,
                 "elevation": station.elevation}

    # Calculate the event-station distances.
    if event:
        event_lat = event.origins[0].latitude
        event_lng = event.origins[0].longitude
        for value in coords.values():
            value["distance"] = locations2degrees(
                value["latitude"], value["longitude"], event_lat, event_lng)

    # Attach the information to the traces.
    #for trace in stream:
        #station = ".".join(trace.id.split(".")[:2])
        #station = ".".join(trace.id.split(".")[1:2])
        #value = coords[station]
        #trace.stats.coordinates = AttribDict()
        #trace.stats.coordinates.latitude = value["latitude"]
        #trace.stats.coordinates.longitude = value["longitude"]
        #trace.stats.coordinates.elevation = value["elevation"]
        #if event:
            #trace.stats.distance = value["distance"]   


def show_distance_plot(stream, event, inventory, starttime, endtime,
                       plot_travel_times=True):
    """
    Plots distance dependent seismogramm sections.

    :param stream: The waveforms.
    :type stream: :class:`obspy.core.stream.Stream`
    :param event: The event.
    :type event: :class:`obspy.core.event.Event`
    :param inventory: The station information.
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param starttime: starttime of traces to be plotted
    :type starttime: UTCDateTime
    :param endttime: endttime of traces to be plotted
    :type endttime: UTCDateTime
    :param plot_travel_times: flag whether phases are marked as traveltime plots
     in the section obspy.taup is used to calculate the phases
    :type pot_travel_times: bool
    """

    # choose the maximun of the starting times
    starttime = max([tr.stats.starttime for tr in stream])
    # choose the ninumun of the ending times
    endtime = min([tr.stats.endtime for tr in stream])
    # keep only the shortest window lenght of the whole seismograms
    #stream.trim(starttime,endtime)

    stream.resample(20)

    #stream = stream.slice(starttime=starttime, endtime=endtime).copy()
    event_depth_in_km = event.origins[0].depth / 1000.0
    event_time = event.origins[0].time

    #attach_coordinates_to_traces(stream, inventory, event=event)

    cm = plt.cm.jet

    stream.traces = sorted(stream.traces, key=lambda x: x.stats.distance)[::-1]
     
    # One color for each trace.
    colors = [cm(_i) for _i in np.linspace(0, 1, len(stream))]

    # Relative event times.
    times_array = stream[0].times() + (stream[0].stats.starttime - event_time)

    distances = [tr.stats.distance for tr in stream]
    min_distance = min(distances)
    max_distance = max(distances)
    distance_range = max_distance - min_distance
    stream_range = distance_range / 10.0

    # Normalize data and "shift to distance".
    stream.normalize()
    for tr in stream:
        tr.data *= stream_range
        tr.data += tr.stats.distance

    plt.figure(figsize=(18, 10))
    print 'times_array',times_array

    for _i, tr in enumerate(stream):
        print 'i',_i
        print 'itr',tr
        plt.plot(times_array,tr.data,label="%s.%s" % (tr.stats.network,tr.stats.station), color=colors[_i])
    plt.grid()
    plt.ylabel("Distance in degree to event")
    plt.xlabel("Time in seconds since event")
    plt.legend()

    dist_min, dist_max = plt.ylim()

    if plot_travel_times:

        distances = defaultdict(list)

        ttimes = defaultdict(list)
        # print'hola1'
        for i in np.linspace(dist_min, dist_max, 100):
            tts = getTravelTimes(i, event_depth_in_km, "ak135e", phase_list=['P', 'PcP', 'PKiKP', 'Pdiff','PP','P^410P','P^660P', 'pP', 'sP'])
            #tts = getTravelTimes(i, event_depth_in_km, "ak135e", phase_list=['Pdiff','PP'])
            for phase in tts:
                name = phase["phase_name"]
                distances[name].append(i)
                ttimes[name].append(phase["time"])
        # print'hola2'
        # print 'distances',distances
        # print 'ttimes',ttimes

        for key in distances.iterkeys():
            min_distance = min(distances[key])
            max_distance = max(distances[key])
            min_tt_time = min(ttimes[key])
            max_tt_time = max(ttimes[key])

            if min_tt_time >= times_array[-1] or \
                    max_tt_time <= times_array[0] or \
                    (max_distance - min_distance) < 0.8 * (dist_max - dist_min):
                continue
            ttime = ttimes[key]
            dist = distances[key]
            if max(ttime) > times_array[0] + 0.9 * times_array.ptp():
                continue
            plt.scatter(ttime, dist, s=0.5, zorder=-10, color="black", alpha=0.8)
            plt.text(max(ttime) + 0.005 * times_array.ptp(),
                     dist_max - 0.02 * (dist_max - dist_min),
                     key)

    plt.ylim(dist_min, dist_max)
    plt.xlim(times_array[0], times_array[-1])

    plt.title(event.short_str())

    plt.show()

def plot_array_stations(stream,center_lon,center_lat,array_name):
    
    plt.figure(num=2,figsize=(17,10),dpi=100) # define plot size in inches (width, height) & resolution(DPI)
    
    nstat = len(stream)
    array = np.empty((nstat, 2))
    
    for i, tr in enumerate(stream):
      array[i, 0] = tr.stats.coordinates.longitude
      array[i, 1] = tr.stats.coordinates.latitude
    
    # minimum 
    puffer = 2
    minlon = np.min(array[:,0]) - puffer
    maxlon = np.max(array[:,0]) + puffer
    minlat = np.min(array[:,1]) - puffer
    maxlat = np.max(array[:,1]) + puffer
        
    # plot results
    map = Basemap(projection='tmerc',llcrnrlon=minlon,llcrnrlat=minlat,urcrnrlon=maxlon,urcrnrlat=maxlat,lon_0=center_lon,lat_0=center_lat,resolution='i')
    map.drawcoastlines()
    # map.fillcontinents()
    # map.drawmapboundary()
    map.fillcontinents(color='#cc9966',lake_color='#99ffff')
    map.drawmapboundary(fill_color='#99ffff')
    plt.title('Array %s' % array_name, fontsize = 20)

    # draw parallels.
    parallels = np.arange(-90,90,2.0)
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    # draw meridians
    meridians = np.arange(0,360.,4)
    map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

    # plot corrdinates
    x, y = map(array[:, 0],array[:, 1])
    map.scatter(x, y, 200, marker='^',color='k',zorder=10)
    
    # plot the geometrical center
    x, y = map(center_lon,center_lat)
    map.scatter(x, y, 200, marker='*',color='r',zorder=10)

    # plot station names
    for i,tr in enumerate(stream):
        x, y = map(array[i,0],array[i,1])
        # print(x,y)
        plt.annotate(tr.stats.station,(x*1.02,y*1.02),fontsize=18,color='k')


def array_analysis_helper(stream, ev, inventory, method, frqlow, frqhigh,
                          filter=False, baz_plot=True, static3D=False,
                          vel_corr=4.8, wlen=-1, slx=(-10, 10),
                          sly=(-10, 10), sls=0.5, array_response=False,
                          component='Z'):
    """
    Array analysis wrapper routine for MESS 2014.

    :param stream: Waveforms for the array processing.
    :type stream: :class:`obspy.core.stream.Stream`
    :param inventory: Station metadata for waveforms
    :type inventory: :class:`obspy.station.inventory.Inventory`
    :param method: Method used for the array analysis
     (one of "FK": Frequnecy Wavenumber, "DLS": Delay and Sum,
     "PWS": Phase Weighted Stack, "SWP": Slowness Whitened Power).
    :type method: str
    :param filter: Whether to bandpass data to selected frequency range
    :type filter: bool
    :param frqlow: Low corner of frequency range for array analysis
    :type frqlow: float
    :param frqhigh: High corner of frequency range for array analysis
    :type frqhigh: float
    :param baz_plot: Whether to show backazimuth-slowness map (True) or
     slowness x-y map (False).
    :type baz_plot: str
    :param static3D: static correction of topography using `vel_corr` as
     velocity (slow!)
    :type static3D: bool
    :param vel_corr: Correction velocity for static topography correction in
     km/s.
    :type vel_corr: float
    :param wlen: sliding window for analysis in seconds, use -1 to use the
     whole trace without windowing.
    :type wlen: float
    :param slx: Min/Max slowness for analysis in x direction.
    :type slx: (float, float)
    :param sly: Min/Max slowness for analysis in y direction.
    :type sly: (float, float)
    :param sls: step width of slowness grid
    :type sls: float
    :param array_response: superimpose array reponse function in plot (slow!)
    :type array_response: bool
    """
    print("BEGIN OF ARRAY ANALYSIS HELPER ...")
    
    if method not in ("FK", "DLS", "PWS", "SWP"):
        raise ValueError("Invalid method: ''" % method)
    
    org = ev.preferred_origin() or ev.origins[0]
    ev_lat = org.latitude
    ev_lon = org.longitude
    ev_depth = org.depth/1000. # in km
    ev_otime = org.time
    
    geometry = get_geometry(stream,coordsys='lonlat',return_center=True,verbose=False)
    #print 'geometry',geometry

    # get center_lat and center_lon values
    center_lon = geometry[-1,0]
    center_lat = geometry[-1,1]
    # print 'center_lon', center_lon
    # print 'center_lat', center_lat

    great_circle_dist, teor_baz, az2 = gps2dist_azimuth(center_lat,center_lon,ev_lat,ev_lon)
    #print 'teor_baz',teor_baz

    sllx, slmx = slx
    slly, slmy = sly
    
    # trim the traces 100 seconds before P arrival
    earthmodel =  TauPyModel(model="ak135e")
    P_arrival = earthmodel.get_pierce_points(ev_depth,great_circle_dist,'P')
    # print(P_arrival)
    # print P_arrival[0].time 
    starttime = max([tr.stats.starttime for tr in stream])
    # starttime = starttime + P_arrival[0].time - 100. # 100 sec before P arrival
    endtime = min([tr.stats.endtime for tr in stream])
    stream.trim(starttime, endtime) # this gives a problem! I don not know why!!
    starttime = max([tr.stats.starttime for tr in stream])
    endtime = min([tr.stats.endtime for tr in stream])
    
    # # select max and min in a loop
    # for i, tr in enumerate(stream):
    #     print 'tr.stats.starttime',tr.stats.starttime
    #     if i == 0:
    #         starttime = tr.stats.starttime
    #         endtime = tr.stats.endtime
    #         # print '0 - starttime',starttime
    #     if tr.stats.starttime > starttime:
    #         starttime = tr.stats.starttime
    #     if tr.stats.endtime < endtime:
    #         endtime = tr.stats.endtime   
    # trim the signals
    # stream.trim(starttime,endtime)

    # for i, tr in enumerate(stream):
    #     print 'checking tr.stats.starttime',tr.stats.starttime       
                            
    # check that starttime is the real start time
    for i, tr in enumerate(stream):
        # print 'checking tr.stats.starttime',tr.stats.starttime
        if tr.stats.starttime > starttime:
            print 'i',i
            print 'tr.stats.starttime', tr.stats.starttime
            msg = "Specified stime %s is smaller than starttime %s in stream"
            raise ValueError(msg % (starttime, tr.stats.starttime))
        if tr.stats.endtime < endtime:
            msg = "Specified etime %s is bigger than endtime %s in stream"
            raise ValueError(msg % (endtime, tr.stats.endtime))

    #stream.attach_response(inventory)
    stream.merge()
    for tr in stream:
        for station in inventory[0].stations:
            if tr.stats.station == station.code:
                tr.stats.coordinates = \
                    AttribDict(dict(latitude=station.latitude,
                                    longitude=station.longitude,
                                    elevation=station.elevation))
                break

    # print stream
    spl = stream.copy()
   
    tmpdir = tempfile.mkdtemp(prefix="obspy-")
    filename_patterns = (os.path.join(tmpdir, 'pow_map_%03d.npy'),
                         os.path.join(tmpdir, 'apow_map_%03d.npy'))

    def dump(pow_map, apow_map, i):
        np.save(filename_patterns[0] % i, pow_map)
        np.save(filename_patterns[1] % i, apow_map)

    try:
        # next step would be needed if the correction velocity needs to be
        # estimated
        #
        sllx /= KM_PER_DEG
        slmx /= KM_PER_DEG
        slly /= KM_PER_DEG
        slmy /= KM_PER_DEG
        sls /= KM_PER_DEG
        vc = vel_corr
        if method == 'FK':
            print 'FK method'
            kwargs = dict(
                #slowness grid: X min, X max, Y min, Y max, Slow Step
                sll_x=sllx, slm_x=slmx, sll_y=slly, slm_y=slmy, sl_s=sls,
                # sliding window properties
                win_len=wlen, win_frac=0.5,
                # frequency properties
                frqlow=frqlow, frqhigh=frqhigh, prewhiten=0,
                # restrict output
                store=dump,
                semb_thres=-1e9, vel_thres=-1e9, verbose=False,
                timestamp='julsec', stime=starttime, etime=endtime,
                method=0, correct_3dplane=False, vel_cor=vc,
                static_3D=static3D)

            # here we do the array processing
            start = UTCDateTime()
            out = array_processing(stream, **kwargs)
            print "Array procesing total time in routine: %f\n" % (UTCDateTime() - start)
            # make output human readable, adjust backazimuth to values
            # between 0 and 360
            t, rel_power, abs_power, baz, slow = out.T
            
        else:
            print 'beamforming method'
            kwargs = dict(
                # slowness grid: X min, X max, Y min, Y max, Slow Step
                sll_x=sllx, slm_x=slmx, sll_y=slly, slm_y=slmy, sl_s=sls,
                # sliding window properties
                # frequency properties
                frqlow=frqlow, frqhigh=frqhigh,
                # restrict output
                store=dump,
                win_len=wlen, win_frac=0.5,
                nthroot=4, method=method,
                verbose=False, timestamp='julsec',
                stime=starttime, etime=endtime, vel_cor=vc,
                static_3D=False)

            # here we do the array processing
            start = UTCDateTime()
            out = beamforming(stream, **kwargs)
            print "Array procesing total time in routine: %f\n" % (UTCDateTime() - start)

            # make output human readable, adjust backazimuth to values
            # between 0 and 360
            trace = []
            t, rel_power, baz, slow_x, slow_y, slow = out.T
            # print 'baz',baz

            # calculating array response
        if array_response:
            stepsfreq = (frqhigh - frqlow) / 10.
            tf_slx = sllx
            tf_smx = slmx
            tf_sly = slly
            tf_smy = slmy
            transff = array_transff_freqslowness(
                stream, (tf_slx, tf_smx, tf_sly, tf_smy), sls, frqlow,
                frqhigh, stepsfreq, coordsys='lonlat',
                correct_3dplane=False, static_3D=False, vel_cor=vc)

        # now let's do the plotting
        cmap = cm.rainbow

        # we will plot everything in s/deg
        slow *= KM_PER_DEG
        sllx *= KM_PER_DEG
        slmx *= KM_PER_DEG
        slly *= KM_PER_DEG
        slmy *= KM_PER_DEG
        sls *= KM_PER_DEG

        numslice = len(t)
        powmap = []
        slx = np.arange(sllx-sls, slmx, sls)
        sly = np.arange(slly-sls, slmy, sls)
        if baz_plot:
            maxslowg = np.sqrt(slmx*slmx + slmy*slmy)
            bzs = np.arctan2(sls, np.sqrt(slmx*slmx + slmy*slmy))*180/np.pi
            xi = np.arange(0., maxslowg, sls)
            yi = np.arange(-180., 180., bzs)
            grid_x, grid_y = np.meshgrid(xi, yi)
        # reading in the rel-power maps
        for i in xrange(numslice):
            powmap.append(np.load(filename_patterns[0] % i))
            if method != 'FK':
                trace.append(np.load(filename_patterns[1] % i))

        npts = stream[0].stats.npts
        df = stream[0].stats.sampling_rate
        T = np.arange(0, npts / df, 1 / df)
        # extract the pahse to plot in the window
        phase_name_info = []
        phase_time_info = []
        slowness_info = []
        phases_list = ["P", "S", "ScS"] # Select the phases that you want to see
        get_phases_arrivals(spl[0],phases_list,ev_lat,ev_lon,ev_depth, \
                            phase_name_info,phase_time_info,slowness_info)
        # print 'phase_name_info',phase_name_info
        # print 'phase_time_info',phase_time_info
        # print 'slowness_info',slowness_info

        # if we choose windowlen > 0. we now move through our slices
        for i in xrange(numslice):
            print "loop over the time windows"
            slow_x = np.sin((baz[i]+180.)*np.pi/180.) * slow[i]
            slow_y = np.cos((baz[i]+180.)*np.pi/180.) * slow[i]
            st = UTCDateTime(t[i]) - starttime
            # print 'st',st
            if wlen <= 0:
                en = endtime
            else:
                en = st + wlen
            # print 'en',en
            # print UTCDateTime(t[i])
            # add polar and colorbar axes
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_axes([0.1, 0.87, 0.7, 0.10])
            # here we plot the first trace on top of the slowness map
            # and indicate the possibiton of the lsiding window as green box
            if method == 'FK':
                print("------------------------")
                print("method FK sliding window")
                # this one plots the first trace of the data base used!!
                # print'T',T
                spl_plot = spl[0]
                ax1.plot(spl_plot.times(), spl_plot.data, 'k')
                ax1.set_xlim(0., np.max(spl_plot.times()))
                # plot traveltimes
                ax1.scatter(phase_time_info,np.zeros(len(phase_time_info)),s=5000,marker=u'|',lw=4,color='g')
                # for i, txt in enumerate(phase_name_info):
                #     ax1.annotate(txt,(phase_time_info[i],0),fontsize=24,color='r')

                if wlen > 0.:
                    try:
                        ax1.axvspan(st, en, facecolor='g', alpha=0.3)
                    except IndexError:
                        pass
            else:
                T = np.arange(0, len(trace[i])/df, 1 / df)
                ax1.plot(T, trace[i],'k')

            ax1.yaxis.set_major_locator(MaxNLocator(3))

            ax = fig.add_axes([0.10, 0.1, 0.70, 0.7])

            # if we have chosen the baz_plot option a re-griding
            # of the sx,sy slowness map is needed
            if baz_plot:
                print('starting baz plot number: ' + str(i))
                slowgrid = []
                transgrid = []
                pow = np.asarray(powmap[i])
                for ix, sx in enumerate(slx):
                    for iy, sy in enumerate(sly):
                        bbaz = np.arctan2(sx, sy)*180/np.pi+180.
                        if bbaz > 180.:
                            bbaz = -180. + (bbaz-180.)
                        slowgrid.append((np.sqrt(sx*sx+sy*sy), bbaz,pow[ix, iy]))
                        if array_response:
                            tslow = (np.sqrt((sx+slow_x) * (sx+slow_x)+(sy+slow_y) * (sy+slow_y)))
                            tbaz = (np.arctan2(sx+slow_x, sy+slow_y) * 180 / np.pi + 180.)
                            if tbaz > 180.:
                                tbaz = -180. + (tbaz-180.)
                            transgrid.append((tslow, tbaz,transff[ix, iy]))

                slowgrid = np.asarray(slowgrid)
                sl = slowgrid[:, 0]
                bz = slowgrid[:, 1]
                slowg = slowgrid[:, 2]
                grid = spi.griddata((sl, bz), slowg, (grid_x, grid_y), method='nearest')
                slow_back_plot = ax.pcolormesh(xi, yi, grid, cmap=cmap)
                # slow_back_plot = ax.pcolormesh(xi, yi, grid, cmap=cmap, vmin=0., vmax=0.7) # set colorbar limits!
                # ax.contour(xi, yi, grid) #  contour lines
                colorbar_ax = fig.add_axes([0.83, 0.1, 0.05, 0.7]) #  add the colorbar
                cbar = fig.colorbar(slow_back_plot, cax=colorbar_ax)
                cbar.set_label('stacked amplitude', rotation=270, labelpad=40, y=0.45, fontsize=24)
                cbar.ax.tick_params(labelsize = 18) 
                ax.xaxis.set_tick_params(labelsize = 18)
                ax.yaxis.set_tick_params(labelsize = 18)

                # specify the limits that you want to see
                perc_see =  0.3 # percentage that you want to see
                y_low = (teor_baz-360)* (1-perc_see)
                y_high = (teor_baz-360)* (1+perc_see)

                # plot the theoretical backazimuth and theoretical slowness
                PLOT_SLOW_BACKAZ_LINES = False
                if PLOT_SLOW_BACKAZ_LINES:
                    ax.hold(True)
                    ax.plot((0.,14.),((teor_baz-360),(teor_baz-360)),'--',lw=4,color='k') # theoretical backazimuth line
                    for slowness_val in slowness_info: # slowness lines
                        ax.plot((slowness_val,slowness_val),(y_low,y_high),'--',lw=4,color='k')
                    for ii, txt in enumerate(phase_name_info): # print phase info
                        ax.annotate(txt,(slowness_info[ii],(teor_baz-360)),fontsize=40,color='r')

                # time_cont = 0
                # for p_time in phase_time_info:
                #     if st <= p_time <= en:
                #         # print 'p_time', p_time
                #         print 'time_cont', time_cont
                #         # ax.hold(True)
                #         ax.plot((slowness_info[time_cont],slowness_info[time_cont]),(y_low,y_high),lw=4,color='k')
                #         # ax.plot((0.,14.),((teor_baz-360),(teor_baz-360)),lw=4,color='k')
                #         time_cont += 1

                
                if array_response:
                    level = np.arange(0.1, 0.5, 0.1)
                    transgrid = np.asarray(transgrid)
                    tsl = transgrid[:, 0]
                    tbz = transgrid[:, 1]
                    transg = transgrid[:, 2]
                    trans = spi.griddata((tsl, tbz), transg,(grid_x, grid_y),method='nearest')
                    ax.contour(xi, yi, trans, level, colors='k',linewidth=0.2)
                
                ##################################
                # changing the axes of the plot! #
                ##################################
                ax.set_xlabel('slowness (s/deg)',fontsize=24)
                ax.set_ylabel('backazimuth (deg)',fontsize=24)
                # ax.set_xlim(xi[0], xi[-1])
                # ax.set_ylim(yi[0], yi[-1])
                # specify the limist that you want to see
                ax.set_ylim(y_low,y_high)
                
            else:
                print("if not baz plot")
                ax.set_xlabel('slowness (s/deg)')
                ax.set_ylabel('slowness (s/deg)')
                slow_x = np.cos((baz[i]+180.)*np.pi/180.)*slow[i]
                slow_y = np.sin((baz[i]+180.)*np.pi/180.)*slow[i]
                ax.pcolormesh(slx, sly, powmap[i].T)
                ax.arrow(0, 0, slow_y, slow_x, head_width=0.005,
                         head_length=0.01, fc='k', ec='k')
                if array_response:
                    tslx = np.arange(sllx+slow_x, slmx+slow_x+sls, sls)
                    tsly = np.arange(slly+slow_y, slmy+slow_y+sls, sls)
                    try:
                        ax.contour(tsly, tslx, transff.T, 5, colors='k',
                                   linewidth=0.5)
                    except:
                        pass
                ax.set_ylim(slx[0], slx[-1])
                ax.set_xlim(sly[0], sly[-1])

            new_time = t[i]

            # result = "BAZ: %.2f, Slow: %.2f s/deg, Time %s" % (baz[i], slow[i], UTCDateTime(new_time))
            result = "BAZ: %.2f, Slow: %.2f s/deg, Begining of Win %s s, Win length %s s" % (baz[i], slow[i], st, wlen)
            ax.set_title(result,fontsize=18)
            
            SAVE_FIG = False
            if SAVE_FIG:
                # firt create the MOVIE folder
                if i == 0:
                    # change the directory depending on your coordinate used
                    if component == 'Z':
                        dir = 'MOVIE_Z'
                    elif component == 'R':
                        dir = 'MOVIE_R'
                    elif component == 'T':
                        dir = 'MOVIE_T'
                    else:
                        raise ValueError('selected component not recognized!')

                    if os.path.exists(dir):
                        shutil.rmtree(dir)
                    os.makedirs(dir)

                # plt.tight_layout()
                # file_name = dir + '/movie_frame_' + str(i) + '.eps'
                file_name = dir + '/' + component + '_component_movie_frame_' + str(i) + '.png'
                plt.savefig(file_name)
                plt.clf() # clear the figure
                plt.close() # close the figure
            else:
                plt.show()

    finally:
        shutil.rmtree(tmpdir)

def array_processing(stream, win_len, win_frac, sll_x, slm_x, sll_y, slm_y,
                     sl_s, semb_thres, vel_thres, frqlow, frqhigh, stime,
                     etime, prewhiten, verbose=False, coordsys='lonlat',
                     timestamp='mlabday', method=0, correct_3dplane=False,
                     vel_cor=4., static_3D=False, store=None):
    
    print("ARRAY PROCESSING ROUTINE")
    
    """
    Method for FK-Analysis/Capon

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type win_len: Float
    :param win_len: Sliding window length in seconds
    :type win_frac: Float
    :param win_frac: Fraction of sliding window to use for step
    :type sll_x: Float
    :param sll_x: slowness x min (lower)
    :type slm_x: Float
    :param slm_x: slowness x max
    :type sll_y: Float
    :param sll_y: slowness y min (lower)
    :type slm_y: Float
    :param slm_y: slowness y max
    :type sl_s: Float
    :param sl_s: slowness step
    :type semb_thres: Float
    :param semb_thres: Threshold for semblance
    :type vel_thres: Float
    :param vel_thres: Threshold for velocity
    :type frqlow: Float
    :param frqlow: lower frequency for fk/capon
    :type frqhigh: Float
    :param frqhigh: higher frequency for fk/capon
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :type prewhiten: int
    :param prewhiten: Do prewhitening, values: 1 or 0
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :type method: int
    :param method: the method to use 0 == bf, 1 == capon
    :param vel_cor: correction velocity (upper layer) in km/s
    :param static_3D: a correction of the station height is applied using
        vel_cor the correction is done according to the formula:
        t = rxy*s - rz*cos(inc)/vel_cor
        where inc is defined by inv = asin(vel_cor*slow)
    :type store: function
    :param store: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.
    :return: numpy.ndarray of timestamp, relative relpow, absolute relpow,
        backazimut, slowness
    """
   
    BF, CAPON = 0, 1
    res = []
    eotr = True

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = ('in array-processing sampling rates of traces in stream are not equal')
        raise ValueError(msg)

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    #geometry = get_geometry(stream,coordsys=coordsys,correct_3dplane=correct_3dplane,verbose=verbose)

    geometry = get_geometry(stream,coordsys=coordsys,verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift(geometry, sll_x, sll_y, sl_s, grdpts_x,
                                     grdpts_y, vel_cor=vel_cor,
                                     static_3D=static_3D)
    # offset of arrays
    mini = np.min(time_shift_table[:, :, :])
    maxi = np.max(time_shift_table[:, :, :])

    spoint, _epoint = get_spoint(stream, stime, etime)

    # loop with a sliding window over the dat trace array and apply bbfk
    nstat = len(stream)
    fs = stream[0].stats.sampling_rate
    if win_len == -1.:
        nsamp = int((etime - stime)*fs)
        nstep = 1
    else:
        nsamp = int(win_len * fs)
        nstep = int(nsamp * win_frac)
    
    # print 'nsamp is ',nsamp
    # print 'nstep is ',nstep

    # generate plan for rfftr
    nfft = next_pow_2(nsamp)
    deltaf = fs / float(nfft)
    nlow = int(frqlow / float(deltaf) + 0.5)
    nhigh = int(frqhigh / float(deltaf) + 0.5)
    nlow = max(1, nlow)  # avoid using the offset
    nhigh = min(nfft / 2 - 1, nhigh)  # avoid using nyquist
    nf = nhigh - nlow + 1  # include upper and lower frequency

    # to spead up the routine a bit we estimate all steering vectors in advance
    steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype='c16')
    clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,deltaf, time_shift_table, steer)
    R = np.empty((nf, nstat, nstat), dtype='c16')
    ft = np.empty((nstat, nf), dtype='c16')
    newstart = stime
    tap = cosine_taper(nsamp, p=0.22) # 0.22 matches 0.2 of historical C bbfk.c

    offset = 0
    count = 0
    relpow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    abspow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')

    while eotr:
        try:
            for i, tr in enumerate(stream):
                dat = tr.data[spoint[i] + offset:spoint[i] + offset + nsamp]
                dat = (dat - dat.mean()) * tap
                ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]
        except IndexError:
            break
        ft = np.require(ft,'c16',['C_CONTIGUOUS'])
        relpow_map.fill(0.)
        abspow_map.fill(0.)
        # computing the covariances of the signal at different receivers
        dpow = 0.
        for i in xrange(nstat):
            for j in xrange(i, nstat):
                R[:, i, j] = ft[i, :] * ft[j, :].conj()
                if method == CAPON:
                    R[:, i, j] /= np.abs(R[:, i, j].sum())
                if i != j:
                    R[:, j, i] = R[:, i, j].conjugate()
                else:
                    dpow += np.abs(R[:, i, j].sum())
        dpow *= nstat
        if method == CAPON:
            # P(f) = 1/(e.H R(f)^-1 e)
            for n in xrange(nf):
                R[n, :, :] = np.linalg.pinv(R[n, :, :], rcond=1e-6)

        # errcode = clibsignal.generalizedBeamformer(relpow_map,abspow_map,steer,R,
        #                                            nsamp,nstat,prewhiten,grdpts_x,
        #                                            grdpts_y,nfft,nf,dpow,method)

        errcode = clibsignal.generalizedBeamformer(relpow_map,abspow_map,steer,R,nstat,prewhiten,
                                                   grdpts_x,grdpts_y,nf,dpow,method)

        if errcode != 0:
            msg = 'generalizedBeamforming exited with error %d'
            raise Exception(msg % errcode)
        ix, iy = np.unravel_index(relpow_map.argmax(), relpow_map.shape)
        relpow, abspow = relpow_map[ix, iy], abspow_map[ix, iy]
        if store is not None:
            store(relpow_map, abspow_map, count)
        count += 1

        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8

        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut % -360 + 180

        if relpow > semb_thres and 1. / slow > vel_thres:
            res.append(np.array([newstart.timestamp, relpow, abspow, baz,slow]))
            if verbose:
                print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
        if (newstart + (nsamp + nstep) / fs) > etime:
            eotr = False
        offset += nstep

        newstart += nstep / fs
    res = np.array(res)
    if timestamp == 'julsec':
        pass
    elif timestamp == 'mlabday':
        # 719162 == hours between 1970 and 0001
        res[:, 0] = res[:, 0] / (24. * 3600) + 719162
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
    return np.array(res)


def array_transff_freqslowness(stream, slim, sstep, fmin, fmax, fstep,
                               coordsys='lonlat', correct_3dplane=False,
                               static_3D=False, vel_cor=4.):
    """
    Returns array transfer function as a function of slowness difference and
    frequency.

    :type coords: numpy.ndarray
    :param coords: coordinates of stations in longitude and latitude in degrees
        elevation in km, or x, y, z in km
    :type coordsys: string
    :param coordsys: valid values: 'lonlat' and 'xy', choose which coordinates
        to use
    :param slim: either a float to use symmetric limits for slowness
        differences or the tupel (sxmin, sxmax, symin, symax)
    :type fmin: double
    :param fmin: minimum frequency in signal
    :type fmax: double
    :param fmin: maximum frequency in signal
    :type fstep: double
    :param fmin: frequency sample distance
    """

    geometry = get_geometry(stream, coordsys=coordsys,verbose=False)

    if isinstance(slim, float):
        sxmin = -slim
        sxmax = slim
        symin = -slim
        symax = slim
    elif isinstance(slim, tuple):
        if len(slim) == 4:
            sxmin = slim[0]
            sxmax = slim[1]
            symin = slim[2]
            symax = slim[3]
    else:
        raise TypeError('slim must either be a float or a tuple of length 4')

    nsx = int(np.ceil((sxmax + sstep / 10. - sxmin) / sstep))
    nsy = int(np.ceil((symax + sstep / 10. - symin) / sstep))
    nf = int(np.ceil((fmax + fstep / 10. - fmin) / fstep))

    transff = np.empty((nsx, nsy))
    buff = np.zeros(nf)

    for i, sx in enumerate(np.arange(sxmin, sxmax + sstep / 10., sstep)):
        for j, sy in enumerate(np.arange(symin, symax + sstep / 10., sstep)):
            for k, f in enumerate(np.arange(fmin, fmax + fstep / 10., fstep)):
                _sum = 0j
                for l in np.arange(len(geometry)):
                    _sum += np.exp(complex(
                        0., (geometry[l, 0] * sx + geometry[l, 1] * sy) *
                        2 * np.pi * f))
                buff[k] = abs(_sum) ** 2
            transff[i, j] = cumtrapz(buff, dx=fstep)[-1]

    transff /= transff.max()
    return transff


def beamforming(stream, sll_x, slm_x, sll_y, slm_y, sl_s, frqlow, frqhigh,
                stime, etime,   win_len=-1, win_frac=0.5,
                verbose=False, coordsys='lonlat', timestamp='mlabday',
                method="DLS", nthroot=1, store=None, correct_3dplane=False,
                static_3D=False, vel_cor=4.):
    """
    Method for Delay and Sum/Phase Weighted Stack/Whitened Slowness Power

    :param stream: Stream object, the trace.stats dict like class must
        contain a obspy.core.util.AttribDict with 'latitude', 'longitude' (in
        degrees) and 'elevation' (in km), or 'x', 'y', 'elevation' (in km)
        items/attributes. See param coordsys
    :type sll_x: Float
    :param sll_x: slowness x min (lower)
    :type slm_x: Float
    :param slm_x: slowness x max
    :type sll_y: Float
    :param sll_y: slowness y min (lower)
    :type slm_y: Float
    :param slm_y: slowness y max
    :type sl_s: Float
    :param sl_s: slowness step
    :type stime: UTCDateTime
    :param stime: Starttime of interest
    :type etime: UTCDateTime
    :param etime: Endtime of interest
    :type win_len: Float
    :param window length for sliding window analysis, default is -1 which means
        the whole trace;
    :type win_frac: Float
    :param fraction of win_len which is used to 'hop' forward in time
    :param coordsys: valid values: 'lonlat' and 'xy', choose which stream
        attributes to use for coordinates
    :type timestamp: string
    :param timestamp: valid values: 'julsec' and 'mlabday'; 'julsec' returns
        the timestamp in secons since 1970-01-01T00:00:00, 'mlabday'
        returns the timestamp in days (decimals represent hours, minutes
        and seconds) since '0001-01-01T00:00:00' as needed for matplotlib
        date plotting (see e.g. matplotlibs num2date)
    :type method: string
    :param method: the method to use "DLS" delay and sum; "PWS" phase weigted
        stack; "SWP" slowness weightend power spectrum
    :type nthroot: Float
    :param nthroot: nth-root processing; nth gives the root (1,2,3,4), default
        1 (no nth-root)
    :type store: function
    :param store: A custom function which gets called on each iteration. It is
        called with the relative power map and the time offset as first and
        second arguments and the iteration number as third argument. Useful for
        storing or plotting the map for each iteration. For this purpose the
        dump function of this module can be used.
    :type correct_3dplane: Boolean
    :param correct_3dplane: if Yes than a best (LSQ) plane will be fitted into
        the array geometry.
        Mainly used with small apature arrays at steep flanks
    :type static_3D: Boolean
    :param static_3D: if yes the station height of am array station is taken
        into account accoring the formula:
            tj = -xj*sxj - yj*syj + zj*cos(inc)/vel_cor
        the inc angle is slowness dependend and thus must
        be estimated for each grid-point:
            inc = asin(v_cor*slow)
    :type vel_cor: Float
    :param vel_cor: Velocity for the upper layer (static correction) in km/s
    :return: numpy.ndarray of timestamp, relative relpow, absolute relpow,
        backazimut, slowness, maximum beam (for DLS)
    """
    res = []
    eotr = True

    # check that sampling rates do not vary
    fs = stream[0].stats.sampling_rate
    nstat = len(stream)
    if len(stream) != len(stream.select(sampling_rate=fs)):
        msg = 'in sonic sampling rates of traces in stream are not equal'
        raise ValueError(msg)

    # loop with a sliding window over the dat trace array and apply bbfk

    grdpts_x = int(((slm_x - sll_x) / sl_s + 0.5) + 1)
    grdpts_y = int(((slm_y - sll_y) / sl_s + 0.5) + 1)

    abspow_map = np.empty((grdpts_x, grdpts_y), dtype='f8')
    # geometry = get_geometry(stream, coordsys=coordsys,
    #                         correct_3dplane=correct_3dplane, verbose=verbose)
    geometry = get_geometry(stream, coordsys=coordsys, verbose=verbose)

    if verbose:
        print("geometry:")
        print(geometry)
        print("stream contains following traces:")
        print(stream)
        print("stime = " + str(stime) + ", etime = " + str(etime))

    time_shift_table = get_timeshift(geometry, sll_x, sll_y, sl_s, grdpts_x,
                                     grdpts_y, vel_cor=vel_cor,
                                     static_3D=static_3D)

    mini = np.min(time_shift_table[:, :, :])
    maxi = np.max(time_shift_table[:, :, :])
    spoint, _epoint = get_spoint(stream, (stime-mini), (etime-maxi))
    minend = np.min(_epoint)
    maxstart = np.max(spoint)

    # recalculate the maximum possible trace length
    #    ndat = int(((etime-maxi) - (stime-mini))*fs)
    if(win_len < 0):
            nsamp = int(((etime-maxi) - (stime-mini))*fs)
    else:
        #nsamp = int((win_len-np.abs(maxi)-np.abs(mini)) * fs)
        nsamp = int(win_len * fs)

    if nsamp <= 0:
        print 'Data window too small for slowness grid'
        print 'Must exit'
        quit()

    nstep = int(nsamp * win_frac)

    stream.detrend()
    newstart = stime
    slow = 0.
    offset = 0
    count = 0
    while eotr:
        max_beam = 0.
        if method == 'DLS':
            for x in xrange(grdpts_x):
                for y in xrange(grdpts_y):
                    singlet = 0.
                    beam = np.zeros(nsamp, dtype='f8')
                    shifted = np.zeros(nsamp, dtype='f8')
                    for i in xrange(nstat):
                        s = spoint[i]+int(time_shift_table[i, x, y] * fs + 0.5)
                        try:
                            shifted = stream[i].data[s + offset:s + nsamp + offset]
                            if len(shifted) < nsamp:
                                shifted = np.pad(shifted,(0,nsamp-len(shifted)),'constant',constant_values=(0,1))
                            singlet += 1./nstat*np.sum(shifted*shifted)
                            beam += 1. / nstat * np.power(np.abs(shifted), 1. / nthroot) * shifted/np.abs(shifted)
                        except IndexError:
                            break
                    beam = np.power(np.abs(beam), nthroot) * beam / np.abs(beam)
                    bs = np.sum(beam*beam)
                    abspow_map[x, y] = bs / singlet
                    if abspow_map[x, y] > max_beam:
                        max_beam = abspow_map[x, y]
                        beam_max = beam
        if method == 'PWS':
            for x in xrange(grdpts_x):
                for y in xrange(grdpts_y):
                    singlet = 0.
                    beam = np.zeros(nsamp, dtype='f8')
                    stack = np.zeros(nsamp, dtype='c8')
                    phase = np.zeros(nsamp, dtype='f8')
                    shifted = np.zeros(nsamp, dtype='f8')
                    coh = np.zeros(nsamp, dtype='f8')
                    for i in xrange(nstat):
                        s = spoint[i] + int(time_shift_table[i, x, y] * fs +
                                            0.5)
                        try:
                            shifted = sp.signal.hilbert(stream[i].data[
                                s + offset: s + nsamp + offset])
                            if len(shifted) < nsamp:
                                shifted = np.pad(shifted,(0,nsamp-len(shifted)),'constant',constant_values=(0,1))
                        except IndexError:
                            break
                        phase = np.arctan2(shifted.imag, shifted.real)
                        stack.real += np.cos(phase)
                        stack.imag += np.sin(phase)
                    coh = 1. / nstat * np.abs(stack)
                    for i in xrange(nstat):
                        s = spoint[i]+int(time_shift_table[i, x, y] * fs + 0.5)
                        shifted = stream[i].data[s+offset: s + nsamp + offset]
                        singlet += 1. / nstat * np.sum(shifted * shifted)
                        beam += 1. / nstat * shifted * np.power(coh, nthroot)
                    bs = np.sum(beam*beam)
                    abspow_map[x, y] = bs / singlet
                    if abspow_map[x, y] > max_beam:
                        max_beam = abspow_map[x, y]
                        beam_max = beam
        if method == 'SWP':
            # generate plan for rfftr
            nfft = next_pow_2(nsamp)
            deltaf = fs / float(nfft)
            nlow = int(frqlow / float(deltaf) + 0.5)
            nhigh = int(frqhigh / float(deltaf) + 0.5)
            nlow = max(1, nlow)  # avoid using the offset
            nhigh = min(nfft / 2 - 1, nhigh)  # avoid using nyquist
            nf = nhigh - nlow + 1  # include upper and lower frequency

            beam = np.zeros((grdpts_x, grdpts_y, nf), dtype='f16')
            steer = np.empty((nf, grdpts_x, grdpts_y, nstat), dtype='c16')
            spec = np.zeros((nstat, nf), dtype='c16')
            time_shift_table *= -1.
            clibsignal.calcSteer(nstat, grdpts_x, grdpts_y, nf, nlow,
                                 deltaf, time_shift_table, steer)
            try:
                for i in xrange(nstat):
                    dat = stream[i].data[spoint[i] + offset:
                                         spoint[i] + offset + nsamp]
                    dat = (dat - dat.mean()) * tap
                    spec[i, :] = np.fft.rfft(dat, nfft)[nlow: nlow + nf]
            except IndexError:
                break

            for i in xrange(grdpts_x):
                for j in xrange(grdpts_y):
                    for k in xrange(nf):
                        for l in xrange(nstat):
                            steer[k, i, j, l] *= spec[l, k]

            beam = np.absolute(np.sum(steer, axis=3))
            less = np.max(beam, axis=1)
            max_buffer = np.max(less, axis=1)

            for i in xrange(grdpts_x):
                for j in xrange(grdpts_y):
                    abspow_map[i, j] = np.sum(beam[:, i, j] / max_buffer[:],
                                              axis=0) / float(nf)

            beam_max = stream[0].data[spoint[0] + offset:
                                      spoint[0] + nsamp + offset]

        ix, iy = np.unravel_index(abspow_map.argmax(), abspow_map.shape)
        abspow = abspow_map[ix, iy]
        if store is not None:
            store(abspow_map, beam_max, count)
        count += 1
        print count
        # here we compute baz, slow
        slow_x = sll_x + ix * sl_s
        slow_y = sll_y + iy * sl_s

        slow = np.sqrt(slow_x ** 2 + slow_y ** 2)
        if slow < 1e-8:
            slow = 1e-8
        azimut = 180 * math.atan2(slow_x, slow_y) / math.pi
        baz = azimut % -360 + 180
        res.append(np.array([newstart.timestamp, abspow, baz, slow_x, slow_y,slow]))
        if verbose:
            print(newstart, (newstart + (nsamp / fs)), res[-1][1:])
        if (newstart + (nsamp + nstep)/fs ) > etime:
            eotr = False
        offset += nstep

        newstart += nstep / fs
    res = np.array(res)
    if timestamp == 'julsec':
        pass
    elif timestamp == 'mlabday':
       # 719162 == hours between 1970 and 0001
        res[:, 0] = res[:, 0] / (24. * 3600) + 719162
    else:
        msg = "Option timestamp must be one of 'julsec', or 'mlabday'"
        raise ValueError(msg)
    return np.array(res)

#    return(baz,slow,slow_x,slow_y,abspow_map,beam_max)


def get_timeshift(geometry, sll_x, sll_y, sl_s, grdpts_x,grdpts_y, vel_cor=4.,static_3D=False):
    """
    Returns timeshift table for given array geometry

    :param geometry: Nested list containing the arrays geometry, as returned by
            get_group_geometry
    :param sll_x: slowness x min (lower)
    :param sll_y: slowness y min (lower)
    :param sl_s: slowness step
    :param grdpts_x: number of grid points in x direction
    :param grdpts_x: number of grid points in y direction
    :param vel_cor: correction velocity (upper layer) in km/s
    :param static_3D: a correction of the station height is applied using
        vel_cor the correction is done according to the formula:
        t = rxy*s - rz*cos(inc)/vel_cor
        where inc is defined by inv = asin(vel_cor*slow)
    """
    if static_3D:
        nstat = len(geometry)  # last index are center coordinates
        time_shift_tbl = np.empty((nstat, grdpts_x, grdpts_y), dtype="float32")
        for k in xrange(grdpts_x):
            sx = sll_x + k * sl_s
            for l in xrange(grdpts_y):
                sy = sll_y + l * sl_s
                slow = np.sqrt(sx*sx + sy*sy)
                if vel_cor*slow <= 1.:
                    inc = np.arcsin(vel_cor*slow)
                else:
                    print ("Warning correction velocity smaller than apparent "
                           "velocity")
                    inc = np.pi/2.
                time_shift_tbl[:, k, l] = sx * geometry[:, 0] + sy * \
                    geometry[:, 1] + geometry[:, 2] * np.cos(inc) / vel_cor
        return time_shift_tbl
    # optimized version
    else:
        mx = np.outer(geometry[:, 0], sll_x + np.arange(grdpts_x) * sl_s)
        my = np.outer(geometry[:, 1], sll_y + np.arange(grdpts_y) * sl_s)
        return np.require(
            mx[:, :, np.newaxis].repeat(grdpts_y, axis=2) +
            my[:, np.newaxis, :].repeat(grdpts_x, axis=1),
            dtype='float32')



def get_phases_arrivals(trace,phases_list,ev_lat,ev_lon,ev_depth, \
                        phase_name_info,phase_time_info,slowness_info):

    distance = locations2degrees(trace.stats.coordinates.latitude,trace.stats.coordinates.longitude,ev_lat,ev_lon)
    earthmodel =  TauPyModel(model="ak135e")
    #earthmodel =  TauPyModel(model="pwdk")

    # get traveltimes
    # arrivals = earthmodel.get_travel_times(distance_in_degree=distance,source_depth_in_km=ev_depth,phase_list=phases_list)
    arrivals = earthmodel.get_pierce_points(ev_depth,distance,phase_list=phases_list)
    # print'arrivals',arrivals
    count = 0
    for i_elem in arrivals:
        dummy_phase = arrivals[count]
        # print('dummy_phase_name',dummy_phase.name)
        # print('dummy_phase.time',dummy_phase.time)
        taup_phase_time = dummy_phase.time  # phase time in seconds
        taup_phase_slowness = dummy_phase.ray_param_sec_degree
        # print dummy_phase.time
        # print dummy_phase.ray_param_sec_degree
        # store the information
        phase_name_info.append(dummy_phase.name)
        phase_time_info.append(taup_phase_time)
        slowness_info.append(taup_phase_slowness)
        count += 1

    # return phase_name_info,phase_name_info

def rotate_seismograms(stream,ev,inv):

    '''
    ROTATE SEISMOGRAMS
    '''

    org = ev.preferred_origin() or ev.origins[0]
    ev_lat = org.latitude
    ev_lon = org.longitude

    # choose the maximun of the starting times
    starttime = max([tr.stats.starttime for tr in stream])
    # choose the ninumun of the ending times
    endtime = min([tr.stats.endtime for tr in stream])
      
    ## check that starttime is the real start time
    #for i, tr in enumerate(stream):
    #    if tr.stats.starttime > starttime:
    #        msg = "Specified stime %s is smaller than starttime %s in stream"
    #        raise ValueError(msg % (starttime, tr.stats.starttime))
    #    if tr.stats.endtime < endtime:
    #        msg = "Specified etime %s is bigger than endtime %s in stream"
    #        raise ValueError(msg % (endtime, tr.stats.endtime))

    # keep only the shortest window lenght of the whole seismograms
    stream.trim(starttime,endtime)

    seismo = stream
    sz = Stream() #  initialize variable
    net = len(inv.networks) # number of networks in the inventory file
    max_sampling_rate = 0.
    min_sampling_rate = 0.
    sampling_rate_count = 0. #  counter to determine if resampling is needed
    for tr in seismo:
        found = False
        for i_network in range(0,net-1): 
            if found: 
                break
            for station in inv[i_network].stations:
                if tr.stats.station == station.code:
                    # calculate the back azimuth for each station
                    great_circle_dist, baz, az2 = gps2dist_azimuth(station.latitude, \
                                                  station.longitude,ev_lat,ev_lon)
                    tr.stats.coordinates = \
                        AttribDict({'latitude': station.latitude,
                                    'longitude': station.longitude,
                                    'elevation': station.elevation})
                                    
                    tr.stats.back_azimuth = baz # assign the value of the backazimuth
                    found = True # exit the loop
                    if tr.stats.sampling_rate > max_sampling_rate:
                        sampling_rate_count += 1.
                        max_sampling_rate = tr.stats.sampling_rate
                    if tr.stats.sampling_rate < max_sampling_rate:
                        sampling_rate_count += 1.
                        min_sampling_rate = tr.stats.sampling_rate
                    if found:
                        break
        sz.append(tr)
    
    if sampling_rate_count > 1.: # resample the data
        sz.resample(min_sampling_rate)
    
    # rotate the seismograms
    sz.rotate('NE->RT')

    return sz
