import obspy
import imp
import sys
MAS = imp.load_source('MAS', '/Users/AnneSophieReiss/Documents/Doktorarbeit/OBSPY/ArraySeismology/Muenster_Array_Seismology_Vespagram.py')

# read the catalog information
cat = obspy.read_events("1998_12_27.xml")

# read the inventory
inventory = obspy.read_inventory('/Users/AnneSophieReiss/Documents/Doktorarbeit/OBSPY/ArraySeismology/INVENTORY_ALL.xml', format='STATIONXML') 

# read the waveforms
st = obspy.read("ROTATED_VELOCITY_SEISMOGRAMS_EVENT_DEC_27_1998_10s_MODEL_1.QHD")

selected_component = st.select(component="T")

#selected_component.plot()
# print(selected_component.__str__(extended=True))

# choose the maximun of the starting times
starttime = max([tr.stats.starttime for tr in selected_component])
# choose the ninumun of the ending times
endtime = min([tr.stats.endtime for tr in selected_component])
# keep only the shortest window lenght of the whole seismograms
selected_component.trim(starttime,endtime)
# remove the trend
selected_component.detrend('linear')

# selected_component.plot()
# filter the data
low_period = 3
high_period = 25
selected_component.filter("bandpass", freqmin=1./high_period, freqmax=1./low_period, corners=2, zerophase=True)

ev = cat[0]
#print ev

#vespagram plot
print MAS.vespagram(selected_component, ev, inventory, "DLS", scale=8, nthroot=4, sl=(1, 13, 0.1), phase_shift=-8.)
