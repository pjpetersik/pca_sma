#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import calendar
import xarray as xr
from datetime import datetime, timedelta
from sklearn.decomposition.pca import PCA
from mpl_toolkits.basemap import Basemap

def decimalYeartoDateTime(decimalYear):
    year = int(decimalYear)
    
    if calendar.isleap(year):
        resDays = int((decimalYear-year)*366)
    else:
        resDays = int((decimalYear-year)*365)
        
    return datetime(year,1,1) + timedelta(days=resDays-1)

def scale(x):
    return (x-x.mean())/x.std()

def scaleMax(x):
    return x/max(abs(x))

#%%  
    
plt.close("all")
    


data_path = "/home/paul/Dokumente/data/maio/ex3/GRACE_CSR_RL06_250km_2003_2016_nomean.nc"

d = xr.open_dataset(data_path,decode_times=False)

time = d['time']
lon =  d['lon']
lat = d['lat']
lon2, lat2 = np.meshgrid(lon,lat)

EWH = d['EWH']


#%%

DateIndex = 109
DateString = decimalYeartoDateTime(time[DateIndex]).strftime("%Y-%m-%d")

fig = plt.figure(figsize=(15,7))
plt.title("EWH, "+ DateString)
m = Basemap(projection='robin',lon_0=0,resolution='c')
x, y = m(lon2, lat2)


m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))
m.drawmapboundary(fill_color='white')
m.drawcoastlines()

norm = cm.colors.Normalize(vmax=-1, vmin=1.)
cmap = cm.bwr
cs = m.pcolormesh(x,y,EWH[DateIndex,:,:],cmap = cmap,norm=norm)

cb = m.colorbar(cs)

#%% EOF

EWH_EOFarr = np.array(EWH[:,:,:])

len_time = len(EWH_EOFarr[:,0,0])
len_lon = len(EWH_EOFarr[0,:,0])
len_lat = len(EWH_EOFarr[0,0,:])

EWH_EOFarr = EWH_EOFarr.reshape((len_time,len_lat*len_lon))


#%%
pca = PCA(n_components=149)
pca.fit(EWH_EOFarr)

print pca.explained_variance_ratio_[0:12]

#%% =============================================================================
# Plots
# =============================================================================

fig = plt.figure(figsize=(15,7))


for i in range(0,2):
    fig.add_subplot(221+i)
    plt.title("EOF"+str(i+1))
    m = Basemap(projection='robin',lon_0=0,resolution='c')
    x, y = m(lon2, lat2)
    
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,360.,60.))
    m.drawmapboundary(fill_color='white')
    m.drawcoastlines()
    
    norm = cm.colors.Normalize(vmax=-1, vmin=1.)
    cmap = cm.bwr
    cs = m.pcolormesh(x,y,pca.components_[i,:].reshape(len_lon,len_lat),cmap = cmap,norm=norm)
    
    cb = m.colorbar(cs)

    
for i in range(0,2):
    fig.add_subplot(223+i)
    projection = np.matmul(EWH_EOFarr,pca.components_[i,:])

    
    plt.plot(time,projection)
    

#%%
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("EOF")
plt.ylabel("cumulative explained variance")


#%%
EWH_from_EOF = xr.zeros_like(EWH[:,:,:])

for t in range(len_time):
    for i in range(8):
        projection = np.matmul(EWH_EOFarr,pca.components_[i,:])
        EWH_from_EOF[t,:,:] = EWH_from_EOF[t,:,:] + projection[t] * pca.components_[i,:].reshape(len_lon,len_lat)

#%%
fig = plt.figure(figsize=(15,7))

plt.title("Reconstructed data from EOF1-8, " +DateString)
m = Basemap(projection='robin',lon_0=0,resolution='c')
x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))
m.drawmapboundary(fill_color='white')
m.drawcoastlines()


cmap = cm.bwr
cs = m.pcolormesh(x,y,EWH_from_EOF[DateIndex,:,:],cmap = cmap,norm=norm)

cb = m.colorbar(cs)

#%%
fig = plt.figure(figsize=(15,7))

plt.title("Difference between reconstructed data and real data, "+DateString)
m = Basemap(projection='robin',lon_0=0,resolution='c')
x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90.,120.,30.))
m.drawmeridians(np.arange(0.,360.,60.))
m.drawmapboundary(fill_color='white')
m.drawcoastlines()


cmap = cm.bwr
cs = m.pcolormesh(x,y,EWH_from_EOF[DateIndex,:,:]-EWH[DateIndex,:,:],cmap = cmap,norm=norm)

cb = m.colorbar(cs)


#%%
varData= np.var(EWH)
varEOF = np.var(EWH_from_EOF)
print "Varaince measurements: " + str(np.array(varData))
print "Varaince reconstructed: " + str(np.array(varEOF))

#%%
lat_index = np.where(lat==65.5)[0][0]
lon_index = np.where(lon==-25.5)[0][0]
plt.figure()
plt.plot(time,EWH[:,lat_index,lon_index],label="EWH measurements")
plt.plot(time,EWH_from_EOF[:,lat_index,lon_index],label="EWH reconstructed")

plt.xlabel("time")
plt.ylabel("EWH")