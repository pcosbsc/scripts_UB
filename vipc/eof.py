"""
Compute and plot the leading EOF of sea surface temperature in the
central and northern Pacific during winter time.

The spatial pattern of this EOF is the canonical El Nino pattern, and
the associated time series shows large peaks and troughs for well-known
El Nino and La Nina events.

This example uses the metadata-retaining iris interface.

Additional requirements for this example:

    * iris (http://scitools.org.uk/iris/)
    * matplotlib (http://matplotlib.org/)
    * cartopy (http://scitools.org.uk/cartopy/)

"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import iris
import iris.coord_categorisation
import iris.plot as iplt
import iris.quickplot as qplt
import matplotlib.pyplot as plt
import numpy as np

from eofs.iris import Eof


import dask.array as da
import scipy.signal

def detrend(cube, dimension='time', method='linear'):
    """
    Detrend data along a given dimension.
    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    dimension: str
        Dimension to detrend
    method: str
        Method to detrend. Available: linear, constant. See documentation of
        'scipy.signal.detrend' for details
    Returns
    -------
    iris.cube.Cube
        Detrended cube
    """
    coord = cube.coord(dimension)
    axis = cube.coord_dims(coord)[0]
    detrended = da.apply_along_axis(
        scipy.signal.detrend,
        axis=axis,
        arr=cube.lazy_data(),
        type=method,
        shape=(cube.shape[axis],)
    )
    return cube.copy(detrended)

def extract_region(cube, lat, lon):
    constr_lat = iris.Constraint(latitude=lambda y: lat[0] < y.point < lat[1])
    constr_lon = iris.Constraint(longitude=lambda x: lon[0] < x.point < lon[1])
    cube_out = cube.extract(constr_lon & constr_lat)
    return cube_out

def plotter(eof,pc,var,num):
    # Plot the leading EOF expressed as correlation in the Pacific domain.
    plt.figure()
    clevs = np.linspace(-1, 1, 11)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=190))
    fill = iplt.contourf(eof, clevs, cmap=plt.cm.RdBu_r)
    ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
    cb = plt.colorbar(fill, orientation='horizontal')
    cb.set_label('correlation coefficient', fontsize=12)
    variance = str("%.2f" % var)
    ax.set_title('EOF'+str(num)+' ('+variance+'%) expressed as correlation', fontsize=16)
    plt.savefig('EOF'+str(num)+'.eps')
    # Plot the leading PC time series.
    plt.figure()
    iplt.plot(pc, color='b', linewidth=2)
    ax = plt.gca()
    ax.axhline(0, color='k')
    ax.set_ylim(-20, 20)
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized Units')
    ax.set_title('PC'+str(num)+' Time Series', fontsize=16)
    plt.savefig('PC'+str(num)+'.eps')


# Read SST anomalies using the iris module. The file contains November-March
# averages of SST anomaly in the central and northern Pacific.
filename = 'ERSSTv4_1950-2010.nc'
cube = iris.load('~/code/develop/vipc/datasets/'+filename, 'Monthly Means of Sea Surface Temperature')[0]
lat_bds = [10, 70]
lon_bds = [110, 260]
sst = extract_region(cube, lat_bds, lon_bds)
var = sst.collapsed('time', iris.analysis.VARIANCE)
std = sst.collapsed('time', iris.analysis.STD_DEV)
plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=190))
fill = iplt.contourf(var, cmap=plt.cm.RdBu_r)
ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
cb = plt.colorbar(fill, orientation='horizontal')
cb.set_label('VARIANCE / K^2 ', fontsize=12)
plt.savefig('var.eps')

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=190))
fill = iplt.contourf(std, cmap=plt.cm.RdBu_r)
ax.add_feature(cfeature.LAND, zorder=100, edgecolor='k')
cb = plt.colorbar(fill, orientation='horizontal')
cb.set_label('STD_DEV / K', fontsize=12)
plt.savefig('std.eps')
# sst_detrend = detrends(sst)
iris.coord_categorisation.add_year(sst, 'time', name='year')
yr_sst = sst.aggregated_by('year', iris.analysis.MEAN)
clim_sst = yr_sst.collapsed('time', iris.analysis.MEAN)

# yr_sst_detrend = detrend(yr_sst)
# clim_sst_detrend = yr_sst_detrend.collapsed('time', iris.analysis.MEAN)

anom_sst = yr_sst - clim_sst
anom_sst_detrend = detrend(anom_sst)



# Create an EOF solver to do the EOF analysis. Square-root of cosine of
# latitude weights are applied before the computation of EOFs.
# sovler = Eof(anom_sst_detrend) #--> NO FUNCIONA PQ yr_sst_detrend SEMBLA QUE JA ESTÀ FETA LA ANOMALIA. BUENO CLAR PUTO,, SI LI TREUS LA REGRESSIÓ LINEAL ET CARREGUES TOT.
solver = Eof(anom_sst, weights='coslat')

# Retrieve the leading EOF, expressed as the correlation between the leading
# PC time series and the input SST anomalies at each grid point, and the
# leading PC time series itself.
n = 3
# eofs = solver.eofsAsCorrelation(neofs=n)
eofs = solver.eofsAsCorrelation(neofs=n)
pcs = solver.pcs(npcs=n)
variance_fractions = solver.varianceFraction(neigs=n)
print(variance_fractions.data)

for i in range(n):
    plotter(eofs[i],pcs[:,i],variance_fractions.data[i]*100,i+1)

# plt.show()
