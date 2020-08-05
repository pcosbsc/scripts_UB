import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.coord_categorisation
from iris.time import PartialDateTime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from iris.cube import Cube
from iris.coords import DimCoord
import cartopy.feature as cfeature
from scipy import stats
import dask.array as da
import scipy.signal
import mask_sealand as msk

from iris.coords import DimCoord
from iris.cube import Cube

def extract_region(cube, lat, lon):
    constr_lat = iris.Constraint(latitude=lambda y: lat[0] < y.point < lat[1])
    constr_lon = iris.Constraint(longitude=lambda x: lon[0] < x.point < lon[1])
    cube_out = cube.extract(constr_lon & constr_lat)
    return cube_out

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

def sorted(cube, coord):
    coord_to_sort = cube.coord(coord)
    assert coord_to_sort.ndim == 1, 'One dim coords only please.'
    dim, = cube.coord_dims(coord_to_sort)
    index = [slice(None)] * cube.ndim
    index[dim] = np.argsort(coord_to_sort.points)
    return cube[tuple(index)]

def main():
    # Load the monthly-valued Southern Oscillation Index (SOI) time-series.

    cube = iris.load('~/code/develop/vipc/datasets/NCEP-NCAR_t2m.mon.mean_1948-2019.nc')[0]
    cube.convert_units('degC')
    t1 = PartialDateTime(year=1969)
    constr_1 = iris.Constraint(time=lambda t: t1 < t.point)
    cube = cube.extract(constr_1)
    iris.coord_categorisation.add_year(cube, 'time', name='year')

    print(cube.coord('longitude').points)
    plt.figure()
    qplt.pcolormesh(cube[0])

    neg_lons = ((cube.coord('longitude').points + 180) % 360)-180
    cube = cube.interpolate([('longitude', neg_lons)], iris.analysis.Linear())
    cube = sorted(cube, 'longitude')
    print(cube.coord('longitude').points)
    plt.figure()
    qplt.pcolormesh(cube[0])
    plt.show()
    #cube.interpolate(sample_points, iris.analysis.Linear())
    # cube = cube.aggregated_by('year', iris.analysis.MEAN)
    # lat_bds = [0, 60]
    # lon_bds = [100, 250]
    # tas1 = cube #extract_region(cube, lat_bds, lon_bds)
    # tas = msk.mask_landsea(tas1, 'land')
    # print(tas[0].data)
    # detr_tas = detrend(tas)
    # trend = tas - detr_tas
    # nlat = tas.coord('latitude').shape[0]
    # nlon = tas.coord('longitude').shape[0]
    # lat = tas.coord('latitude').points
    # lon = tas.coord('longitude').points
    # slope = np.zeros([nlat, nlon])
    # for j in range(nlat):
    #     for k in range(nlon):
    #         slope[j,k] = (trend[-1,j,k].data-trend[0,j,k].data)/5
    # latitude = DimCoord(lat, standard_name='latitude', units='degrees')
    # longitude = DimCoord(lon, standard_name='longitude', units='degrees')
    # slope_cube = Cube(slope, dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
    #
    #
    # ts = trend.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
    #
    # print(ts.data)
    # ts_conv = ts*trend.coord('time').points/5
    # print(ts_conv.data)
    # plt.figure()
    # qplt.plot(ts)
    # plt.grid()
    # plt.figure()
    # qplt.pcolormesh(slope_cube, coords=('longitude','latitude'))
    # plt.gca().coastlines()
    # # plt.figure()
    # # plt.plot(np.arange(1971,2020,1), ts_conv.data)
    # # plt.grid()
    # plt.figure()
    # qplt.pcolormesh(trend.collapsed('time', iris.analysis.MEAN), coords=('longitude','latitude'))
    # plt.gca().coastlines()
    #
    # list = [trend, ts]
    # print(list[1])
    #
    # #### ------------ ####
    #
    # time_array = np.arange(1,tas.coord('time').shape[0]+1,1)
    # regr = np.zeros([nlat, nlon])
    # off = np.zeros([nlat, nlon])
    # for j in range(nlat):
    #     for k in range(nlon):
    #         # slope, intercept, r_value, p_value, std_err = stats.linregress(time_array, tas[:,j,k].data)
    #         # regr[j, k] = slope
    #         p = np.polyfit(time_array, tas[:,j,k].data, 1)
    #         regr[j, k] = p[0]
    #         off[j,k] = p[1]
    #
    # latitude = DimCoord(lat, standard_name='latitude', units='degrees')
    # longitude = DimCoord(lon, standard_name='longitude', units='degrees')
    # regr_cube = Cube(regr, dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
    # #### ----------- #####
    # regr_cube_msk = msk.mask_landsea(regr_cube, 'land')
    #
    # # iris comes complete with a method to put bounds on a simple point
    # # coordinate. This is very useful...
    # regr_cube_msk.coord('latitude').guess_bounds()
    # regr_cube_msk.coord('longitude').guess_bounds()
    #
    # # turn the iris Cube data structure into numpy arrays
    # gridlons = regr_cube_msk.coord('longitude').contiguous_bounds()
    # gridlats = regr_cube_msk.coord('latitude').contiguous_bounds()
    # data_tas = regr_cube_msk.data
    #
    # # set up a map
    #
    # # define the coordinate system that the grid lons and grid lats are on
    #
    #
    # plt.figure()
    # qplt.pcolormesh(regr_cube_msk, coords=('longitude','latitude'))
    # plt.gca().coastlines()
    #
    #
    #
    # plt.show()
    #
    # #
    # plt.figure()
    # proj=ccrs.PlateCarree()
    # ax = plt.axes(projection=proj)
    # fill = iplt.pcolormesh(regr, )
    # plt.gca().coastlines()
    # plt.show()
    # p = np.polyfit(soi.coord('time').points, soi.data, 10)
if __name__ == '__main__':
    main()
