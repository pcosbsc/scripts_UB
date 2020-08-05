import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.coord_categorisation
# from iris.time import PartialDateTime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
#
# from iris.cube import Cube
# from iris.coords import DimCoord
# import cartopy.feature as cfeature
# from scipy import stats
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


def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    """
    order = ((window - 1) // 2) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]


def plotter_2D(cube_array, name):
    proj = ccrs.PlateCarree(central_longitude=180)
    vmin = [0, 0]
    vmax = [4, 2]
    title = ['Variance', 'Standard Deviation']
    color = [plt.cm.Reds, plt.cm.Reds]
    for cube, vn, vx, t, cl in zip(cube_array, vmin, vmax, title, color):
        plt.figure(figsize=(9, 6))
        ax = plt.axes(projection=proj)
        plt.gca().coastlines()
        contour = iplt.pcolormesh(cube, vmin=vn, vmax=vx, coords=('longitude', 'latitude'), cmap=cl)
        plt.title(t)
        cb = plt.colorbar(contour, ax=ax, orientation="horizontal")
        cb.set_label('Temperature / '+str(cube.units), size=15)
        # plt.savefig(str(i)+name+'plot.jpg')


def trend(cube):
    cube_detr = detrend(cube)
    cube_trend_val = cube - cube_detr
    decades = 71/10
    cube_trends = (cube_trend_val[-1]-cube_trend_val[0])/decades
    return cube_trends


def main():

    cube = iris.load('~/code/develop/vipc/datasets/t2m_mon.era20cr_1950-2010.nc')[0]
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    yearly = cube.aggregated_by('year', iris.analysis.MEAN)
    clim = cube.collapsed('time', iris.analysis.MEAN)
    anomaly = yearly - clim
    anom_detr = detrend(anomaly)

    # ---------------------------------------- #
    # ---------------1 OUTPUT----------------- #
    SDtot = anom_detr.collapsed('time', iris.analysis.STD_DEV)
    TRd = trend(anomaly)
    TRd_SDtot = TRd / SDtot
    # ---------------1 OUTPUT----------------- #
    # ---------------------------------------- #

    # ---------------------------------------- #
    # ---------------2 OUTPUT----------------- #
    anom_detr
    # ---------------2 OUTPUT----------------- #
    # ---------------------------------------- #

    # ---------------------------------------- #
    # ---------------3 OUTPUT----------------- #
    window = 11
    # Construct 2-year (24-month) and 7-year (84-month) low pass filters
    # for the SOI data which is monthly.
    wgts5 = low_pass_weights(window, 1. / 5.)
    wgts10 = low_pass_weights(window, 1. / 10.)
    soi5 = anom_detr.rolling_window('time', iris.analysis.MEAN, len(wgts5), weights=wgts5)
    soi10 = anom_detr.rolling_window('time', iris.analysis.MEAN, len(wgts10), weights=wgts10)
    # SD
    SD5 = soi5.collapsed('time', iris.analysis.STD_DEV)
    SD10 = soi10.collapsed('time', iris.analysis.STD_DEV)
    SD5p2_SDtotp2 = SD5**2/SDtot**2
    SD10p2_SDtotp2 = SD10**2/SDtot**2
    # ---------------3 OUTPUT----------------- #
    # ---------------------------------------- #

    p = np.polyfit(anom_ts.coord('time').points, anom_ts.data, 1)
    y = np.polyval(p, anom_ts.coord('time').points)
    long_name_1 = 'linear_regression_d1'
    trend_1 = iris.coords.AuxCoord(y, long_name=long_name_1, units=anom_ts.units)
    anom_ts.add_aux_coord(trend_1, 0)

    pdiv_sd = [p[0]/anom_mn_glob.data, p[1]]
    ydiv_sd = np.polyval(pdiv_sd, anom_ts.coord('time').points)
    long_name_2 = 'linear_regression_div_sd_d1'
    trend_2 = iris.coords.AuxCoord(ydiv_sd, long_name=long_name_2, units=None)
    anom_ts.add_aux_coord(trend_2, 0)

    slope = np.polyfit(np.arange(1950, 2011, 1), anom_ts.data, 1)
    decadal_trend = slope[0]*10

    plt.figure(figsize=(9, 4))
    iplt.plot(anom_ts.coord('time'), anom_ts.coord(long_name_1))
    # iplt.plot(anom_ts.coord('time'), anom_ts.coord(long_name_2))
    iplt.plot(anom_detr_ts)
    iplt.plot(anom_ts)
    plt.grid()
    plt.title('Anomaly of global t2m mean (trend = {0:.3f} $^o$C / Decade)'.format(decadal_trend))
    plt.tight_layout()
    # plotter_2D([anom_var, anom_std], 't2m')

    # # # # # # # # # # # # # # # #
    # ----- begin - lagging ----- #
    # # # # # # # # # # # # # # # #

    # latitude = DimCoord(lat, standard_name='latitude', units='degrees')
    # longitude = DimCoord(lon, standard_name='longitude', units='degrees')
    # regr_cube = Cube(regr, dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
    # # # # # # # # # # # # # # # #
    # ------ end - lagging ------ #
    # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # #
    # ---- begin - filtering ---- #
    # # # # # # # # # # # # # # # #
    # Window length for filters.
    window = 11
    # Construct 2-year (24-month) and 7-year (84-month) low pass filters
    # for the SOI data which is monthly.
    wgts5 = low_pass_weights(window, 1. / 5.)
    wgts10 = low_pass_weights(window, 1. / 10.)
    # Apply each filter using the rolling_window method used with the weights
    # keyword argument. A weighted sum is required because the magnitude of
    # the weights are just as important as their relative sizes.
    soi5 = anom_detr_ts.rolling_window('time', iris.analysis.MEAN, len(wgts5), weights=wgts5)
    soi10 = anom_detr_ts.rolling_window('time', iris.analysis.MEAN, len(wgts10), weights=wgts10)
    soi5_2D = anom_detr.rolling_window('time', iris.analysis.MEAN, len(
        wgts5), weights=wgts5).collapsed('time', iris.analysis.MEAN)
    soi10_2D = anom_detr.rolling_window('time', iris.analysis.MEAN, len(
        wgts10), weights=wgts10).collapsed('time', iris.analysis.MEAN)
    std_dev_5 = soi5.collapsed('time', iris.analysis.STD_DEV)
    std_dev_10 = soi10.collapsed('time', iris.analysis.STD_DEV)
    std_dev_5_2D = anom_detr.rolling_window('time', iris.analysis.MEAN, len(
        wgts5), weights=wgts5).collapsed('time', iris.analysis.STD_DEV)
    std_dev_10_2D = anom_detr.rolling_window('time', iris.analysis.MEAN, len(
        wgts5), weights=wgts5).collapsed('time', iris.analysis.STD_DEV)
    std_dev_noft = anom_detr_ts.collapsed('time', iris.analysis.STD_DEV)
    print('sd no filt: '+str(std_dev_noft.data))
    print('sd 5-year filter: '+str(std_dev_5.data))
    print('sd 10-year filter: '+str(std_dev_10.data))
    # plots for filtered sd
    plt.figure(figsize=(9, 4))
    qplt.pcolormesh(soi5_2D)

    sd_coc_5 = (std_dev_5_2D ** 2)/(anom_std ** 2)
    sd_coc_10 = (std_dev_10_2D ** 2)/(anom_std ** 2)
    plt.figure(figsize=(9, 4))
    qplt.pcolormesh(sd_coc_5)
    print(sd_coc_5)

    # Plot the SOI time series and both filtered versions.
    plt.figure(figsize=(9, 4))
    iplt.plot(anom_detr_ts, color='0.7', linewidth=1., linestyle='-', alpha=1., label='no filter')
    # plt.fill_between(soi24+std_dev_norm, soi23-std_dev_norm, color='0.7',label='std_dev')
    iplt.plot(soi5, color='b', linewidth=2., linestyle='-', alpha=.7, label='5-year filter')
    iplt.plot(soi10, color='r', linewidth=2., linestyle='-', alpha=.7, label='10-year filter')
    plt.title('T2m')
    plt.xlabel('Time')
    plt.ylabel('T [K]')
    plt.legend(fontsize=10)
    plt.grid()
    # plt.savefig('MEAN.eps')
    # # # # # # # # # # # # # # # #
    # ----- end - filtering ----- #
    # # # # # # # # # # # # # # # #

    iplt.show()


if __name__ == '__main__':
    main()
