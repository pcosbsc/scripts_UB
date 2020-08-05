import iris
import iris.plot as iplt
# import iris.quickplot as qplt
import iris.coord_categorisation
from iris.time import PartialDateTime
import iris.analysis.stats as istats
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import cm
from iris.cube import Cube
from iris.coords import DimCoord
# import cartopy.feature as cfeature
# from scipy import stats
import dask.array as da
import scipy.signal


def detrend(cube, dimension='time', method='linear'):
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


def trend(cube):
    cube_detr = detrend(cube)
    cube_trend_val = cube - cube_detr
    decades = 71/10
    cube_trends = (cube_trend_val[-1]-cube_trend_val[0])/decades
    return cube_trends


def appends(cube, cmap, title, vn, vx, lc=False):
    map.append(cube)
    color_map.append(cmap)
    titles.append(title)
    vmin.append(vn)
    vmax.append(vx)
    lcs.append(lc)


def lagged_correlation(cube, lag):

    end, start = 2010-lag, 1950+lag
    tnolag = PartialDateTime(year=end)
    tlag = PartialDateTime(year=start)
    # constrsain a lag and an unlagged cube.
    nolag = cube.extract(iris.Constraint(time=lambda t: t.point <= tnolag))
    lag = cube.extract(iris.Constraint(time=lambda t: tlag <= t.point))
    # define coords to unify the dimensions where the data lays.
    time = DimCoord(nolag.coord('time').points, standard_name='time')
    latitude = DimCoord(cube.coord('latitude').points, standard_name='latitude', units='degrees')
    longitude = DimCoord(cube.coord('longitude').points, standard_name='longitude', units='degrees')
    # create two cubes with lag btwn them but same coords.
    lag_cube = Cube(lag.data, dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)])
    nolag_cube = Cube(nolag.data, dim_coords_and_dims=[(time, 0), (latitude, 1), (longitude, 2)])
    # Calculate correlation
    corr_cube = istats.pearsonr(lag_cube, nolag_cube, corr_coords='time')
    return corr_cube


def main():

    cube = iris.load('~/code/develop/vipc/datasets/t2m_mon.era20cr_1950-2010.nc')[0]
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    yearly = cube.aggregated_by('year', iris.analysis.MEAN)
    clim = cube.collapsed('time', iris.analysis.MEAN)
    anomaly = yearly - clim
    anom_detr = detrend(anomaly)
    window = 11
    # appends(cube, cmap, title, vn, vx)
    # ---------------------------------------- #
    # ---------------1 OUTPUT----------------- #
    SDtot = anom_detr.collapsed('time', iris.analysis.STD_DEV)  # paint
    # appends(SDtot, plt.cm.Reds, 'SDtot', 0, 2.25)
    TRd = trend(anomaly)  # paint -negative values
    # appends(TRd, plt.cm.RdBu_r, 'TRd', -0.9, 0.9)
    TRd_SDtot = TRd / SDtot  # paint -negative values
    # appends(TRd_SDtot, plt.cm.RdBu_r, 'TRd/SDtot', -0.8, 0.8)
    # ---------------1 OUTPUT----------------- #
    # ---------------------------------------- #

    # ---------------------------------------- #
    # ---------------2 OUTPUT----------------- #
    anom_detr.remove_coord('year')
    corr_5y_lag = lagged_correlation(anom_detr, 5)
    corr_3y_lag = lagged_correlation(anom_detr, 3)
    corr_1y_lag = lagged_correlation(anom_detr, 1)
    appends(corr_5y_lag, plt.cm.RdBu_r, '5-year lagged correlation', -0.7, 0.7, lc=True)
    appends(corr_3y_lag, plt.cm.RdBu_r, '3-year lagged correlation', -0.7, 0.7, lc=True)
    appends(corr_1y_lag, plt.cm.RdBu_r, '1-year lagged correlation', -0.7, 0.7, lc=True)
    # ---------------2 OUTPUT----------------- #
    # ---------------------------------------- #

    # ---------------------------------------- #
    # ---------------3 OUTPUT----------------- #
    # Construct 2-year (24-month) and 7-year (84-month) low pass filters
    # for the SOI data which is monthly.
    wgts5 = low_pass_weights(window, 1. / 5.)
    wgts10 = low_pass_weights(window, 1. / 10.)
    soi5 = anom_detr.rolling_window('time', iris.analysis.MEAN, len(wgts5), weights=wgts5)
    soi10 = anom_detr.rolling_window('time', iris.analysis.MEAN, len(wgts10), weights=wgts10)
    # SD
    SD5 = soi5.collapsed('time', iris.analysis.STD_DEV)  # paint
    SD10 = soi10.collapsed('time', iris.analysis.STD_DEV)  # paint
    SD5p2_SDtotp2 = SD5**2/SDtot**2  # paint
    SD10p2_SDtotp2 = SD10**2/SDtot**2  # paint
    # appends(SD5, plt.cm.Reds, 'SD5', 0, 1.9)
    # appends(SD10, plt.cm.Reds, 'SD10', 0, 1.9)
    # appends(SD5p2_SDtotp2, plt.cm.Reds, 'SD$5^2$/SDtot$^2$', 0, 1)
    # appends(SD10p2_SDtotp2, plt.cm.Reds, 'SD$10^2$/SDtot$^2$', 0, 1)
    # ---------------3 OUTPUT----------------- #
    # ---------------------------------------- #
    # LOS LOLES
    corr_10y_lag_f = lagged_correlation(soi5, 10)
    corr_5y_lag_f = lagged_correlation(soi5, 5)
    corr_3y_lag_f = lagged_correlation(soi5, 3)
    appends(corr_10y_lag_f, plt.cm.RdBu_r, '10-year, 5y-filt, lagged correlation', -1, 1, lc=True)
    appends(corr_5y_lag_f, plt.cm.RdBu_r, '5-year, 5y-filt, lagged correlation', -1, 1, lc=True)
    appends(corr_3y_lag_f, plt.cm.RdBu_r, '3-year, 5y-filt, lagged correlation', -1, 1, lc=True)

    # ---------------------------------------- #
    # -----------------PLOT------------------- #
    proj = ccrs.PlateCarree(central_longitude=180)
    for i, (cube, cmp, t, vn, vx, lc) in enumerate(zip(map, color_map, titles, vmin, vmax, lcs)):
        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=proj)
        ax.set_aspect('auto')
        contour = iplt.pcolormesh(cube, vmin=vn, vmax=vx, coords=(
            'longitude', 'latitude'), cmap=cmp)
        # contour = iplt.pcolormesh(cube, coords=('longitude', 'latitude'), cmap=cmp)

        plt.title(t+', (1950-2010)')
        ax.coastlines('50m', linewidth=0.8)
        cb = plt.colorbar(contour, ax=ax, orientation="vertical")
        if lc:
            iplt.contour(cube, [-0.275], colors='k')
            iplt.contour(cube, [0.275], colors='k')
            cb.set_label('Correlation', size=15)
        else:
            cb.set_label('Temp. / '+str(cube.units), size=15)
        plt.tight_layout(h_pad=1)
        plt.savefig('/Users/Pep/code/develop/vipc/plots_e2/'+str(i)+'.eps')
    # plt.show()


if __name__ == '__main__':
    map = []
    color_map = []
    titles = []
    vmin = []
    vmax = []
    lcs = []
    main()
