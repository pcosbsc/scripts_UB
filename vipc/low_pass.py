"""
Applying a filter to a time-series
==================================

This example demonstrates low pass filtering a time-series by applying a
weighted running mean over the time dimension.

The time-series used is the Darwin-only Southern Oscillation index (SOI),
which is filtered using two different Lanczos filters, one to filter out
time-scales of less than two years and one to filter out time-scales of
less than 7 years.

References
----------

    Duchon C. E. (1979) Lanczos Filtering in One and Two Dimensions.
    Journal of Applied Meteorology, Vol 18, pp 1016-1022.

    Trenberth K. E. (1984) Signal Versus Noise in the Southern Oscillation.
    Monthly Weather Review, Vol 112, pp 326-332

"""

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

def extract_region(cube, lat, lon):
    constr_lat = iris.Constraint(latitude=lambda y: lat[0] < y.point < lat[1])
    constr_lon = iris.Constraint(longitude=lambda x: lon[0] < x.point < lon[1])
    cube_out = cube.extract(constr_lon & constr_lat)
    return cube_out

def main():
    # Load the monthly-valued Southern Oscillation Index (SOI) time-series.

    cube = iris.load('~/code/develop/vipc/datasets/NCEP-NCAR_t2m.mon.mean_1948-2019.nc')[0]
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    cube = cube.aggregated_by('year', iris.analysis.MEAN)
    lat_bds = [25, 50]
    lon_bds = [0, 60]
    tas = extract_region(cube, lat_bds, lon_bds)
    nlat = tas.coord('latitude').shape[0]
    nlon = tas.coord('longitude').shape[0]
    lat = tas.coord('latitude').points
    lon = tas.coord('longitude').points
    print(tas.coord('time').shape[0])


    # qplt.pcolormesh(tas[1])
    # plt.gca().coastlines()
    # plt.show()

    time_array = np.arange(1,tas.coord('time').shape[0]+1,1)
    print(time_array.shape)
    regr = np.zeros([nlat, nlon])
    for j in range(nlat):
        for k in range(nlon):
            p = np.polyfit(time_array, tas[:,j,k].data, 1)
            regr[j, k] = p[0]
    print(regr)

    latitude = DimCoord(lat, standard_name='latitude', units='degrees')
    longitude = DimCoord(lon, standard_name='longitude', units='degrees')
    regr_cube = Cube(regr, dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
    print(regr_cube)

    plt.figure()
    proj=ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    fill = iplt.pcolormesh(regr, )
    plt.gca().coastlines()
    plt.show()
    p = np.polyfit(soi.coord('time').points, soi.data, 10)

    # iris.coord_categorisation.add_year(cube, 'time', name='year')
    # soi = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN)
    # mn = soi.aggregated_by('year', iris.analysis.MEAN)
    # trend = mn.rolling_window('time', iris.analysis.MEAN, 20)
    # val = np.array(trend[0])
    # p = np.polyfit(soi.coord('time').points, soi.data, 10)
    # y = np.polyval(p, soi.coord('time').points)
    # # long_name = 'polyfit'
    # # fit = iris.coords.AuxCoord(y, long_name=long_name,
    # #                            units=soi.units)
    # # soi.add_aux_coord(fit, 0)
    # #mn = mn.data-soi.coord(long_name)
    #
    #
    # normalized = soi-y
    # print(normalized)
    # #qplt.plot(soi)
    # # qplt.plot(normalized)
    # #qplt.plot(soi.coord('time'), soi.coord(long_name))
    #
    # # Window length for filters.
    # window = 121
    #
    # # Construct 2-year (24-month) and 7-year (84-month) low pass filters
    # # for the SOI data which is monthly.
    # wgts24 = low_pass_weights(window, 1. / 24.)
    # wgts84 = low_pass_weights(window, 1. / 84.)
    # # Apply each filter using the rolling_window method used with the weights
    # # keyword argument. A weighted sum is required because the magnitude of
    # # the weights are just as important as their relative sizes.
    # soi24 = normalized.rolling_window('time', iris.analysis.MEAN, len(wgts24), weights=wgts24)
    # soi84 = normalized.rolling_window('time', iris.analysis.MEAN, len(wgts84), weights=wgts84)
    # std_dev_norm = normalized.aggregated_by('time', iris.analysis.STD_DEV)
    # print(std_dev_norm)
    #
    # # Plot the SOI time series and both filtered versions.
    # plt.figure(figsize=(9, 4))
    # iplt.plot(normalized, color='0.7', linewidth=1., linestyle='-', alpha=1., label='no filter')
    # # plt.fill_between(soi24+std_dev_norm, soi23-std_dev_norm, color='0.7',label='std_dev')
    # iplt.plot(soi24, color='b', linewidth=2., linestyle='-', alpha=.7, label='2-year filter')
    # iplt.plot(soi84, color='r', linewidth=2., linestyle='-', alpha=.7, label='7-year filter')
    # plt.title('T2m')
    # plt.xlabel('Time')
    # plt.ylabel('T [K]')
    # plt.legend(fontsize=10)
    # #plt.savefig('MEAN.eps')
    # iplt.show()


if __name__ == '__main__':
    main()
