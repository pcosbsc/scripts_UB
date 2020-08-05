import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.coord_categorisation
from iris.time import PartialDateTime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

def low_pass_weights(window, cutoff):
    """
    Calculate weights for a low pass Lanczos filter.
    Method borrowed from `iris example
    <https://scitools.org.uk/iris/docs/latest/examples/General/
    SOI_filtering.html?highlight=running%20mean>`_
    Parameters
    ----------
    window: int
        The length of the filter window.
    cutoff: float
        The cutoff frequency in inverse time steps.
    Returns
    -------
    list:
        List of floats representing the weights.
    """
    order = ((window - 1) // 2) + 1
    nwts = 2 * order + 1
    weights = np.zeros([nwts])
    half_order = nwts // 2
    weights[half_order] = 2 * cutoff
    kidx = np.arange(1., half_order)
    sigma = np.sin(np.pi * kidx / half_order) * half_order / (np.pi * kidx)
    firstfactor = np.sin(2. * np.pi * cutoff * kidx) / (np.pi * kidx)
    weights[(half_order - 1):0:-1] = firstfactor * sigma
    weights[(half_order + 1):-1] = firstfactor * sigma

    return weights[1:-1]


def timeseries_filter(cube, window, span, filter_type='lowpass', filter_stats='sum'):
    """
    Apply a timeseries filter.
    Method borrowed from `iris example
    <https://scitools.org.uk/iris/docs/latest/examples/General/
    SOI_filtering.html?highlight=running%20mean>`_
    Apply each filter using the rolling_window method used with the weights
    keyword argument. A weighted sum is required because the magnitude of
    the weights are just as important as their relative sizes.
    See also the `iris rolling window
    <https://scitools.org.uk/iris/docs/v2.0/iris/iris/
    cube.html#iris.cube.Cube.rolling_window>`_
    Parameters
    ----------
    cube: iris.cube.Cube
        input cube.
    window: int
        The length of the filter window (in units of cube time coordinate).
    span: int
        Number of months/days (depending on data frequency) on which
        weights should be computed e.g. 2-yearly: span = 24 (2 x 12 months).
        Span should have same units as cube time coordinate.
    filter_type: str, optional
        Type of filter to be applied; default 'lowpass'.
        Available types: 'lowpass'.
    filter_stats: str, optional
        Type of statistic to aggregate on the rolling window; default 'sum'.
        Available operators: 'mean', 'median', 'std_dev', 'sum', 'min', 'max'
    Returns
    -------
    iris.cube.Cube
        cube time-filtered using 'rolling_window'.
    Raises
    ------
    iris.exceptions.CoordinateNotFoundError:
        Cube does not have time coordinate.
    NotImplementedError:
        If filter_type is not implemented.
    """
    try:
        cube.coord('time')
    except iris.exceptions.CoordinateNotFoundError:
        logger.error("Cube %s does not have time coordinate", cube)
        raise

    # Construct weights depending on frequency
    # TODO implement more filters!
    # supported_filters = ['lowpass', ]
    # if filter_type in supported_filters:
    #     if filter_type == 'lowpass':
    #         wgts = low_pass_weights(window, 1. / span)
    # else:
    #     raise NotImplementedError(
    #         "Filter type {} not implemented, \
    #         please choose one of {}".format(filter_type,
    #                                         ", ".join(supported_filters)))

    # Apply filter
    aggregation_operator = getattr(iris.analysis, filter_stats.upper())
    cube = cube.rolling_window('time',
                               aggregation_operator,
                               window)

    return cube

def main():
    cube = iris.load('~/code/develop/vipc/datasets/NOAA20CRv2_t2m.mon.mean_2.5x2.5.nc', 'air_temperature')[0]
    iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(cube, 'time', name='year_season')
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    print(cube.coord('year'))
    mean = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN)
    window = 10*12 #5*12 #24
    span = 24
    flt_mean = timeseries_filter(mean, window, span, filter_type='lowpass', filter_stats='mean')
    qplt.plot(mean)
    qplt.plot(flt_mean)
    plt.show()

if __name__ == '__main__':
    main()
