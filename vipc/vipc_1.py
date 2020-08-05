import iris
import iris.plot as iplt
import iris.quickplot as qplt
import iris.coord_categorisation
from iris.time import PartialDateTime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

def cube_extract (cube, season):
    t1 = PartialDateTime(year=1871, month=2)
    t2 = PartialDateTime(year=2012, month=12)
    # t1 = PartialDateTime(year=1948, month=2)
    # t2 = PartialDateTime(year=2019, month=12)
    constr_1 = iris.Constraint(time=lambda t: t1 < t.point < t2)
    constr_2 = iris.Constraint(clim_season=season.lower())
    cube_out = cube.extract(constr_1 & constr_2)
    return cube_out

def basic_analysis(cube):
    mn = cube.collapsed('time', iris.analysis.MEAN)
    var = cube.collapsed('time', iris.analysis.VARIANCE)
    std = cube.collapsed('time', iris.analysis.STD_DEV)
    return mn, var, std

def plotter (cube_array, name):
    proj = ccrs.PlateCarree(central_longitude=-180)
    for i, cube in enumerate(cube_array):
        plt.figure(figsize=(8, 6))
        plt.axes(projection=proj)
        contour = iplt.contourf(cube, cmap=get_cmap("jet"))
        plt.gca().coastlines()
        cbar = plt.colorbar(shrink=0.6)
        cbar.set_label(cube.units)
        #plt.savefig(str(i)+name+'plot.jpg')
def annual(cube):
    first_30 = cube.extract(iris.Constraint(year=lambda t: 1870 < t.point < 1901)).collapsed('time', iris.analysis.MEAN)
    year_mean = cube.aggregated_by('year', iris.analysis.MEAN)
    print(year_mean)
    anom_1871_1902 = year_mean - first_30
    anom_trend = anom_1871_1902.collapsed(['longitude', 'latitude'], iris.analysis.MEAN)
    qplt.plot(anom_trend, label='year anomaly')
    plt.title('monthly anomaly with respect to the 1870-1900 period mean')

def main():
    # cube = iris.load('~/code/develop/vipc/datasets/NOAA20CRv2_t2m.mon.mean_2.5x2.5.nc', 'air_temperature')[0]
    cube = iris.load('~/code/develop/vipc/datasets/NCEP-NCAR_t2m.mon.mean_1948-2019.nc')[0]

    iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(cube, 'time', name='year_season')
    iris.coord_categorisation.add_year(cube, 'time', name='year')

    #annual(cube)
    winter_cube = cube_extract(cube, 'djf') # chose from 'djf', 'mam', 'jja', 'som'
    spring_cube = cube_extract(cube, 'mam')

    djf_mn = winter_cube.aggregated_by(['clim_season', 'year_season'], iris.analysis.MEAN)
    mam_mn = spring_cube.aggregated_by(['clim_season', 'year_season'], iris.analysis.MEAN)

    overall_mean = cube.collapsed('time', iris.analysis.MEAN)

    winter_mn, winter_var, winter_std = basic_analysis(djf_mn)
    spring_mn, spring_var, spring_std = basic_analysis(mam_mn)
    anom_winter = djf_mn - winter_mn

    # first_30 = winter_cube.extract(iris.Constraint(year=lambda t: 1872 < t.point < 1902)).collapsed('time', iris.analysis.MEAN)
    # anom_1871_1902 = djf_mn - first_30
    # anom_trend = anom_1871_1902.collapsed(['longitude', 'latitude'], iris.analysis.MEAN)
    # qplt.plot(anom_trend, label='winter anom')
    # plt.grid()
    # plt.legend()

    # last_10 = anom_winter.extract(iris.Constraint(year=lambda t: 1999 < t.point < 2013))
    # last_10_e = last_10.collapsed('time', iris.analysis.MEAN)
    plotter([winter_mn, winter_std], 'anomaly_decade')

    #anom_area_avg = anom_winter.collapsed(['longitude', 'latitude'], iris.analysis.MEAN)


    # https://ess.bsc.es/content/ecmwf-s4-and-era-interim-seasonal-climatology

    # ls = [anom_var,anom_std]
    # plotter(ls, 'djf_anom_var_std_1870')

    #########################•###########################•##############•###••##••#
                                ####FILTERING####
    #########################•###########################•##############•###••##••#
    #########################•###########################•##############•###••##••#
#  https://scitools.org.uk/iris/docs/latest/examples/General/SOI_filtering.html?highlight=time
    #########################•###########################•##############•###••##••#

    #plotter(ls,'djf')
    # dynamic_mn = djf_mn
    # years = 5
    # for pack in range(20):
    #     sum_of_cube = 0
    #     for i in range(years):
    #         sum_of_cube += dynamic_mn[pack+i]
    #         print(pack+i)
    #     dyn_mn = sum_of_cube[:,:]/years
    # print(dyn_mn)
    # contour = qplt.contourf(dyn_mn)
    # plt.gca().coastlines()
    # plt.clabel(contour, inline=False)
    # # S'ha de fer les mitjes dinàmiques (després de treure djf_mn) amb un for com una sarten
    # S'ha de treure
    # ls = [winter_mn,winter_var,winter_std,spring_mn,spring_var,spring_std]

    plt.show()


if __name__ == '__main__':
    main()
