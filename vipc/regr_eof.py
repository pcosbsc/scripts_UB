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
import iris
# from iris.cube import Cube
# from iris.coords import DimCoord
import iris.coord_categorisation
import iris.plot as iplt
# import iris.quickplot as qplt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
import numpy as np
from scipy import stats
import dask.array as da
import scipy.signal

from eofs.iris import Eof

import mask_sealand as msk
# from scipy import eye, asarray, dot, sum, svd
import scipy


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


def plotter(lons, lats, eof, pc, var, num, pc0):
    # Plot the leading EOF expressed as correlation in the Pacific domain.
    plt.figure()
    # clevs = np.linspace(-160, 160, 22)
    # clevs = np.linspace(-0.7, 0.7, 22)
    projection = ccrs.PlateCarree(central_longitude=190)
    ax = plt.axes(projection=projection)
    fill = ax.contourf(lons, lats, eof.squeeze(),
                       transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r)
    ax.coastlines('50m', linewidth=0.8)
    cb = plt.colorbar(fill, orientation='horizontal', format='%.2f')
    cb.set_label('Regr. [Pa]', fontsize=12)
    variance = str("%.2f" % var)
    ax.set_title('EOF'+str(num)+' ('+variance+'%) detr. As regression (1950-2010)', fontsize=16)
    plt.savefig(EXP+'_EOF'+str(num)+'.eps')

    # Plot the leading PC time series.
    plt.figure()
    plt.plot(pc0, pc, color='b', linewidth=2)
    ax = plt.gca()
    ax.axhline(0, color='k')
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized Units')
    ax.set_title('PC'+str(num)+' Time Series', fontsize=16)
    plt.savefig(EXP+'_PC'+str(num)+'.eps')


def plotter2(eof, pc, var, num, pc0):
    # Plot the leading EOF expressed as correlation in the Pacific domain.
    plt.figure()
    # clevs = np.linspace(-160, 160, 22)
    # clevs = np.linspace(-0.7, 0.7, 22)
    projection = ccrs.PlateCarree(central_longitude=190)
    ax = plt.axes(projection=projection)
    fill = iplt.contourf(eof, cmap=plt.cm.RdBu_r)
    ax.coastlines('50m', linewidth=0.8)
    cb = plt.colorbar(fill, orientation='horizontal', format='%.2f')
    cb.set_label('Regr. [Pa]', fontsize=12)
    variance = str("%.2f" % var)
    ax.set_title('EOF'+str(num)+' ('+variance+'%) detr. As regression (1950-2010)', fontsize=16)
    plt.savefig('newtry_'+EXP+'_EOF'+str(num)+'.eps')

    # Plot the leading PC time series.
    plt.figure()
    plt.plot(pc0, pc, color='b', linewidth=2)
    ax = plt.gca()
    ax.axhline(0, color='k')
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Year')
    ax.set_ylabel('Normalized Units')
    ax.set_title('PC'+str(num)+' Time Series', fontsize=16)
    plt.savefig(EXP+'_PC'+str(num)+'.eps')


def anomaly(cube, sst=False):
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    yr_sst = cube.aggregated_by('year', iris.analysis.MEAN)
    clim_sst = yr_sst.collapsed('time', iris.analysis.MEAN)
    anom_sst = yr_sst - clim_sst
    anom_sst_detrend = detrend(anom_sst)
    if sst:
        anom_sst_detrend = msk.mask_landsea(anom_sst_detrend, 'land')  # _detrend
    return anom_sst_detrend


def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = eye(k)
    d = 0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u, s, vh = svd(dot(Phi.T, asarray(Lambda)**3 - (gamma/p) *
                           dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d_old != 0 and d/d_old < 1 + tol:
            break
    return dot(Phi, R)


def ortho_rotation(lam, method='varimax', gamma=None,
                   eps=1e-6, itermax=100):
    """
    Return orthogal rotation matrix
    TODO: - other types beyond
    """
    if gamma is None:
        if (method == 'varimax'):
            gamma = 1.0
        if (method == 'quartimax'):
            gamma = 0.0

    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0

    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new

    return R


def _varimax_kernel(eofs, eps=1e-10, itermax=1000, kaisernorm=True):
    """
    NEED AN EXTRA DIMENSION?

    Rotation of EOFs according to the varimax criterion.
    **Arguments:**
    *eofs*
        A 2-dimensional `~numpy.ndarray` with dimensions [neofs, nspace]
        containing no missing values.
    **Optional arguments:**
    *epsilon*
        Tolerance value used to determine convergence of the rotation
        algorithm. Defaults to 1e-10.
    *itermax*
        Maximum number of iterations to allow in the rotation algorithm.
    *kaisernorm*
        If *True* uses Kaiser row normalization. If *False* no
        normalization is used in the kernel. Defaults to *True*.
    """
    try:
        neofs, nspace = eofs.shape
    except ValueError:
        raise ValueError('kernel requires a 2-D input')
    if neofs < 2:
        raise ValueError('at least 2 EOFs are required for rotation')
    if kaisernorm:
        # Apply Kaiser row normalization.
        scale = np.sqrt((eofs ** 2).sum(axis=0))
        eofs /= scale
    # Define the initial values of the rotation matrix and the convergence
    # monitor.
    rotation = np.eye(neofs, dtype=eofs.dtype)
    delta = 0.
    # Iteratively compute the rotation matrix.
    for i in xrange(itermax):
        z = np.dot(eofs.T, rotation)
        b = np.dot(eofs,
                   z ** 3 - np.dot(z, np.diag((z ** 2).sum(axis=0)) / nspace))
        u, s, v = np.linalg.svd(b)
        rotation = np.dot(u, v)
        delta_previous = delta
        delta = s.sum()
        if delta < delta_previous * (1. + eps):
            # Convergence is reached, stop the iteration.
            break
    # Apply the rotation to the input EOFs.
    reofs = np.dot(eofs.T, rotation).T
    if kaisernorm:
        # Remove the normalization.
        reofs *= scale
    return reofs


EXP = 'AO'
# Read SST anomalies using the iris module. The file contains November-March
# averages of SST anomaly in the central and northern Pacific.
filename = 'ERSSTv4_1950-2010.nc'
# filename = 'slp_mon.era20cr_1950-2010.nc'
sst_cube = iris.load('~/code/develop/vipc/datasets/'+filename,
                     'Monthly Means of Sea Surface Temperature')[0]
slp_cube = iris.load('~/code/develop/vipc/datasets/slp_mon.era20cr_1950-2010.nc')[0]
# print(slp_cube)
sst_cube = sst_cube.collapsed('time', iris.analysis.MEAN)

# print(dif.data)
# cube = slp_cube
# NP
# lat_bds = [20, 70]
# lon_bds = [120, 250]
# SP
# lat_bds = [-70, -20]
# lon_bds = [135, 300]
# TP
# lat_bds = [0, 90]
# lon_bds = [0, 360]
# sst = extract_region(cube, lat_bds, lon_bds)
if 1 == 1:
    # anom_sst_detrend = anomaly(sst_cube, sst=True)
    anom_slp_detrend = anomaly(slp_cube)

    # # Create an EOF solver to do the EOF analysis. Square-root of cosine of
    # # latitude weights are applied before the computation of EOFs.
    solver = Eof(anom_slp_detrend, weights='coslat')
    # Retrieve the leading EOF, expressed as the correlation between the leading
    # PC time series and the input SST anomalies at each grid point, and the
    # leading PC time series itself.
    n = 3
    # eofs = solver.eofsAsCorrelation(neofs=n)
    eofs = solver.eofs(neofs=n)
    print(eofs)
    pcs = solver.pcs(npcs=n)
    pc0 = pcs.coord('year').points
    pc1 = pcs[:, 0].data/np.std(pcs[:, 0].data)
    pc2 = (pcs[:, 1].data*-1)/np.std(pcs[:, 1].data)
    pc3 = pcs[:, 2].data/np.std(pcs[:, 2].data)
    pcs_np = np.array([pc1, pc2, pc3])

    variance_fractions = solver.varianceFraction(neigs=n)
    # pseudo_pcs = solver.projectField(anom_slp_detrend)
    # print(pseudo_pcs)

    # print(eofs[1])
    # plotter2(eofs[0], pc1, variance_fractions[0].data*100, 1, pc0)
    # rotatedPCA=varimax(eofs)
    # print('rota')
    # print(rotatedPCA)
    # anom_slp_detr_gl = anomaly(slp_cube)
    # #
    # lat = anom_slp_detr_gl.coord('latitude').points
    # lon = anom_slp_detr_gl.coord('longitude').points
    # nlat = anom_slp_detr_gl.coord('latitude').shape[0]
    # nlon = anom_slp_detr_gl.coord('longitude').shape[0]
    #
    # regr = np.zeros([nlat, nlon])
    # for i in range(n):
    #     for j in range(nlat):
    #         for k in range(nlon):
    #             slope, intercept, r_value, p_value, std_err = stats.linregress(
    #                 pcs_np[i, :].data, anom_slp_detr_gl[:, j, k].data)
    #             regr[j, k] = slope
    #     print(regr.shape)
    #     plotter(lon, lat, regr, pcs_np[i, :], variance_fractions.data[i]*100, i+1, pc0)
'''
    anom_detr_gl = anomaly(cube)

    lat = anom_detr_gl.coord('latitude').points
    lon = anom_detr_gl.coord('longitude').points
    nlat = anom_detr_gl.coord('latitude').shape[0]
    nlon = anom_detr_gl.coord('longitude').shape[0]

    regr = np.zeros([nlat, nlon])
    for i in range(n):
        for j in range(nlat):
            for k in range(nlon):
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    pcs_np[i, :].data, anom_detr_gl[:, j, k].data)
                regr[j, k] = slope
        print(regr.shape)
        plotter(lon, lat, regr, pcs_np[i, :], variance_fractions.data[i]*100, i+1, pc0)
'''
# plt.show()
