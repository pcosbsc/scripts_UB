import os

import iris
import iris.analysis
import iris.plot as iplt
import matplotlib.pyplot as plt
import iris.quickplot as qplt
from iris.analysis import Aggregator
from iris.util import rolling_window
import numpy as np
import shapely.vectorized as shp_vect
import cartopy.io.shapereader as shpreader


def _get_geometries_from_shp(shapefilename):
    """Get the mask geometries out from a shapefile."""
    reader = shpreader.Reader(shapefilename)
    # Index 0 grabs the lowest resolution mask (no zoom)
    geometries = [contour for contour in reader.geometries()]
    if not geometries:
        msg = "Could not find any geometry in {}".format(shapefilename)
        raise ValueError(msg)

    # TODO might need this for a later, more enhanced, version
    # geometries = sorted(geometries, key=lambda x: x.area, reverse=True)

    return geometries

def _mask_with_shp(cube, shapefilename, region_indices=None):
    """
    Apply a Natural Earth land/sea mask.
    Apply a pre-made land or sea mask that is extracted form a
    Natural Earth shapefile (proprietary file format). The masking
    process is performed by checking if any given (x, y) point from
    the data cube lies within the desired geometries (eg land, sea) stored
    in the shapefile (this is done via shapefle vectorization and is fast).
    region_indices is a list of indices that the user will want to index
    the regions on (select a region by its index as it is listed in
    the shapefile).
    """
    # Create the region
    regions = _get_geometries_from_shp(shapefilename)
    if region_indices:
        regions = [regions[idx] for idx in region_indices]

    # Create a mask for the data
    mask = np.zeros(cube.shape, dtype=bool)

    # Create a set of x,y points from the cube
    # 1D regular grids
    if cube.coord('longitude').points.ndim < 2:
        x_p, y_p = np.meshgrid(
            cube.coord(axis='X').points,
            cube.coord(axis='Y').points)
    # 2D irregular grids; spit an error for now
    else:
        msg = (f"No fx-files found (sftlf or sftof)!"
               f"2D grids are suboptimally masked with "
               f"Natural Earth masks. Exiting.")
        raise ValueError(msg)

    # Wrap around longitude coordinate to match data
    x_p_180 = np.where(x_p >= 180., x_p - 360., x_p)
    # the NE mask has no points at x = -180 and y = +/-90
    # so we will fool it and apply the mask at (-179, -89, 89) instead
    x_p_180 = np.where(x_p_180 == -180., x_p_180 + 1., x_p_180)
    y_p_0 = np.where(y_p == -90., y_p + 1., y_p)
    y_p_90 = np.where(y_p_0 == 90., y_p_0 - 1., y_p_0)

    for region in regions:
        # Build mask with vectorization
        if cube.ndim == 2:
            mask = shp_vect.contains(region, x_p_180, y_p_90)
        elif cube.ndim == 3:
            mask[:] = shp_vect.contains(region, x_p_180, y_p_90)
        elif cube.ndim == 4:
            mask[:, :] = shp_vect.contains(region, x_p_180, y_p_90)

        # Then apply the mask
        if isinstance(cube.data, np.ma.MaskedArray):
            cube.data.mask |= mask
        else:
            cube.data = np.ma.masked_array(cube.data, mask)

    return cube

def mask_landsea(cube, mask_out):
    cwd = os.path.dirname(__file__)
    shapefiles = { 'land' : os.path.join(cwd, 'ne_masks/ne_10m_land.shp'),
                    'ocean' : os.path.join(cwd, 'ne_masks/ne_10m_ocean.shp')}
    if cube.coord('longitude').points.ndim < 2:
            print(shapefiles[mask_out])
            cube = _mask_with_shp(cube, shapefiles[mask_out], [
                0,
            ])

    return cube

if __name__ == '__main__':
    filename = 'ERSSTv4_1950-2010.nc'
    cwd = os.path.dirname(__file__)
    path = os.path.join(cwd, '/datasets', filename)
    cube = iris.load(path)[0]
    output_cube = mask_landsea(cube, 'ocean')
    print(output_cube)
