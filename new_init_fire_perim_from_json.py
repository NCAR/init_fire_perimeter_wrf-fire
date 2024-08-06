# Copyright 2024      Research Applications Laboratory (RAL), 
#                     National Center for Atmospheric Research (NCAR),
#                     University Corporation for Atmospheric Research (UCAR)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#--------------------------------------------------------------------------------
# Author: Maria Frediani
# Originally created on 2019 (COFPS)
# Last updated on April, 2024 (UFS-Fire)  
#--------------------------------------------------------------------------------


#!/python3
import numpy as np
import os
import sys
import shutil
import glob
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy import ndimage
import matplotlib.path as path
import argparse

# --------------------------------------------------------
# --------------------------------------------------------
def ll_to_ij(xlon, xlat, mylon, mylat): 

    #find the closest index to coordinate
    
    ind = np.unravel_index( 
        np.argmin(
            (np.ma.masked_where(np.isnan(xlat) & (xlat == 0), xlat)
             - mylat)**2 + (
             np.ma.masked_where(np.isnan(xlon) & (xlon == 0), xlon)
            - mylon)**2), np.shape(xlon)) 

    jmin = ind[0] 
    imin = ind[1] 
    print(f'll_to_ij: {jmin},{imin} {xlon[jmin,imin]}, {xlat[jmin,imin]}' )
    return [jmin,imin]

# --------------------------------------------------------
# --------------------------------------------------------

def read_geogrid(var, fname, **kwargs):

    # Set attr = True to fetch data dimension attributes (WRF-Fire)
    attr = kwargs.get('attr', False)

    # Open WRF input file and read in variables and global attributes
    with Dataset(fname, 'r+') as fnc:

        if var is not None:
            dat = np.squeeze(fnc.variables[var][:])
            if attr is False:
                return dat

        if attr is True:
            dx = getattr(fnc, 'DX')
            dy = getattr(fnc, 'DY')
            ny = fnc.getncattr('WEST-EAST_GRID_DIMENSION') # (time,bottom_top,south_north,west_east)
            nx = fnc.getncattr('SOUTH-NORTH_GRID_DIMENSION')
            srx = fnc.getncattr('sr_x')  # refinement ratio 
            sry = fnc.getncattr('sr_y')  # refinement ratio 
            # nrxy are calculated in respect to staggered grid
            nry = fnc.dimensions['west_east_subgrid'].size
            nrx = fnc.dimensions['south_north_subgrid'].size   
            c_lats = fnc.getncattr('corner_lats')
            c_lons = fnc.getncattr('corner_lons')
            # https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/wps.html#:~:text=The%20global%20attributes%20corner_lats,table%20and%20figure%20below.
            # Indices for the unstaggered corners:
            # lowleft 13
            # upleft 14
            # upright 15
            # lowright 16
            c_lats = fnc.getncattr('corner_lats')[-4:]
            c_lons = fnc.getncattr('corner_lons')[-4:]
            
            attr = dict(dx=dx, nx=nx, srx=srx, nrx=nrx, c_lats=c_lats,
                        dy=dy, ny=ny, sry=sry, nry=nry, c_lons=c_lons)

            if var is None: return attr 
            if var is not None: return dat, attr



# --------------------------------------------------------
# --------------------------------------------------------

def read_wrffile(var, fname, **kwargs):

    # Set attr = True to fetch data dimension attributes (WRF-Fire)
    attr = kwargs.get('attr', False)

    # Open WRF input file and read in variables and global attributes
    with Dataset(fname, 'r+') as fnc:

        if var is not None:
            dat = np.squeeze(fnc.variables[var][:])
            if attr is False:
                return dat

        if attr is True:
            dx = getattr(fnc, 'DX')
            dy = getattr(fnc, 'DY')
            ny = fnc.getncattr('WEST-EAST_GRID_DIMENSION') # (time,bottom_top,south_north,west_east)
            nx = fnc.getncattr('SOUTH-NORTH_GRID_DIMENSION')
            srx = int(fnc.dimensions['south_north_subgrid'].size / nx)  # refinement ratio 
            sry = int(fnc.dimensions['west_east_subgrid'].size / ny)  # refinement ratio 
            # Points beyond these indices are zero (including fx_lat/fx_lon) 
            nrx = int((nx - 1) * srx) + 1
            nry = int((ny - 1) * sry) + 1
            attr = dict(dx=dx, nx=nx, srx=srx, nrx=nrx, 
                        dy=dy, ny=ny, sry=sry, nry=nry)

            if var is None: return attr 
            if var is not None: return dat, attr


# --------------------------------------------------------
# --------------------------------------------------------


def write_wrffile(var, data, fname, **kwargs):

    nx, ny = data.shape

    # Open WRF input file and write in variables
    with Dataset(fname, 'r+') as fnc:
        # print(fnc.variables[var].shape)
        # print(data.shape)
        if var in fnc.variables.keys():
            fnc.variables[var][:] = 1.0
            fnc.variables[var][0, :nx, :ny] = data

        else:

            print(f"Variable {var} not in {fname}")
    return


# --------------------------------------------------------
# --------------------------------------------------------

def atm2fire(aij, srx):
    # assumes index starting at 1
    return (aij-1)*(srx) + 1


def refine_dom(lon, lat, rr):

    from scipy.interpolate import griddata
    # rr: refinement ratio, 0 < rr < 1 
 
    nx, ny = lon.shape
    # i, j = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1), indexing='ij')

    # Starting in WPS 4.4: 
    # lat/lon are at center points, however, they should be at the low-left corner
    # to adjust for this, we add 0.5 to the mesh, then we adjust it to align with the center of the fire grid point
    half = 0.5  - rr/2. 
    print(f'shifting fire coordinates by {half} atm point')

    # In WPS versions prior to 4.4, there's no adjustment. half = 0
    i, j = np.meshgrid(np.arange(half, nx+half, 1), np.arange(half, ny+half, 1), indexing='ij')

    ii, jj = np.meshgrid(np.arange(0, nx + rr, rr),
                         np.arange(0, ny + rr, rr),
                         indexing='ij')
 
    xxlon = griddata(np.array([i.flatten(), j.flatten()]).T,
                     lon.flatten(), (ii, jj),
                     method='linear')
    xxlat = griddata(np.array([i.flatten(), j.flatten()]).T,
                     lat.flatten(), (ii, jj),
                     method='linear')
 

    return (xxlon, xxlat)


# --------------------------------------------------------
# --------------------------------------------------------


def mask_perim(perim, coords):

    lon, lat = coords

    # if perim dimensions >2, drop 3rd dim (file created from ArcGIS)
    if perim.shape[-1] > 2: 
        perim = perim[:,:2]

    mp = path.Path(perim, closed=True)
    points = np.array((lon.flatten(), lat.flatten())).T
    mask = mp.contains_points(points).reshape(lat.shape)

    return mask


# --------------------------------------------------------
# --------------------------------------------------------


def read_json_perim(filename):

    fjson = filename
    print(fjson)

    import json
    with open(fjson) as f:
        gj = json.load(f)['features']
    perim = [i for i in gj if i['geometry'] is not None]
    return perim


# --------------------------------------------------------------------
# --------------------------------------------------------------------

if __name__ == '__main__':

    options = argparse.ArgumentParser()

    options.add_argument("-perim_file",
                         action='store',
                         type=str,
                         default='',
                         help="Path to (geo)json file with the fire perimeter")

    options.add_argument("-wrfinput_file",
                         action='store',
                         type=str,
                         default=os.getcwd() + '/wrfinput_d02',
                         help="Path to wrfinput_d0? file to create the perimeter")

    args = options.parse_args()

    perimfile = args.perim_file
    wrffile0 = args.wrfinput_file

    print("Using these arguments:")
    print("-perim_file: ", perimfile)
    print("-wrfinput_file: ", wrffile0)

    # run new_init_geog_fire_perim_from_json.py -wrfinput_file='geo_em.d01.nc' -perim_file='Cameron_Peak_Perimeter_2.geojson'

    # --------------------------------------------------------
    # --------------------------------------------------------

    if not os.path.isfile(wrffile0):
        sys.exit("Cannot find WRF input file")
    else:
        wrffile = wrffile0 + '_perim'
        print('Creating a copy for wrfinput: ', wrffile)
        shutil.copy2(wrffile0, wrffile)

    if not os.path.isfile(perimfile):
        sys.exit("Cannot find perimeter file")

    # --------------------------------------------------------
    # Read fire perimeter
    # --------------------------------------------------------

    gj = read_json_perim(perimfile)

    # This is not going to work for all perimeter files.  
    # The file nested layers depend on the app/website that created it and there's always an exception.
    # This has worked most of the time

    if gj[0]['geometry']['type'] == 'MultiPolygon':

        # --------------------------------------------------------
        # Multipolygon - 1 feature with multiple coordinate sets
        # --------------------------------------------------------

        print('Multipolygon with one feature')

        mpoly = gj[0]['geometry']['coordinates']

        perim = []
        for kk, ii in enumerate(mpoly):
            perim.append(np.squeeze(np.array(ii)))
            print(f'polygon {kk}: {perim[kk].shape}')

    else:

        # --------------------------------------------------------
        # Single or Multipolygon (multiple features each with one coordinate set)
        # --------------------------------------------------------

        print('Single polygon or Multiple features')

        perim = []
        for kk, ii in enumerate(gj):
            perim.append(np.squeeze(np.array(ii['geometry']['coordinates'])))
            print(f'polygon {kk}: {perim[kk].shape}')

    # --------------------------------------------------------
    # Read wrf_input
    # --------------------------------------------------------

    attr = read_wrffile(var=None, attr=True, fname=wrffile)
    nrx = attr['nrx'] # south-north
    nry = attr['nry'] # west-east

    nfuel_cat0 = read_wrffile(var='NFUEL_CAT', fname=wrffile) # shape nrx|sn, nry|we
    nfuel_cat = nfuel_cat0[:nrx, :nry]
    lfn_init = np.zeros(nfuel_cat.shape)

    xlat = read_wrffile(var='XLAT', fname=wrffile)
    xlon = read_wrffile(var='XLONG', fname=wrffile)

    fxlon, fxlat = refine_dom(xlon, xlat, 1./attr['srx'])

    #diag = np.sqrt((attr['dx']/attr['srx'])**2 + (attr['dy']/attr['sry'])**2)
    rdx = attr['dx']/attr['srx'] # ref grid resolution

    # Mask perimeter polygons - inside is True
    per2d = [mask_perim(ii, [fxlon, fxlat]) for ii in perim]

    ## assign 1 to pts inside the shape (where mask is True)
    # calculates the distance between each internal gridpont and the perimeter
    lfn0 = [ndimage.distance_transform_edt(np.where(ii, 1, 0)) for ii in per2d]

    # Points inside the perimeter are assigned 0, and
    # calculates the distance between each external gridpont and the perimeter
    lfn1 = [ndimage.distance_transform_edt(np.where(ii, 0, 1)) for ii in per2d]

    # Sets fuel category to zero inside the perimeter
    # (ideally, we should set fueld load to zero, but it is not a variable in the wrfinput)
    nfuel_cat_update = np.squeeze(np.array([np.where(ii, 14, nfuel_cat) for ii in per2d]))

    # Set distances inside the perimeter to a negative sign 
    llin= 0 - np.min(np.array(lfn0), axis=0)

    # In case of multipolygons, set the distances outside the perimeter to the closest polygon
    llout=np.min(np.array(lfn1), axis=0)

    # Combine inside and outside distances
    lfn = np.where(llout > 0, llout, llin)

    # Calculate distance in meters based on ref grid spacing (assumes dx = dy)
    distlfn = rdx * lfn 

    write_wrffile('NFUEL_CAT', nfuel_cat_update, wrffile)
    write_wrffile('LFN_HIST', distlfn, wrffile)

    # --------------------------------------------------------
    # Plot it
    # --------------------------------------------------------

    import matplotlib.pyplot as plt
    plt.ion()


    # --------------------------------------------------------
    # Plot new fuel field

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, fc='lightgray')
    # ax.imshow(nfuel_cat_update,
    #           extent=[-0.5, nrx + 0.5, -0.5, nry + 0.5],
    #           origin='lower')

    # ax.imshow(np.where(np.any(per2d, axis=0), 0, 1) * 255,
    #           extent=[-0.5, nrx + 0.5, -0.5, nry + 0.5],
    #           origin='lower',
    #           cmap='gist_yarg',
    #           alpha=0.25)

    # ax.set_xlim([2700,2800])
    # ax.set_ylim([4920,5020])

    # p0 = ax.imshow(lfn,
    #                extent=[-0.5, fxnx + 0.5, -0.5, fxny + 0.5],
    #                origin='lower',
    #                cmap='gist_earth')

    # --------------------------------------------------------
    # Plot LFN in and LFN out

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, fc='lightgray')

    # ii, jj = np.meshgrid(np.arange(0, nrx, 1), np.arange(0, nry, 1), indexing='ij')

    # from matplotlib.colors import BoundaryNorm
    # lev = np.arange(0,np.ceil(np.max(llout)),10)
    # lev[0] = 1
    # cmap = plt.get_cmap('gist_earth', len(lev))

    # p1 = ax.pcolor(
    #     # fxlon,
    #     # fxlat,
    #     jj[:nrx,:nry], ii[:nrx,:nry],
    #     np.ma.masked_less_equal(llout, 0),
    #     norm=BoundaryNorm(lev, ncolors=cmap.N, clip=True),
    #     cmap=cmap,
    #     shading='nearest',  # color is centered on the grid 
    #     snap=False, 
    #     rasterized=True, 
    #     zorder=1)
    # cbar1 = fig.colorbar(p1, shrink=0.75, orientation='vertical')

    # lev2 = np.arange(np.floor(np.min(llin)),0,1) 
    # cmap2 = plt.get_cmap('hot', len(lev2))

    # p2 = ax.pcolor(
    #     # fxlon,
    #     # fxlat,
    #     jj[:nrx,:nry], ii[:nrx,:nry],
    #     np.ma.masked_equal(llin, 0),
    #     norm=BoundaryNorm(lev2, ncolors=cmap2.N, clip=True),
    #     cmap=cmap2,
    #     shading='nearest',  # color is centered on the grid 
    #     snap=False, 
    #     rasterized=True, 
    #     zorder=2)

    # cbar2 = fig.colorbar(p2, shrink=0.75, orientation='vertical')

    # ax.plot(jj[100,:], ii[100,:], color='k')

    # # --------------------------------------------------------
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, fc='lightgray')

    # ax.plot(lfn[100,:])
    # ax.plot([ii[0,0],ii[-1,0]],[0,0], color='k')
    # ax.set_ylabel('Distance in grid points')
    # ax.set_xlabel('x(y=100)')

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, fc='lightgray')

    # ax.plot(distlfn[100,:])
    # ax.plot([ii[0,0],ii[-1,0]],[0,0], color='k')
    # ax.set_ylabel('Distance in meters')
    # ax.set_xlabel('x(y=100)')
 
    # # --------------------------------------------------------

    # # --------------------------------------------------------
    # # --------------------------------------------------------

    # import matplotlib.path as path
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as mpatches
    # import cartopy.crs as ccrs
    # import matplotlib as mpl
    # from matplotlib.colors import BoundaryNorm

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, fc='lightgray', projection=ccrs.PlateCarree())

    # ij = ll_to_ij(fxlon, fxlat, np.mean(perim[0][:,0]), np.mean(perim[0][:,1]))
    # delta = 30
    # i0 = ij[0] - delta
    # i1 = ij[0] + delta
    # j0 = ij[1] - delta
    # j1 = ij[1] + delta

    # lev = np.arange(nfuel_cat.max()) 
    # #lev = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # cmap = mpl.cm.get_cmap('Spectral_r', len(lev))

    # p0 = ax.pcolor(
    #     fxlon[i0:i1,j0:j1],
    #     fxlat[i0:i1,j0:j1],
    #     nfuel_cat[i0:i1,j0:j1],
    #     norm=BoundaryNorm(lev, ncolors=cmap.N, clip=True),
    #     transform=ccrs.PlateCarree(),
    #     alpha=0.5,
    #     cmap=cmap,
    #     shading='nearest',  # color is centered on the grid 
    #     snap=False,
    #     rasterized=True,
    #     zorder=1)

    # p0 = ax.pcolor(
    #     fxlon[i0:i1,j0:j1],
    #     fxlat[i0:i1,j0:j1],
    #     nfuel_cat_update[i0:i1,j0:j1],
    #     norm=BoundaryNorm(lev, ncolors=cmap.N, clip=True),
    #     transform=ccrs.PlateCarree(),
    #     alpha=0.25,
    #     cmap=cmap,
    #     shading='nearest',  # color is centered on the grid 
    #     snap=False,
    #     rasterized=True,
    #     zorder=2)

    # mp = path.Path(perim[0], closed=True)
    # patch = mpatches.PathPatch(mp, edgecolor='b', facecolor='none', alpha=0.95)
    # ax.add_patch(patch)

    # arr = np.array(perim[0])
    # ax.plot(arr[:,0], arr[:,1], marker='x', color='b')

    # # Check if coordinates map to the same point
    # wrfij = ll_to_ij(xlon, xlat, np.mean(perim[0][:,0]), np.mean(perim[0][:,1]))
    # fireij = ll_to_ij(fxlon, fxlat, xlon[*wrfij], xlat[*wrfij])

    # ax.plot(xlon[*wrfij], xlat[*wrfij], marker='o', color='k' )
    # ax.plot(fxlon[*fireij], fxlat[*fireij], marker='x', color='r' )

    # # --------------------------------------------------------
    # # --------------------------------------------------------
