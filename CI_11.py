import netCDF4 as nc4
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def write2cvimgs(img_t, filename_grey, filename_color):
    img = np.rot90(img_t, 2)
    max_p = np.nanmax(img)
    min_p = np.nanmin(img)
    n_img = (img - min_p) / (max_p - min_p)
    img_grey = n_img * 255
    img_grey_uint8 = img_grey.astype(np.uint8)
    img_color = cv2.applyColorMap(img_grey_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(filename_grey, img_grey_uint8)
    cv2.imwrite(filename_color, img_color)

def write2cvimgs2(img_t, filename_grey, filename_color, mask_cloud):
    img = np.rot90(img_t, 2)
    max_p = np.nanmax(img)
    min_p = np.nanmin(img)
    n_img = (img - min_p) / (max_p - min_p)
    img_grey = n_img * 255
    img_grey_uint8 = img_grey.astype(np.uint8)
    img_markcloud_g = np.where(mask_cloud, 255, img_grey_uint8)
    img_color = cv2.applyColorMap(img_markcloud_g, cv2.COLORMAP_JET)
    cv2.imwrite(filename_grey, img_markcloud_g)
    cv2.imwrite(filename_color, img_color)

def cal_CI11(Rrc_469, Rrc_555, Rrc_645, c_mask):
    Rrc_555_p = (Rrc_645-Rrc_469) * ((555.0-469.0)/(645.0-469.0)) + Rrc_469
    CI = Rrc_555 - Rrc_555_p

    return CI

def correct_glint(Rrc_469, Rrc_555, Rrc_645, Rrc_859, Rg_859=0.02, alpha=0.73, beta=0.87, gamma=0.93):
    Rg_469 = alpha * (Rrc_859 - Rg_859)
    Rg_555 = beta * (Rrc_859 - Rg_859)
    Rg_645 = gamma * (Rrc_859 - Rg_859)

    Rrc_469_c = Rrc_469 - Rg_469
    Rrc_555_c = Rrc_555 - Rg_555
    Rrc_645_c = Rrc_645 - Rg_645

    return Rrc_469_c, Rrc_555_c, Rrc_645_c

def mask_cloud(Rrc_1240, Rrc_469, Rrc_555):
    con1= Rrc_1240 >= 0.35
    con2 = Rrc_1240 > 0.04
    con3 = Rrc_1240 < 0.35
    S = Rrc_555 - 1.27*Rrc_469
    con4 = S < -0.06
    con5 = np.logical_and(con2, con3)
    con6 = np.logical_and(con5, con4)
    c_mask = np.logical_or(con1, con6)

    return c_mask

if __name__=="__main__":

    filename = './ocdata/input_data/A2010168191000.nc'
    nc_file = nc4.Dataset(filename, 'r')

    dim_dic = nc_file.dimensions

    vars_g = nc_file.groups['geophysical_data'].variables
    vars_n = nc_file.groups['navigation_data'].variables

    lon = np.array(vars_n['longitude'][:])
    lat = np.array(vars_n['latitude'][:])

    nl = dim_dic['number_of_lines'].size
    pl = dim_dic['pixels_per_line'].size
    pcp = dim_dic['pixel_control_points'].size

    rhos_469 = np.array(vars_g['rhos_469'][:])
    rhos_555 = np.array(vars_g['rhos_555'][:])
    rhos_645 = np.array(vars_g['rhos_645'][:])
    rhos_859 = np.array(vars_g['rhos_859'][:])
    rhos_1240 = np.array(vars_g['rhos_1240'][:])
    rhos_469 = np.where(rhos_469 == -32767.0, np.nan, rhos_469)
    rhos_555 = np.where(rhos_555 == -32767.0, np.nan, rhos_555)
    rhos_645 = np.where(rhos_645 == -32767.0, np.nan, rhos_645)
    rhos_859 = np.where(rhos_859 == -32767.0, np.nan, rhos_859)
    rhos_1240 = np.where(rhos_1240 == -32767.0, np.nan, rhos_1240)

    Lt_469 = np.array(vars_g['Lt_469'][:])
    Lt_555 = np.array(vars_g['Lt_555'][:])
    Lt_645 = np.array(vars_g['Lt_645'][:])
    Lt_859 = np.array(vars_g['Lt_859'][:])
    Lt_1240 = np.array(vars_g['Lt_1240'][:])

    Lr_469 = np.array(vars_g['Lr_469'][:])
    Lr_555 = np.array(vars_g['Lr_555'][:])
    Lr_645 = np.array(vars_g['Lr_645'][:])
    Lr_859 = np.array(vars_g['Lr_859'][:])
    Lr_1240 = np.array(vars_g['Lr_1240'][:])

    nc_file.close()

    rhos_469 = np.where(rhos_469 == -32767.0, np.nan, rhos_469)
    rhos_555 = np.where(rhos_555 == -32767.0, np.nan, rhos_555)
    rhos_645 = np.where(rhos_645 == -32767.0, np.nan, rhos_645)
    rhos_859 = np.where(rhos_859 == -32767.0, np.nan, rhos_859)
    rhos_1240 = np.where(rhos_1240 == -32767.0, np.nan, rhos_1240)

    Rrc_469_c, Rrc_555_c, Rrc_645_c = correct_glint(rhos_469, rhos_555, rhos_645, rhos_859)

    c_mask = mask_cloud(rhos_1240, rhos_469, rhos_555)

    CI = cal_CI11(Rrc_469_c, Rrc_555_c, Rrc_645_c, c_mask)
    CI_rem_c = np.where(c_mask, np.nan, CI)

    CI_rem_c_t = np.rot90(CI_rem_c, 2)

    Image.fromarray(CI_rem_c_t).save('./ocdata/output_data/CI.tif')

    nrows, ncols = CI.shape
    f = nc4.Dataset('./ocdata/output_data/CI.nc', 'w', format='NETCDF4')

    f.createDimension('number_of_lines', nl)
    f.createDimension('pixels_per_line', pl)
    f.createDimension('pixel_control_points', pcp)

    tempgrp = f.createGroup('temp_data')

    CI_nc= tempgrp.createVariable('CI', 'f4', ('number_of_lines', 'pixels_per_line'))
    longitude_nc = tempgrp.createVariable('longitude', 'f4', ('number_of_lines', 'pixel_control_points'))
    latitude_nc = tempgrp.createVariable('latitude', 'f4', ('number_of_lines', 'pixel_control_points'))

    lon = np.rot90(lon, 2)
    lat = np.rot90(lat, 2)

    longitude_nc[:, :] = lon[:, :]
    latitude_nc[:, :] = lat[:, :]

    CI_nc[:, :] = CI_rem_c_t[:, :]

    f.close()

    print("done")

