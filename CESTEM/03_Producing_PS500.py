import re
import os
import glob2
import numpy as np
from osgeo import ogr, osr, gdal
gtiff_driver = gdal.GetDriverByName("GTiff")
np.set_printoptions(suppress=True)

'''
The current script is primarily used to upscale each 3.125-meter resolution PlanetScope (PS) image
to a 500-meter resolution (referred to as the PS 500 image), using the geographic frame of the study area's
MCD43A4 (MCD43) imagery as a reference.
'''

# Search for all .tif files in the specified path (excluding files containing "udm2.tif" in their names)
def get_files(path):
    image_files = glob2.glob(os.path.join(path, "**", "*.tif"))
    image_files = np.array([i for i in image_files if "udm2.tif" not in i])
    return image_files

# MCD43 Frame (Reference imagery frame)
Area = "GreenLand"
MCD43_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\MCD43"
MCD43_files = glob2.glob(os.path.join(MCD43_imagefile_folder, "**", "*.tif"))
MCD43_ds = gdal.Open(MCD43_files[0])
MCD43_proj = MCD43_ds.GetProjection()
MCD43_srs = osr.SpatialReference()
MCD43_srs.ImportFromWkt(MCD43_proj)
MCD43_trans = MCD43_ds.GetGeoTransform()
MCD43_xsize = MCD43_ds.GetRasterBand(1).XSize
MCD43_ysize = MCD43_ds.GetRasterBand(1).YSize

# PS images (PS imagery with 3.125m resolution)
PS_imagefile_folder = r"H:\Planet_Data" + "\\" + Area + "\\" +"Unharmonized"
PS_files = get_files(PS_imagefile_folder)
PS_files = sorted(PS_files, key=lambda x: (
    os.path.basename(x)[0:10].replace("-",""), 
    int(x[-21:-14])))

# PS500 images (PS imagery resampled to 500m resolution)
PS500_imagefile_output_folder = r"H:\Planet_Data"+"\\" + Area + "\\"+ "CESTEM Data\PS500"
PS500_image_output_files = glob2.glob(os.path.join(PS500_imagefile_output_folder, "**", "*.tif"))
PS500_image_output_files = [os.path.basename(i) for i in PS500_image_output_files]

#Produce PS500
for PS_path in PS_files:
    # Avoid redundant processing
    if(os.path.basename(PS_path) in PS500_image_output_files):
        print("pass ",os.path.basename(PS_path))
        continue

    print("Processing: ", os.path.basename(PS_path))
    
    # Open the current PS image
    PS_ds = gdal.Open(PS_path)
    PS_udm_ds = gdal.Open(PS_path[:-4]+"_udm2.tif")
    PS_proj = PS_ds.GetProjection()
    PS_srs = osr.SpatialReference()
    PS_srs.ImportFromWkt(PS_proj)

    #Reproject PS image to MCD43 projection (if necessary)
    if(PS_srs.IsSame(MCD43_srs) == 0):
        # Reproject both PS_ds and PS_udm_ds
        warp_options = gdal.WarpOptions(
            srcSRS=PS_srs, 
            dstSRS=MCD43_srs,
            resampleAlg="near"
        )
        #Read in projected image info
        PS_proj_ds = gdal.Warp(' ', PS_ds, options=warp_options)
        PS_proj = PS_proj_ds.GetProjection()
        PS_trans = PS_proj_ds.GetGeoTransform()
        PS_invtrans = gdal.InvGeoTransform(PS_trans)
        PS_xsize = PS_proj_ds.GetRasterBand(1).XSize
        PS_ysize = PS_proj_ds.GetRasterBand(1).YSize
        PS_img_B = PS_proj_ds.GetRasterBand(2).ReadAsArray()
        PS_img_G = PS_proj_ds.GetRasterBand(4).ReadAsArray()
        PS_img_R = PS_proj_ds.GetRasterBand(6).ReadAsArray()
        PS_img_N = PS_proj_ds.GetRasterBand(8).ReadAsArray()
        PS_udm_proj_ds = gdal.Warp(' ', PS_udm_ds, options=warp_options)
        PS_img_udm = PS_udm_proj_ds.GetRasterBand(1).ReadAsArray()
    else:
        #Read in original image info
        PS_trans = PS_ds.GetGeoTransform()
        PS_invtrans = gdal.InvGeoTransform(PS_trans)
        PS_xsize = PS_ds.GetRasterBand(1).XSize
        PS_ysize = PS_ds.GetRasterBand(1).YSize
        PS_img_B = PS_ds.GetRasterBand(2).ReadAsArray()
        PS_img_G = PS_ds.GetRasterBand(4).ReadAsArray()
        PS_img_R = PS_ds.GetRasterBand(6).ReadAsArray()
        PS_img_N = PS_ds.GetRasterBand(8).ReadAsArray()
        PS_img_udm = PS_udm_ds.GetRasterBand(1).ReadAsArray()

    # Resample to 500m × 500m grids using L8 data as a reference
    PS500 = np.zeros((MCD43_ysize,MCD43_xsize,4))
    for x in range(MCD43_xsize):
        cur_xmin = MCD43_trans[0] + x*MCD43_trans[1]
        cur_xmax = MCD43_trans[0] + (x+1)*MCD43_trans[1]
        for y in range(MCD43_ysize):
            cur_ymax = MCD43_trans[3] + y*MCD43_trans[5]
            cur_ymin = MCD43_trans[3] + (y+1)*MCD43_trans[5]
            # Find the corresponding pixels in 3.125m PS imagery for the current 500m × 500m grid
            lu_x, lu_y = map(int, gdal.ApplyGeoTransform(PS_invtrans, cur_xmin, cur_ymax))
            rd_x, rd_y = map(int, gdal.ApplyGeoTransform(PS_invtrans, cur_xmax, cur_ymin))
            if(lu_x > 0) and (rd_x < PS_xsize) and (lu_y > 0) and (rd_y < PS_ysize):
                PS_img_udm_block = PS_img_udm[lu_y : rd_y+1, lu_x : rd_x + 1].flatten()
                if(len(np.where(PS_img_udm_block == 0)[0]) > 0):
                    continue
                # Store in the order of Red-Green-Blue-NIR
                PS500[y, x, 0] = np.mean(PS_img_R[lu_y : rd_y+1, lu_x : rd_x + 1])
                PS500[y, x, 1] = np.mean(PS_img_G[lu_y : rd_y+1, lu_x : rd_x + 1])
                PS500[y, x, 2] = np.mean(PS_img_B[lu_y : rd_y+1, lu_x : rd_x + 1])
                PS500[y, x, 3] = np.mean(PS_img_N[lu_y : rd_y+1, lu_x : rd_x + 1])
    
    # Output the resampled image
    img_out_path =  os.path.join(PS500_imagefile_output_folder,os.path.basename(PS_path))
    img_out_ds = gtiff_driver.Create(img_out_path, MCD43_xsize, MCD43_ysize, 4, gdal.GDT_UInt16)
    img_out_ds.SetProjection(MCD43_proj)
    img_out_ds.SetGeoTransform(MCD43_trans)
    for b in range(4):
        img_out_band = img_out_ds.GetRasterBand(b+1)
        img_out_band.WriteArray(PS500[:,:,b])
    del img_out_ds
