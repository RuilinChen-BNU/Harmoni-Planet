import os
import glob2
import warnings
import numpy as np
from osgeo import ogr, osr, gdal
from cubist import Cubist
import pandas as pd
'''
This script is used to harmonize each PS image based on their corresponding L8COM images.
'''

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gtiff_driver = gdal.GetDriverByName("GTiff")

# Search for all .tif files in the specified path (excluding files containing "udm2.tif" in their names)
def get_files(path):
    image_files = glob2.glob(os.path.join(path, "**", "*.tif"))
    image_files = np.array([i for i in image_files if "udm2.tif" not in i])
    return image_files

# Open original PS images, PS30 images, PS30_registration and L8COM images
block_size = 1000000 #Limited memory requires step-by-step application of the Secondary_model.
Area = "Nile"
PS_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"Unharmonized"
PS_files = get_files(PS_imagefile_folder)
PS_files = sorted(PS_files, key=lambda x: (
    os.path.basename(x)[0:10].replace("-",""),
    int(x[-21:-14])))
PS_files_name = np.array([os.path.basename(i) for i in PS_files])
PS30_Registration_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\PS30 Registration"
PS30_Registration_files = glob2.glob(os.path.join(PS30_Registration_imagefile_folder, "**", "*.tif"))
PS30_Registration_files = sorted(PS30_Registration_files, key=lambda x: (
    os.path.basename(x)[0:10].replace("-",""),
    int(x[-21:-14])))
PS30_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\PS30"
PS30_files = glob2.glob(os.path.join(PS30_imagefile_folder, "**", "*.tif"))
PS30_files = sorted(PS30_files, key=lambda x: (
    os.path.basename(x)[0:10].replace("-",""),
    int(x[-21:-14])))
L8COM_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\L8COM"
L8COM_files = glob2.glob(os.path.join(L8COM_imagefile_folder, "**", "*.tif"))
L8COM_files_name = np.array([os.path.basename(i) for i in L8COM_files])
Harmonized_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\Harmonized"
Harmonized_files = glob2.glob(os.path.join(Harmonized_imagefile_folder, "**", "*.tif"))
Harmonized_files = [os.path.basename(i) for i in Harmonized_files]

# Obtain the width, height, affine coefficients, and projection information for PS images at 30m and 500m resolutions respectively
PS30_width = gdal.Open(PS30_files[0]).GetRasterBand(1).XSize
PS30_height = gdal.Open(PS30_files[0]).GetRasterBand(1).YSize
PS30_trans = gdal.Open(PS30_files[0]).GetGeoTransform()
PS30_proj = gdal.Open(PS30_files[0]).GetProjection()
PS30_srs = osr.SpatialReference()
PS30_srs.ImportFromWkt(PS30_proj)

for k in range(len(L8COM_files)):
    
    # Avoid redundant processing
    if(os.path.basename(L8COM_files[k]) in Harmonized_files):
        print(os.path.basename(L8COM_files[k])," has been harmonized.")
        continue
    print("Pocessing ",os.path.basename(L8COM_files[k]))
    
    #【1】Identify the corresponding PS30 and PS image of current L8COM
    p = np.where(PS_files_name == L8COM_files_name[k])[0][0]
    
    # Reproject PS image to L8 projection (if necessary)
    PS_ds = gdal.Open(PS_files[p])
    PS_proj = PS_ds.GetProjection()
    PS_srs = osr.SpatialReference()
    PS_srs.ImportFromWkt(PS_proj)
    if(PS_srs.IsSame(PS30_srs) == 0):
        # Reproject both PS_ds and PS_udm_ds
        warp_options = gdal.WarpOptions(
            srcSRS=PS_srs, 
            dstSRS=PS30_srs,
            resampleAlg="near"
        )
        # Read in projected image info and data
        PS_proj_ds = gdal.Warp(' ', PS_ds, options=warp_options)
        PS_proj = PS_proj_ds.GetProjection()
        PS_trans = PS_proj_ds.GetGeoTransform()
        PS_invtrans = gdal.InvGeoTransform(PS_trans)
        PS_width = PS_proj_ds.GetRasterBand(1).XSize
        PS_height = PS_proj_ds.GetRasterBand(1).YSize
        PS_img = PS_proj_ds.ReadAsArray()[[5,3,1,7],:,:]/10000 #RGBN
    else:
        # Read in original image info and data
        PS_trans = PS_ds.GetGeoTransform()
        PS_invtrans = gdal.InvGeoTransform(PS_trans)
        PS_width = PS_ds.GetRasterBand(1).XSize
        PS_height = PS_ds.GetRasterBand(1).YSize
        PS_img = PS_ds.ReadAsArray()[[5,3,1,7],:,:]/10000 #RGBN

    #【2】Train the Cubist model using L8COM and PS30_Registration, and apply it to PS30 and PS images.
    # Open PS30, PS30_Registration and L8COM image.
    L8COM_img = gdal.Open(L8COM_files[k]).ReadAsArray()/10000
    PS30_Registration_img = gdal.Open(PS30_Registration_files[p]).ReadAsArray()/10000
    PS30_img = gdal.Open(PS30_files[p]).ReadAsArray()/10000
    
    # Sample every 5 rows and 5 columns
    row_indices = np.arange(0, PS30_height, 5)
    col_indices = np.arange(0, PS30_width, 5)
    row_grid, col_grid = np.meshgrid(row_indices, col_indices, indexing='ij')
    sampled_indices = np.column_stack((row_grid.ravel(), col_grid.ravel()))
    
    # Harmonize PS images
    PS_HARM = np.zeros((4, PS_height, PS_width))
    for b in range(4):
        print(b)
        #【Sampling】
        X = PS30_Registration_img[b ,sampled_indices[:, 0], sampled_indices[:, 1]]
        y = L8COM_img[b, sampled_indices[:, 0], sampled_indices[:, 1]]
        X[X==0] = np.nan
        y[y==0] = np.nan
        valid_id = np.where(np.logical_and(~np.isnan(X), ~np.isnan(y)) == True)
        if(len(valid_id[0]) < 10):
            print("Insufficient samples after sampling")
            continue
        X = X[valid_id]
        X = X.reshape(-1,1)
        y = y[valid_id]

        #【Training】
        Primary_model = Cubist(n_rules=50)
        Primary_model.fit(X, y)
        Secondary_model = Cubist(n_rules=1)
        Secondary_model.fit(X, y)

        #【Prediction】
        PS_to_predict = PS_img[b,:,:]
        PS30_to_predict = PS30_img[b,:,:]
        # Flatten to a one-dimensional array and skip where the value is zero
        PS_to_predict_flattened = PS_to_predict.reshape(-1, 1)
        PS_to_predict_flattened_valid_id = np.where(PS_to_predict_flattened > 0)[0]
        PS_to_predict_flattened_valid = PS_to_predict_flattened[PS_to_predict_flattened_valid_id]
        PS30_to_predict_flattened = PS30_to_predict.reshape(-1, 1)
        PS30_to_predict_flattened_valid_id = np.where(PS30_to_predict_flattened > 0)[0]
        PS30_to_predict_flattened_valid = PS30_to_predict_flattened[PS30_to_predict_flattened_valid_id]            
        # Two-stage prediction (Due to memory limitations, the Secondary_model is applied in a step-by-step manner)
        PS30_predicted_core = Primary_model.predict(PS30_to_predict_flattened_valid).reshape(-1,1)
        PS30_to_predict_flattened[PS30_to_predict_flattened_valid_id] = PS30_predicted_core
        PS30_predicted = PS30_to_predict_flattened.reshape(PS30_height, PS30_width) 
        PS30_predicted[PS30_predicted == 0] = np.nan
        PS_number_to_infer = PS_to_predict_flattened_valid.shape[0]
        PS_predicted_core = np.zeros((PS_number_to_infer))
        for i in np.arange(0,PS_number_to_infer,block_size):
            if(i+block_size >= PS_number_to_infer):
                PS_predicted_core[i:PS_number_to_infer] = Secondary_model.predict(PS_to_predict_flattened_valid[i:PS_number_to_infer])
            else:
                PS_predicted_core[i:i+block_size] = Secondary_model.predict(PS_to_predict_flattened_valid[i:i+block_size])
        PS_predicted_core = PS_predicted_core.reshape(-1,1)            
        PS_to_predict_flattened[PS_to_predict_flattened_valid_id] = PS_predicted_core            
        PS_predicted = PS_to_predict_flattened.reshape(PS_height, PS_width)          
        PS_predicted[PS_predicted == 0] = np.nan
        
         #【Fusion】
        for x in range(PS30_width):
            if(len(np.where(PS30_predicted[:,x]>0)[0]) == 0):
                continue
            cur_xmin = PS30_trans[0] + x*PS30_trans[1]
            cur_xmax = PS30_trans[0] + (x+1)*PS30_trans[1]
            for y in range(PS30_height):
                if(np.isnan(PS30_predicted[y,x]) == True):
                    continue
                cur_ymax = PS30_trans[3] + y*PS30_trans[5]
                cur_ymin = PS30_trans[3] + (y+1)*PS30_trans[5]
                lu_x, lu_y = map(int, gdal.ApplyGeoTransform(PS_invtrans, cur_xmin, cur_ymax))
                rd_x, rd_y = map(int, gdal.ApplyGeoTransform(PS_invtrans, cur_xmax, cur_ymin))

                if(lu_x > 0) and (rd_x < PS_width) and (lu_y > 0) and (rd_y < PS_height):
                    PS30_value = PS30_predicted[y,x]
                    PS_block = PS_predicted[lu_y : rd_y+1, lu_x : rd_x + 1]
                    # Only when the current block image is completely valid
                    if(np.isnan(PS30_value) == True) | (np.isnan(np.mean(PS_block)) == True):
                        continue
                    adjust_para = np.mean(PS_block)/PS30_value
                    PS_HARM[b, lu_y : rd_y+1, lu_x : rd_x + 1] = PS_block/adjust_para
    
    #【3】Export Harmonized PS images
    img_out_path =  os.path.join(Harmonized_imagefile_folder, os.path.basename(L8COM_files[k]))
    img_out_ds = gtiff_driver.Create(img_out_path,PS_width,PS_height,4,gdal.GDT_UInt16)
    img_out_ds.SetProjection(PS_proj)
    img_out_ds.SetGeoTransform(PS_trans)
    for b in range(4):
        img_out_band = img_out_ds.GetRasterBand(b+1)
        img_out_band.WriteArray(PS_HARM[b,:,:]*10000)
    del img_out_ds
