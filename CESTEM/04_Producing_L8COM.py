import os
import glob2
import warnings
import numpy as np
from cubist import Cubist
from osgeo import ogr, osr, gdal
from datetime import datetime, timedelta

'''
This script is used to generate composite L8 images for training harmonization models for each PS image.
For SuperDove and DoveR
'''

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gtiff_driver = gdal.GetDriverByName("GTiff")

# Open all images except the original PS images (PS30, PS500, MCD43, L8)
Area = "GreenLand"
window = 32
L8_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\L8"
L8_files = glob2.glob(os.path.join(L8_imagefile_folder, "**", "*.tif"))
MCD43_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\MCD43"
MCD43_files = glob2.glob(os.path.join(MCD43_imagefile_folder, "**", "*.tif"))
PS30_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\PS30 Registration"
PS30_files = glob2.glob(os.path.join(PS30_imagefile_folder, "**", "*.tif"))
PS30_files = sorted(PS30_files, key=lambda x: (
    os.path.basename(x)[0:10].replace("-",""),
    int(x[-21:-14])))
PS500_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\PS500"
PS500_files = glob2.glob(os.path.join(PS500_imagefile_folder, "**", "*.tif"))
PS500_files = sorted(PS500_files, key=lambda x: (
    os.path.basename(x)[0:10].replace("-",""), 
    int(x[-21:-14])))
L8COM_imagefile_folder = r"H:\Planet_Data"+"\\"+Area+"\\"+"CESTEM Data\L8COM"
L8COM_files = glob2.glob(os.path.join(L8COM_imagefile_folder, "**", "*.tif"))
L8COM_files = [os.path.basename(i) for i in L8COM_files]

# Obtain the width, height, affine coefficients, and projection information for PS images at 30m and 500m resolutions respectively
PS30_width = gdal.Open(PS30_files[0]).GetRasterBand(1).XSize
PS30_height = gdal.Open(PS30_files[0]).GetRasterBand(1).YSize
PS30_trans = gdal.Open(PS30_files[0]).GetGeoTransform()
PS30_invtrans = gdal.InvGeoTransform(PS30_trans)
PS30_proj = gdal.Open(PS30_files[0]).GetProjection()
PS500_width = gdal.Open(PS500_files[0]).GetRasterBand(1).XSize
PS500_height = gdal.Open(PS500_files[0]).GetRasterBand(1).YSize
PS500_trans = gdal.Open(PS500_files[0]).GetGeoTransform()

# Get imaging dates for L8, MCD43, and PS images respectively
L8_date = np.array([datetime.strptime(os.path.basename(i)[-12:-4], "%Y%m%d") for i in L8_files])
MCD43_date = np.array([datetime.strptime(os.path.basename(i).replace("_","")[:-4], "%Y%m%d")  for i in MCD43_files])
PS_date = np.array([datetime.strptime(os.path.basename(i)[0:10].replace("-",""), "%Y%m%d")  for i in PS500_files])

#Produce L8COM
for k in range(35, len(PS30_files)):
    
    # Avoid redundant processing
    if(os.path.basename(PS30_files[k]) in L8COM_files):
        print(os.path.basename(PS30_files[k])," has been generated")
        continue
    
    print("processing ", k, " ",os.path.basename(PS30_files[k]))
    
    # Case1: If there is an L8 image with a large overlapping region and a high proportion of clear observations within ±1 day
    # of the PS image acquisition date (Pday), it can be directly used as L8COM
    Pday = datetime.strptime(os.path.basename(PS30_files[k])[0:10].replace("-",""), "%Y%m%d")
    L8_date_delta = L8_date - Pday
    L8_date_delta = np.array([abs(i.days) for i in L8_date_delta])
    if(len(np.where(L8_date_delta <= 1)[0])>0):
         # There may be multiple L8 images that meet the criteria (e.g., the current PS30 image is split into two tiles),
         # so select the one with the most overlap
        valid30 = []
        for i in np.where(L8_date_delta <= 1)[0]:
            valid30.append(len(np.where((np.max(gdal.Open(PS30_files[k]).ReadAsArray(),axis=0) > 0)&\
                             (np.isnan(gdal.Open(L8_files[i]).ReadAsArray()[1,:,:]) == False))[0]))
        valid30 = np.array(valid30)
        if(valid30[np.where(valid30 == np.max(valid30))[0][0]] > 10):
            # Directly use the L8 image as the L8COM output for the current PS image
            L8_ds = gdal.Open(L8_files[np.where(L8_date_delta <= 1)[0][np.where(valid30 == np.max(valid30))][0]])
            L8_img = L8_ds.ReadAsArray()[[3,2,1,4],:,:]
            L8_img[np.isnan(L8_img)] = 0
            img_out_path =  os.path.join(L8COM_imagefile_folder, (os.path.basename(PS30_files[k])))
            img_out_ds = gtiff_driver.Create(img_out_path,PS30_width,PS30_height,4,gdal.GDT_UInt16)
            img_out_ds.SetProjection(PS30_proj)
            img_out_ds.SetGeoTransform(PS30_trans)
            for b in range(4):
                img_out_band = img_out_ds.GetRasterBand(b+1)
                img_out_band.WriteArray(L8_img[b,:,:]*10000)
            del img_out_ds
            continue

    # Case2: If there isn't an overlapping area with the current PS image within ±1 day of Pday that's large enough and
    # has a higher percentage of clear observation for L8, then start synthesizing L8COM.

    #【1】First, check if there is an MCD43 image on Pday. If it exists, check whether the MCD43 image and
    # the current PS500 have enough common valid pixels; if not satisfied, skip the current PS image directly.
    MCD43_pday_id = np.where(MCD43_date == Pday)[0]
    if(len(MCD43_pday_id)>0):
        valid500 = np.where((np.max(gdal.Open(PS500_files[k]).ReadAsArray(),axis=0) > 0)&\
                            (np.max(gdal.Open(MCD43_files[MCD43_pday_id[0]]).ReadAsArray(),axis=0) > 0))
        if(len(valid500[0]) < 10):
            print("Cannot generate L8 COM due to insufficient common valid pixels between MCD43 and PS500 on Pday.")
            continue
    else:
            print("Cannot generate L8 COM because there is no MCD43 image on Pday.")
            continue
   
    #【2】Then, find L8-PS-MCD43 pairs within the ±window period of the current Pday 
    # that spatially overlap with the PS30 con Pday, to determine the dates to synthesize PSMOD.
    # First, filter out all L8, PS, and MCD43 images within the window period.
    start_date = Pday - timedelta(days=window)  # Beginning Date
    end_date = Pday + timedelta(days=window)    # Ending Date
    L8_id = np.where((L8_date >= start_date) & (L8_date <= end_date))[0]
    MCD43_id = np.where((MCD43_date >= start_date) & (MCD43_date <= end_date))[0]
    PS_id = np.where((PS_date >= start_date) & (PS_date <= end_date))[0]

    # Then validate which L8-PS-MCD43 pair were acquired on the same date and have large overlaps with the PS30 on Pday.
    PSMOD_id = []
    PSMOD_date = []
    for i in L8_id:
        for j in MCD43_id:
            for m in PS_id:
                if(L8_date[i] == MCD43_date[j] == PS_date[m]):                 
                    # Ensure that PS30[k], L8[i], and PS30[m] largely overlap.
                    valid30 = np.where((np.max(gdal.Open(PS30_files[k]).ReadAsArray(),axis=0) > 0)&\
                            (np.max(gdal.Open(PS30_files[m]).ReadAsArray(),axis=0) > 0)&\
                             (np.isnan(gdal.Open(L8_files[i]).ReadAsArray()[1,:,:]) == False))                    
                    # Also ensure that PS500[k], MCD43[j] and PS500[m] largely overlap.
                    valid500 = np.where((np.max(gdal.Open(PS500_files[k]).ReadAsArray(),axis=0) > 0)&\
                            (np.max(gdal.Open(PS500_files[m]).ReadAsArray(),axis=0) > 0)&\
                            (np.max(gdal.Open(MCD43_files[j]).ReadAsArray(),axis=0) > 0))
                    # If overlaps are satisfied.
                    if(len(valid30[0]) > 10) and (len(valid500[0]) > 10):
                        PSMOD_id.append([i,j,m])
                        PSMOD_date.append(L8_date[i])

    if(len(PSMOD_id) == 0):
        print("Cannot generate L8COM as there are no PSMODs.")
        continue

    #【3】Generate PSMOD
    PSMOD = np.zeros((len(PSMOD_id)+1, 4, PS30_height, PS30_width))
    for i in range(len(PSMOD_id)+1):
        print(i,"out of ", len(PSMOD_id), " is being processed")
        
        #【A】Open images on Pday and the PSMOD dates
        if(i == len(PSMOD_id)):
            # Open images on Pday
            MCD43_ds = gdal.Open(MCD43_files[MCD43_pday_id[0]])
            PS30_ds = gdal.Open(PS30_files[k])
            PS500_ds = gdal.Open(PS500_files[k])
        else:
            # Open images on other PSMOD date
            MCD43_ds = gdal.Open(MCD43_files[PSMOD_id[i][1]])
            PS30_ds = gdal.Open(PS30_files[PSMOD_id[i][2]])
            PS500_ds = gdal.Open(PS500_files[PSMOD_id[i][2]])
        MCD43_img = np.zeros((4, PS500_height, PS500_width))
        MCD43_img[2,:,:] = MCD43_ds.GetRasterBand(3).ReadAsArray() #Blue
        MCD43_img[3,:,:] = MCD43_ds.GetRasterBand(2).ReadAsArray() #NIR
        MCD43_img[0,:,:] = MCD43_ds.GetRasterBand(1).ReadAsArray() #Red
        MCD43_img[1,:,:] = MCD43_ds.GetRasterBand(4).ReadAsArray() #Green
        PS30_img = PS30_ds.ReadAsArray().astype(np.float32)/10000
        PS500_img = PS500_ds.ReadAsArray().astype(np.float32)/10000

        #【B】Sample every 5 rows and 5 columns
        row_indices = np.arange(0, PS500_height, 5)  # Sample at every 5 row
        col_indices = np.arange(0, PS500_width, 5)  # Sample at every 5 column
        row_grid, col_grid = np.meshgrid(row_indices, col_indices, indexing='ij')
        sampled_indices = np.column_stack((row_grid.ravel(), col_grid.ravel()))

        #【C】Generate PSMOD
        for b in range(4):
            #【Sampling】
            X = PS500_img[b ,sampled_indices[:, 0], sampled_indices[:, 1]]
            y = MCD43_img[b, sampled_indices[:, 0], sampled_indices[:, 1]]
            X[X==0] = np.nan
            valid_id = np.where(np.logical_and(~np.isnan(X), ~np.isnan(y)) == True)
            if(len(valid_id[0]) < 10):               
                print("Insufficient samples after sampling")
                continue
            X = X[valid_id]
            X = X.reshape(-1,1)
            y = y[valid_id]

            #【Training】
            Primary_model = Cubist(n_rules=5)
            Primary_model.fit(X, y)
            Secondary_model = Cubist(n_rules=1)
            Secondary_model.fit(X, y)

            #【Prediction】
            PS30_to_predict = PS30_img[b,:,:]
            PS500_to_predict = PS500_img[b,:,:]
            # Flatten to a one-dimensional array and skip where the value is zero
            PS30_to_predict_flattened = PS30_to_predict.reshape(-1, 1)
            PS30_to_predict_flattened_valid_id = np.where(PS30_to_predict_flattened > 0)[0]
            PS30_to_predict_flattened_valid = PS30_to_predict_flattened[PS30_to_predict_flattened_valid_id]         
            PS500_to_predict_flattened = PS500_to_predict.reshape(-1, 1)
            PS500_to_predict_flattened_valid_id = np.where(PS500_to_predict_flattened > 0)[0]
            PS500_to_predict_flattened_valid = PS500_to_predict_flattened[PS500_to_predict_flattened_valid_id]
            # Two-stage prediction
            PS30_predicted_core = Secondary_model.predict(PS30_to_predict_flattened_valid).reshape(-1,1)
            PS30_to_predict_flattened[PS30_to_predict_flattened_valid_id] = PS30_predicted_core
            PS30_predicted = PS30_to_predict_flattened.reshape(PS30_height, PS30_width)
            PS30_predicted[PS30_predicted == 0] = np.nan
            PS500_predicted_core = Primary_model.predict(PS500_to_predict_flattened_valid).reshape(-1,1)
            PS500_to_predict_flattened[PS500_to_predict_flattened_valid_id] = PS500_predicted_core
            PS500_predicted = PS500_to_predict_flattened.reshape(PS500_height, PS500_width)
            PS500_predicted[PS500_predicted == 0] = np.nan
            
            #【Fusion】
            for x in range(PS500_width):
                if(len(np.where(PS500_predicted[:,x]>0)[0]) == 0):
                    continue
                cur_xmin = PS500_trans[0] + x*PS500_trans[1]
                cur_xmax = PS500_trans[0] + (x+1)*PS500_trans[1]
                for y in range(PS500_height):
                    if(np.isnan(PS500_predicted[y,x]) == True):
                        continue
                    cur_ymax = PS500_trans[3] + y*PS500_trans[5]
                    cur_ymin = PS500_trans[3] + (y+1)*PS500_trans[5]
                    lu_x, lu_y = map(int, gdal.ApplyGeoTransform(PS30_invtrans, cur_xmin, cur_ymax))
                    rd_x, rd_y = map(int, gdal.ApplyGeoTransform(PS30_invtrans, cur_xmax, cur_ymin))

                    if(lu_x > 0) and (rd_x < PS30_width) and (lu_y > 0) and (rd_y < PS30_height):
                        PS500_value = PS500_predicted[y,x]
                        PS30_block = PS30_predicted[lu_y : rd_y+1, lu_x : rd_x + 1]
                        # Only when the current block image is completely valid
                        if(np.isnan(PS500_value) == True) | (np.isnan(np.mean(PS30_block)) == True):
                            continue                          
                        adjust_para = np.mean(PS30_block)/PS500_value
                        PSMOD[i, b, lu_y : rd_y+1, lu_x : rd_x + 1] = PS30_block/adjust_para
                            
    PSMOD[PSMOD == 0] = np.nan

    #【4】Determine invariant pixels
    # Count the number of valid samples of PSMOD on Pday
    valid30_count = len(np.where(np.mean(PSMOD[-1,:,:,:], axis=0) >0)[0])
    if(valid30_count < 10):
        print("Cannot generate L8COM because there are insufficient samples of PSMOD on Pday.")
        continue

    # Compute the reflectance differences between each PSMOD and the adjusted PS30 on Pday
    dSR = np.zeros((len(PSMOD_id), 4, PS30_height, PS30_width))
    combined_dSR = np.zeros((len(PSMOD_id), PS30_height, PS30_width)) # Absolute mean reflectance difference across all bands
    for i in range(len(PSMOD_id)):
        for b in range(4):
            dSR[i,b,:,:] = (PSMOD[-1,b,:,:] - PSMOD[i,b,:,:])/PSMOD[i,b,:,:]
        combined_dSR[i,:,:] = np.mean(np.abs(dSR[i,:,:,:]), axis = 0) 

    # Determine invariant pixels based on reflectance differences
    invariant_px = np.zeros((len(PSMOD_id), PS30_height, PS30_width))
    Cth = 0.15
    for i in range(len(PSMOD_id)):
        invariant_px[i,:,:][np.where(combined_dSR[i,:,:]<Cth)] = 1 # Mark as invariant pixels
    valid_invar_count = len(np.where(np.max(invariant_px, axis = 0) == 1)[0])

    if(valid_invar_count/valid30_count < 0.5):
        print("Cannot to generate L8COM because there are insufficient invariant pixels.")
        continue

    # Use the invariant pixel mask to update dSR
    for i in range(len(PSMOD_id)):
        for b in range(4):
             dSR[i,b,:,:][invariant_px[i,:,:] == 0] = np.nan

    # Determine the adjustment coefficients for invariant pixels
    COM_adjpara = 1 - dSR 

    #【5】Generate L8COM
    # Open L8 images according to PSMOD_id
    L8_img = np.zeros((len(PSMOD_id), 4, PS30_height, PS30_width))
    for i in range(len(PSMOD_id)):
        L8_ds = gdal.Open(L8_files[PSMOD_id[i][0]])
        L8_img[i,:,:,:] = L8_ds.ReadAsArray()[[3,2,1,4],:,:]

    # For each L8 image: a) Locate invariant pixels, b) Adjust compensations, and c) Perform mean composition
    L8COM = np.zeros((4, PS30_height, PS30_width))
    for b in range(4):
        L8COM[b,:,:] = np.nanmean(L8_img[:,b,:,:]*COM_adjpara[:,b,:,:], axis = 0)
        
    #【6】Export L8COM
    img_out_path =  os.path.join(L8COM_imagefile_folder, (os.path.basename(PS30_files[k])))
    img_out_ds = gtiff_driver.Create(img_out_path,PS30_width,PS30_height,4,gdal.GDT_UInt16)
    img_out_ds.SetProjection(PS30_proj)
    img_out_ds.SetGeoTransform(PS30_trans)
    for b in range(4):
        img_out_band = img_out_ds.GetRasterBand(b+1)
        img_out_band.WriteArray(L8COM[b,:,:]*10000)
    del img_out_ds

        

