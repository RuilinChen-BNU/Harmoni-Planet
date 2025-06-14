import os
import glob2
import numpy as np
from osgeo import ogr, osr, gdal
gtiff_driver = gdal.GetDriverByName("GTiff")

'''
This script is used to eliminate sub-pixel level registration errors between PS30 and L8 imagery.
'''

# Registration-related parameters
percentile = 75  # Take the top quartile of gradient image features as the registration target
x_shift_min = -1.1
x_shift_max = 1.1
x_shift_interval = 0.1
y_shift_min = -1.1
y_shift_max = 1.1
y_shift_interval = 0.1

# L8 Frame (Reference imagery for registration, chosen based on high percentage of valid observations)
Area = "Nile"
if(Area == "Nile"):
    L8_path = r"H:\Planet_Data\Nile\CESTEM Data\L8\LC08_175042_20230925.tif"
elif(Area == "Beijing"):
    L8_path = r"H:\Planet_Data\Beijing\CESTEM Data\L8\LC08_123032_20230508.tif"
elif(Area == "Indonesia"):
    L8_path = r"H:\Planet_Data\Indonesia\CESTEM Data\L8\LC08_114066_20240815.tif"
else:
    L8_path = r"H:\Planet_Data\GreenLand\CESTEM Data\L8\LC08_231008_20230917.tif"
L8_ds = gdal.Open(L8_path)
L8_proj = L8_ds.GetProjection()
L8_trans = L8_ds.GetGeoTransform()
origin_x, pixel_width, x_rotation, origin_y, y_rotation, pixel_height = L8_trans
L8_xsize = L8_ds.GetRasterBand(1).XSize
L8_ysize = L8_ds.GetRasterBand(1).YSize
L8_img = L8_ds.ReadAsArray()[[3,2,1,4],:,:] # Similarly, read in the bands in the order Red, Green, Blue and NIR
L8_img = np.nan_to_num(L8_img) # Replace NaN values with 0

# PS30 images (PS imagery resampled to 30m resolution, but not yet registered to L8 reference imagery)
PS30_imagefile_folder = r"H:\Planet_Data" + "\\" + Area + "\\" + "CESTEM Data\PS30"
PS30_image_files = glob2.glob(os.path.join(PS30_imagefile_folder, "**", "*.tif"))

# PS30 images after registration
PS30_Registration_imagefile_output_folder = r"H:\Planet_Data" + "\\" + Area + "\\" + "CESTEM Data\PS30 Registration"
PS30_Registration_image_output_files = glob2.glob(os.path.join(PS30_Registration_imagefile_output_folder, "**", "*.tif"))
PS30_Registration_image_output_files = [os.path.basename(i) for i in PS30_Registration_image_output_files]

# Perform registration for each PS30 image with the L8 reference image
for PS30_path in PS30_image_files:
    # Avoid redundant processing
    if(os.path.basename(PS30_path) in PS30_Registration_image_output_files):
        print("pass ",os.path.basename(PS30_path))
        continue

    print("Processing", os.path.basename(PS30_path))
    PS30_ds = gdal.Open(PS30_path)
    
    # Attempt different offset coordinates
    record = [] #To store x_shift, y_shift, correlation
    for x_shift in np.arange(x_shift_min, x_shift_max, x_shift_interval):
        for y_shift in np.arange(y_shift_min, y_shift_max, y_shift_interval):
            
            # Calculate new coordinates after shifting by several pixels
            new_origin_x = origin_x + (pixel_width * x_shift)  # Adjust the upper-left corner x coordinate
            new_origin_y = origin_y + (pixel_height * y_shift)  # Adjust the upper-left corner y coordinate

            # Resample: Use gdal.Warp to realign the data
            warp_options = gdal.WarpOptions(
                outputBounds=(new_origin_x, new_origin_y + L8_ysize * pixel_height,  # Upper-left (x, y)
                              new_origin_x + L8_xsize * pixel_width, new_origin_y),   # Lower-right (x, y)
                width=L8_xsize,  # Maintain original raster width
                height=L8_ysize,  # Maintain original raster height
                resampleAlg="bilinear"  # Resampling algorithm, e.g., bilinear interpolation
            )
            PS30_shifted_ds = gdal.Warp(' ', PS30_ds, options=warp_options)
            PS30_shifted_img = PS30_shifted_ds.ReadAsArray()
        
            # Calculate gradient magnitude image 
            gradient_y, gradient_x = np.gradient(np.mean(PS30_shifted_img, axis = 0))
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

            # Select specific sharp pixels for correlation calculation (valid high-gradient pixels)
            cor_per_band = []
            for b in range(4):
                valid_id = np.where((PS30_shifted_img[b,:,:] != 0)&(L8_img[b,:,:] != 0))
                if(len(valid_id[0]) == 0):
                    continue
                grad_threshold = np.percentile(gradient_magnitude[valid_id], percentile)
                valid_id = np.where((PS30_shifted_img[b,:,:] != 0)&(L8_img[b,:,:] != 0)&(gradient_magnitude>grad_threshold))
                cor_per_band.append(np.corrcoef(PS30_shifted_img[b,:,:][valid_id], L8_img[b,:,:][valid_id])[0, 1])

            if(len(cor_per_band) != 0):
                record.append([x_shift, y_shift, np.mean(cor_per_band)])

    # Calculate the optimal offset (the one that maximizes the correlation in edge pixels)
    if(len(record) != 0):
        record = np.array(record)
        sorted_indices = np.argsort(record[:, 2])[::-1]
        sorted_record = record[sorted_indices]
        x_shift = sorted_record[0][0]
        y_shift = sorted_record[0][1]
    else:
        x_shift = 0
        y_shift = 0
    print(x_shift, y_shift)

    # Apply the optimal offset to the original PS30 image for registration
    new_origin_x = origin_x + (pixel_width * x_shift)
    new_origin_y = origin_y + (pixel_height * y_shift)
    warp_options = gdal.WarpOptions(
        outputBounds=(new_origin_x, new_origin_y + L8_ysize * pixel_height,
                      new_origin_x + L8_xsize * pixel_width, new_origin_y),
        width=L8_xsize,
        height=L8_ysize,
        resampleAlg="bilinear"
    )

    # Resample and output the registered image
    PS30_shifted_ds = gdal.Warp(' ', PS30_ds, options=warp_options)
    img_out_path =  os.path.join(PS30_Registration_imagefile_output_folder,os.path.basename(PS30_path))
    img_out_ds = gtiff_driver.Create(img_out_path,L8_xsize,L8_ysize,4,gdal.GDT_UInt16)
    img_out_ds.SetProjection(L8_proj)
    img_out_ds.SetGeoTransform(L8_trans)
    for b in range(4):
        img_out_band = img_out_ds.GetRasterBand(b+1)
        img_out_band.WriteArray(PS30_shifted_ds.GetRasterBand(b+1).ReadAsArray())
    del img_out_ds

