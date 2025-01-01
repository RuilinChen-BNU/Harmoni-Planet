import os
import re
import csv
import json
import math
import copy
import glob2
import numpy as np
import pandas as pd
from osgeo import ogr, osr, gdal
from numpy.linalg import eig, inv
from scipy.stats import chi2,mode
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

wgs84_osr = osr.SpatialReference()
wgs84_osr.ImportFromEPSG(4326)
gtiff_driver = gdal.GetDriverByName("GTiff")
np.set_printoptions(suppress=True)


'''============================================================================================================'''
'''================================================Function Area=================================================='''
'''============================================================================================================'''
'''***********************************************************Get Files***************************************************************'''
#Enumerate the image files along with their corresponding footprint objects.
def get_files(path):
    image_files = glob2.glob(os.path.join(path, "**", "*.tif"))
    image_files = np.array([i for i in image_files if "udm2.tif" not in i])
    metadata_files = np.array([i.replace(".tif","_metadata.json") for i in image_files])
    footprint_names = np.array([''.join(re.findall(r'\d+',  i.split('\\')[-1])) for i in image_files])
    footprint_objects = get_footprint_object(metadata_files)
    return image_files, footprint_names, footprint_objects


#Extract the footprint of each strip image in the form of OGR geometry from its associated metadata.
def get_footprint_object(metadata_files):
    object_list = []
    for md in metadata_files:
        with open(md, 'r', encoding='utf-8') as file:
            metadata = json.load(file)
        geometry_type = metadata["geometry"]["type"]
        coordinates = metadata["geometry"]["coordinates"]
        
        if geometry_type == "Polygon":
            polygon = ogr.Geometry(ogr.wkbPolygon)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for coord in coordinates[0]:
                ring.AddPoint(coord[0], coord[1])
            polygon.AddGeometry(ring)
            object_list.append(polygon)
            
        elif geometry_type == "MultiPolygon":
            multi_polygon = []
            for polygon_coords in coordinates:
                polygon = ogr.Geometry(ogr.wkbPolygon)
                ring = ogr.Geometry(ogr.wkbLinearRing)
                for coord in polygon_coords[0]:
                    ring.AddPoint(coord[0], coord[1])
                polygon.AddGeometry(ring)
                if not multi_polygon:
                    multi_polygon = polygon
                else:
                    multi_polygon = multi_polygon.Union(polygon)
            object_list.append(multi_polygon)                

    return np.array(object_list)



'''******************************************************Graph Construction**********************************************************'''
#Construct a graph based on the topographical relationships between the footprints of each strip.
def construct_graph(footprint_names,footprint_objects):
    node_names, edge_names, edge_objects = [], [], []
    footprint_data = [(obj, name, datetime(int(name[0:4]), int(name[4:6]), int(name[6:8])), obj.Centroid().GetX()) for obj, name in zip(footprint_objects, footprint_names)]
    footprint_data.sort(key=lambda x: (x[2],x[3])) #Sort by dates (key1) and lontitude (key2)
    #Generate nodes and edges
    for i, (obj_node1, name_node1, date_node1, lon_node1) in enumerate(footprint_data):
        node_names.append(name_node1)
        for obj_node2, name_node2, date_node2, lon_node2 in footprint_data[i+1:]:
            date_delta = (date_node2 - date_node1).days
            if date_delta <= max_span_days and obj_node1.Intersects(obj_node2):
                edge_names.append([name_node1,name_node2])
                edge_objects.append(obj_node1.Intersection(obj_node2))
    return np.array(node_names), np.array(edge_names), np.array(edge_objects)



'''******************************************************Graph Initialization**********************************************************'''
#Initialize the graph by applying linear regression modeling to each edge connection.
def initialize_graph(node_names, edge_names, edge_objects, image_files):
    edge_names_refined, edge_attributes, edge_invar_pxs = [], [], []
    node_attributes_k = [[] for i in range(len(node_names))]
    node_attributes_b = [[] for i in range(len(node_names))]
    node_attributes_r2 = [[] for i in range(len(node_names))]
    node_attributes_mae = [[] for i in range(len(node_names))]
    node_attribute_names = [[] for i in range(len(node_names))]
    for i,obj in enumerate(edge_objects):
        print("graph initialization: ",i/len(edge_objects)*100," %")
        #Identify sample plots in which invariant pixels can be ascertained.
        sample_plots = get_sample_plots(obj)
        #If no sample plot is available within this region.
        if(len(sample_plots) == 0):
            continue
        #Collect the invariant pixels from the designated sample plots.
        image_path_node1 = image_files[np.where(footprint_names == edge_names[i][0])][0]
        image_path_node2 = image_files[np.where(footprint_names == edge_names[i][1])][0]
        invariant_pixels_node1, invariant_pixels_node2 = get_invariant_pixels(sample_plots, image_path_node1, image_path_node2)
        
        #If no valid invariant pixels within this region.
        if(len(invariant_pixels_node1) == 0):
            continue
        
        #If more than the desired number of invariant pixels are identified, they are sampled down to a specified number.
        if(len(invariant_pixels_node1[0]) > maximum_invar_px):
            sample_indices = np.random.choice(len(invariant_pixels_node1[0]), maximum_invar_px, replace=False)
            invariant_pixels_node1 = invariant_pixels_node1[:,sample_indices]
            invariant_pixels_node2 = invariant_pixels_node2[:,sample_indices]

        #Discard the edge with RIE r_2 less than certain threshold
        k, b, r2, mae = linear_regression(invariant_pixels_node2, invariant_pixels_node1)#with node1 is the dependent variable
        if(np.min(r2) < R2_threshold):
            continue
        
        #Initialize the edge attributes, ensuring the original invariant pixels are preserved.
        edge_names_refined.append(edge_names[i])
        edge_attributes.append(np.mean(invariant_pixels_node1 - invariant_pixels_node2,axis=1))
        edge_invar_pxs.append([invariant_pixels_node1,invariant_pixels_node2])

        #Initialize the node attributes
        #For Node 1
        node_attribute_names[np.where(node_names == edge_names[i][0])[0][0]].append(edge_names[i][1])
        node_attributes_k[np.where(node_names == edge_names[i][0])[0][0]].append(k)
        node_attributes_b[np.where(node_names == edge_names[i][0])[0][0]].append(b)
        node_attributes_r2[np.where(node_names == edge_names[i][0])[0][0]].append(r2)
        node_attributes_mae[np.where(node_names == edge_names[i][0])[0][0]].append(mae)
        #For Node 2
        k, b, r2, mae = linear_regression(invariant_pixels_node1, invariant_pixels_node2)#with node2 is the dependent variable
        node_attribute_names[np.where(node_names == edge_names[i][1])[0][0]].append(edge_names[i][0])
        node_attributes_k[np.where(node_names == edge_names[i][1])[0][0]].append(k)
        node_attributes_b[np.where(node_names == edge_names[i][1])[0][0]].append(b)
        node_attributes_r2[np.where(node_names == edge_names[i][1])[0][0]].append(r2)
        node_attributes_mae[np.where(node_names == edge_names[i][1])[0][0]].append(mae)

    return np.array(edge_names_refined), np.array(edge_attributes), edge_invar_pxs, [node_attribute_names, node_attributes_k, node_attributes_b, node_attributes_r2, node_attributes_mae]


#Identify sample plots in which invariant pixels can be ascertained.
def get_sample_plots(edge_object):
    #Get the envelope of the edge_object
    min_x, max_x, min_y, max_y = edge_object.GetEnvelope()
    #Create sample plots based on the specified size and density
    sample_plots = []
    for x in np.arange(min_x + sample_interval_x, max_x - sample_interval_x , sample_interval_x):
        for y in np.arange(min_y + sample_interval_y, max_y - sample_interval_y , sample_interval_y):
            sample_plot = ogr.Geometry(ogr.wkbPolygon)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x-sample_size/2, y-sample_size/2)  #Bottom left
            ring.AddPoint(x-sample_size/2, y+sample_size/2)  #Top left
            ring.AddPoint(x+sample_size/2, y+sample_size/2) #Top right
            ring.AddPoint(x+sample_size/2, y-sample_size/2)  #Bottom right
            ring.AddPoint(x-sample_size/2, y-sample_size/2)  #Close the ring
            sample_plot.AddGeometry(ring)
            #Adopt the sample plot only if it is completely contained within the edge_object
            if(edge_object.Contains(sample_plot)):
               sample_plots.append(sample_plot)
    return sample_plots


#Extract invariant pixels from the overlapping region for RIE fitting.
def get_invariant_pixels(sample_plots, image_path_node1, image_path_node2):
    #Link to images
    image_ds_node1 = gdal.Open(image_path_node1)
    udm_ds_node1 = gdal.Open(image_path_node1.replace(".tif","_udm2.tif"))
    coord_trans_node1 = osr.CoordinateTransformation(wgs84_osr,osr.SpatialReference(image_ds_node1.GetProjection()))
    invgeo_trans_node1 = gdal.InvGeoTransform(image_ds_node1.GetGeoTransform())
    image_ds_node2 = gdal.Open(image_path_node2)
    udm_ds_node2 = gdal.Open(image_path_node2.replace(".tif","_udm2.tif"))
    coord_trans_node2 = osr.CoordinateTransformation(wgs84_osr,osr.SpatialReference(image_ds_node2.GetProjection()))
    invgeo_trans_node2 = gdal.InvGeoTransform(image_ds_node2.GetGeoTransform())
    
    #Get invariant pixels
    invariant_pixels_node1, invariant_pixels_node2 = [], []
    for sp in sample_plots:
        #Determine the requisite reading extent for both images in accordance with the size and location of the current sample plot.
        lu_x1, lu_y1, rd_x1, rd_y1 = get_readin_extent(sp, coord_trans_node1, invgeo_trans_node1)
        lu_x2, lu_y2, rd_x2, rd_y2 = get_readin_extent(sp, coord_trans_node2, invgeo_trans_node2)
        #To perevent the array out of boundary
        lu_x1, lu_y1, rd_x1, rd_y1, lu_x2, lu_y2, rd_x2, rd_y2, width, height = bound_check(udm_ds_node1, udm_ds_node2, lu_x1, lu_y1, rd_x1, rd_y1, lu_x2, lu_y2, rd_x2, rd_y2)
        #Import the udm data
        udm_data_node1 = udm_ds_node1.GetRasterBand(1).ReadAsArray(lu_x1, lu_y1, width, height)
        udm_data_node2 = udm_ds_node2.GetRasterBand(1).ReadAsArray(lu_x2, lu_y2, width, height)
        #If the proportion of valid pixels (i.e., those deemed valid in both images) is lower than the designated 'clear_threshold' value
        if len(np.where((udm_data_node1 == 1)&(udm_data_node2 == 1))[0])/(height*width) < clear_threshold:
            continue
        #Import the surface reflectance data
        image_data_node1 = readin_reflectance_data(image_ds_node1, lu_x1, lu_y1, height, width)
        image_data_node2 = readin_reflectance_data(image_ds_node2, lu_x2, lu_y2, height, width)
        #Test if current area is filled out (invalid area)
        X, Y, Z = image_data_node1.shape
        flattened_image_data_node1 = image_data_node1.reshape(X, Y * Z)
        flattened_image_data_node2 = image_data_node2.reshape(X, Y * Z)
        if (np.any(np.all(flattened_image_data_node1 == 1, axis = 1)) == True) or (np.any(np.all(flattened_image_data_node2 == 1, axis = 1))):
            continue
        #Determine the invariant pixels using MAD (Multivariate Alteration Detection)
        chi_square_dis = Multivariate_Alteration_Detection(image_data_node1, image_data_node2)
        t = chi2.ppf(significance_threshold, len(channels))
        valid_pixel_id = np.where((udm_data_node1 == 1) & (udm_data_node2 == 1) & (chi_square_dis < t))
        #If no valid invariant pixels can be ascertained.
        if len(valid_pixel_id[0]) == 0:
            continue
        else:
            if len(invariant_pixels_node1) == 0:
                invariant_pixels_node1 = mask_image(valid_pixel_id, image_data_node1)
                invariant_pixels_node2 = mask_image(valid_pixel_id, image_data_node2)
            else:
                invariant_pixels_node1 = np.concatenate((invariant_pixels_node1, mask_image(valid_pixel_id, image_data_node1)), axis=1)
                invariant_pixels_node2 = np.concatenate((invariant_pixels_node2, mask_image(valid_pixel_id, image_data_node2)), axis=1)
    return invariant_pixels_node1, invariant_pixels_node2


#Get the readin extent of raster images based on overlapping region vectors.
def get_readin_extent(sample_plot, coord_trans, invgeo_trans):
    #Get the envelope of the edge_object
    min_x, max_x, min_y, max_y = sample_plot.GetEnvelope()
    #If the sample plot intersects the ±180° longitudinal meridian
    if is_cross_180(sample_plot):
        #Perform coordinate transformation for the upper-left and lower-right corner points from the WGS1984 to the UTM coordinate system.
        utm_coords = coord_trans.TransformPoints([(max_y, max_x), (min_y, min_x)])#First upper-left then lower-right
    else:
        utm_coords = coord_trans.TransformPoints([(max_y, min_x), (min_y, max_x)])
    #Apply geo-transformations to determine the initial and final row and column indices for the images (image_node1 and image_node2) within the overlapping region.
    lu_x, lu_y = map(int, gdal.ApplyGeoTransform(invgeo_trans, utm_coords[0][0], utm_coords[0][1]))
    rd_x, rd_y = map(int, gdal.ApplyGeoTransform(invgeo_trans, utm_coords[1][0], utm_coords[1][1]))
    return lu_x, lu_y, rd_x, rd_y


#Check whether this sample plot cross the 180-degree meridian
def is_cross_180(sample_plot):
    min_x, max_x, min_y, max_y = sample_plot.GetEnvelope()
    test_point = ogr.Geometry(ogr.wkbPoint)
    test_point.AddPoint(180, (max_y+min_y)/2)
    return sample_plot.Contains(test_point)


#Check the readin boundary of images
def bound_check(udm_ds1, udm_ds2, lu_x1, lu_y1, rd_x1, rd_y1, lu_x2, lu_y2, rd_x2, rd_y2):
        udm_band1 = udm_ds1.GetRasterBand(1)
        udm_band2 = udm_ds2.GetRasterBand(1)
        xsize1 = udm_band1.XSize
        ysize1 = udm_band1.YSize
        xsize2 = udm_band2.XSize
        ysize2 = udm_band2.YSize
        #Adjust X
        if(lu_x1 < 0):
            lu_x2 += abs(lu_x1)
            lu_x1 = 0
        if(lu_x2 < 0):
            lu_x1 += abs(lu_x2)
            lu_x2 = 0
        if(rd_x1 >= xsize1):
            rd_x2 -= rd_x1 - xsize1  + 1
            rd_x1 = xsize1 - 1
        if(rd_x2 >= xsize2):
            rd_x1 -= rd_x2 - xsize2  + 1
            rd_x2 = xsize2 - 1
        #Adjust Y
        if(lu_y1 < 0):
            lu_y2 += abs(lu_y1)
            lu_y1 = 0
        if(lu_y2 < 0):
            lu_y1 += abs(lu_y2)
            lu_y2 = 0
        if(rd_y1 >= ysize1):
            rd_y2 -= rd_y1 - ysize1  + 1
            rd_y1 = ysize1 - 1
        if(rd_y2 >= ysize2):
            rd_y1 -= rd_y2 - ysize2  + 1
            rd_y2 = ysize2 - 1      

        return lu_x1, lu_y1, rd_x1, rd_y1, lu_x2, lu_y2, rd_x2, rd_y2, min((rd_x1-lu_x1+1), (rd_x2-lu_x2+1)), min((rd_y1-lu_y1+1), (rd_y2-lu_y2+1))
    

#Read in reflectance data according to the pre-determined readin extent
def readin_reflectance_data(image_ds, lu_x, lu_y, height, width):
    c_id = 0
    image_data = np.zeros((len(channels), height, width))
    for c in channels:
        if c == "Coastal Blue":
            c = 1
        elif c == "Blue":
            c = 2
        elif c == "Green I":
            c = 3
        elif c == "Green II":
            c = 4
        elif c == "Yellow":
            c = 5
        elif c == "Red":
            c = 6
        elif c == "Red-Edge":
            c = 7
        elif c == "NIR":
            c = 8
        image_data[c_id,:,:] = image_ds.GetRasterBand(c).ReadAsArray(lu_x, lu_y, width, height)
        c_id += 1
    return image_data


#Implement Multivariate Alteration Detection
def Multivariate_Alteration_Detection(image1, image2):
    #Flatten image
    channel, height, width = image1.shape
    image1 = np.reshape(image1, (channel, -1))
    image2 = np.reshape(image2, (channel, -1))
    bands_count_1, num = image1.shape
    weight = np.ones((1, num))
    can_corr = 100 * np.ones((bands_count_1, 1))
    #Centralization for original pixels'
    mean_1 = np.sum(weight * image1, axis=1, keepdims=True) / np.sum(weight)
    mean_2 = np.sum(weight * image2, axis=1, keepdims=True) / np.sum(weight)
    center_1 = image1 - mean_1
    center_2 = image2 - mean_2
    #Basic elements for calculating a and b
    cov_12 = covw(center_1, center_2, weight)
    size = cov_12.shape[0]
    sigma_11 = cov_12[0:bands_count_1, 0:bands_count_1]
    sigma_22 = cov_12[bands_count_1:size, bands_count_1:size]
    sigma_12 = cov_12[0:bands_count_1, bands_count_1:size]
    sigma_21 = sigma_12.T
    #Now we solve Σ11' Σ12 Σ22' Σ21 a = ρ^2 a
    target_mat = np.dot(np.dot(np.dot(inv(sigma_11), sigma_12), inv(sigma_22)), sigma_21)
    eigenvalue, eigenvector_1 = eig(target_mat)
    eigenvalue = np.sqrt(eigenvalue)
    id1 = eigenvalue.argsort()
    eigenvalue = eigenvalue[id1]
    eigenvector_1 = eigenvector_1[:, id1]
    #Now we solve Σ22' Σ21 Σ11' Σ12 b = ρ^2 b, under the constrain that ρ eq ρ, then b = Σ22' Σ21 a 
    eigenvector_2 = np.dot(np.dot(inv(sigma_22), sigma_21), eigenvector_1)
    norm_1 = np.sqrt(1 / np.diag(np.dot(eigenvector_1.T, np.dot(sigma_11, eigenvector_1))))
    norm_2 = np.sqrt(1 / np.diag(np.dot(eigenvector_2.T, np.dot(sigma_22, eigenvector_2))))
    eigenvector_1 = norm_1 * eigenvector_1
    eigenvector_2 = norm_2 * eigenvector_2
    #mad value for each pixel: aTx - bTy
    mad_variates = np.dot(eigenvector_1.T, center_1) - np.dot(eigenvector_2.T, center_2)
    can_corr = eigenvalue
    mad_var = np.reshape(2 * (1 - can_corr), (bands_count_1, 1))
    chi_square_dis = np.sum(mad_variates * mad_variates / mad_var, axis=0, keepdims=True)
    chi_square_dis = chi_square_dis.reshape(height, width)
    return chi_square_dis


#Calculate weighted cov matrix
def covw(center_X, center_Y, w):
    n = w.shape[1]
    sqrt_w = np.sqrt(w)
    sum_w = w.sum()
    V = np.concatenate((center_X, center_Y), axis=0)
    V = sqrt_w * V
    dis = np.dot(V, V.T) / sum_w * (n / (n - 1))
    return dis


#Return the vector of masked pixel 
def mask_image(mask, image):
    masked_pixels = []
    for c_id in range(image.shape[0]):
        masked_pixels.append(image[c_id,:,:][mask])
    return np.array(masked_pixels)


#Derive the linear transformation coefficients between two strips
def linear_regression(arr1, arr2):
    k, b, r2, mae = [], [], [], []
    for c_id in range(arr1.shape[0]):
        model = LinearRegression()
        model.fit(arr1[c_id,:].reshape(-1,1), arr2[c_id,:].reshape(-1,1))
        y_pred = model.predict(arr1[c_id,:].reshape(-1,1))
        k.append(model.coef_[0][0])
        b.append(model.intercept_[0])
        r2.append(model.score(arr1[c_id,:].reshape(-1,1), arr2[c_id,:].reshape(-1,1)))
        mae.append(np.mean(np.abs(arr2[c_id,:].reshape(-1,1) - y_pred)))
    return k, b, r2, mae


'''*****************************************************Graph Export&Import********************************************************'''
#Export graph to files
def export_graph(nodefile_path, edgefile_path, invarpx_path, node_names, node_attributes, edge_names, edge_attributes, edge_invar_pxs, Harm_k = -1, Harm_b = -1):
    if isinstance(Harm_k, int) == True:
        Harm_k =  np.ones((len(node_names), len(channels)))
        Harm_b = np.zeros((len(node_names), len(channels)))
    #Write out node attributes
    with open(nodefile_path,"w") as f:
        for i in range(len(node_names)):
            if(len(node_attributes[0][i]) == 0):
                to_write = []
                for j in range(2):
                    for c in range(len(channels)):
                        if j == 0:
                            to_write.append(Harm_k[i,c])
                        else:
                            to_write.append(Harm_b[i,c])
                to_write.append(node_names[i])
                np.savetxt(f, [to_write], delimiter=',', fmt='%s')
            else:
                row = np.concatenate((np.array(node_attributes[0][i]).reshape(-1,1), np.array(node_attributes[1][i]), np.array(node_attributes[2][i]),\
                                      np.array(node_attributes[3][i]), np.array(node_attributes[4][i])),axis = 1)
                row = row.flatten()
                row = np.insert(row, 0, node_names[i])
                for j in range(2):
                    for c in range(len(channels)):
                        if j == 0:
                            row = np.insert(row, 0, Harm_b[i,c])
                        else:
                            row = np.insert(row, 0, Harm_k[i,c])
                np.savetxt(f, [row], delimiter=',', fmt='%s')
    
    #Write out edge attributes
    edge_data = np.concatenate((edge_names, edge_attributes),axis = 1)
    with open(edgefile_path,"w") as f:
        for row in edge_data:
            np.savetxt(f, [row], delimiter=',', fmt='%s')

    #Write out invariant pixels
    with open(invarpx_path,"w") as f:
        for row in edge_invar_pxs:
            row = np.concatenate((row[0], row[1]),axis = 0)
            row = row.T.flatten()
            np.savetxt(f, [row], delimiter=',', fmt='%s')


#Import graph from files
def import_graph(nodefile_path, edgefile_path, invarpx_path):
    #Read in node attributes
    node_names, harmk, harmb = [], [], []
    with open(nodefile_path, newline='', encoding='utf-8') as f:
        csvreader = csv.reader(f, delimiter=',')
        k, b, r_2, mae, names = [], [], [], [], []
        for row in csvreader:
            harmk.append(row[0:len(channels)])
            harmb.append(row[len(channels):2*len(channels)])
            node_names.append(row[2*len(channels)])
            if(len(row) == 2*len(channels) + 1):
                k.append([])
                b.append([])
                r_2.append([])
                mae.append([])
                names.append([])
            else:
                cur_k, cur_b, cur_r_2, cur_mae, cur_names = [], [], [], [], []
                interval = 4*len(channels) + 1
                headerlen = 2*len(channels) + 1
                for i in range(int((len(row)-headerlen)/interval)):
                    cur_names.append(row[i*interval+(2*len(channels) + 1)])
                    cur_k.append([float(j) for j in row[i*interval+headerlen+1:i*interval+headerlen+len(channels)+1]])
                    cur_b.append([float(j) for j in row[i*interval+headerlen+len(channels)+1:i*interval+headerlen+2*len(channels)+1]])
                    cur_r_2.append([float(j) for j in row[i*interval+headerlen+2*len(channels)+1:i*interval+headerlen+3*len(channels)+1]])
                    cur_mae.append([float(j) for j in row[i*interval+headerlen+3*len(channels)+1:i*interval+headerlen+4*len(channels)+1]])
                k.append(cur_k)
                b.append(cur_b)
                r_2.append(cur_r_2)
                mae.append(cur_mae)
                names.append(cur_names)
        node_attributes = [names,k,b,r_2,mae]

    #Read in edge attributes
    edge_names, edge_attributes = [], []
    with open(edgefile_path, newline='', encoding='utf-8') as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            edge_names.append(np.array([row[0],row[1]]))
            edge_attributes.append(np.array(row[2:]).astype(np.float64))

    #Read in invariant pixels
    edge_invar_pxs = []
    with open(invarpx_path, newline='', encoding='utf-8') as f:
        csvreader = csv.reader(f, delimiter=',')
        for row in csvreader:
            invar_px1, invar_px2 = [], []
            for i in range(int(len(row)/(2*len(channels)))):
                invar_px1.append(row[i*2*len(channels):i*2*len(channels)+len(channels)])
                invar_px2.append(row[i*2*len(channels)+len(channels):i*2*len(channels)+2*len(channels)])
            edge_invar_pxs.append([np.array(invar_px1).astype(np.float64).T,np.array(invar_px2).astype(np.float64).T])
            
    return np.array(node_names), node_attributes, np.array(edge_names), np.array(edge_attributes), edge_invar_pxs, np.array(harmk).astype(np.float64), np.array(harmb).astype(np.float64)



'''******************************************************Graph Optmization**********************************************************'''
#Graph Optmization.
def optmize_graph(node_names, node_attributes, edge_names, edge_attributes, edge_invar_pxs):
    #Classify nodes into: 1) nodes from main cluster. 2) nodes from sub-cluster. 3) isolated nodes.
    clusters = classify_nodes(node_names, edge_names)
    #Cluster statistic
    main_cluster_id, main_cluster_node_count, other_clusters_node_count, isolated_node_count = cluster_statistic(clusters)
    #ONLY optimize the nodes belonging to the main cluster.
    valid_edge_id = np.array([i for i in range(len(edge_names)) if clusters[np.where(node_names == edge_names[i][0])][0] == main_cluster_id])

    #Optmization Initialization
    #Calculate the total error
    Error_t = Error_Total(edge_attributes, valid_edge_id)
    #Calculate the node error (Not all the error can be corrected, calculate the correctable  error for node)
    Error_n_cor = Error_Node_Correctable(node_names, edge_names, edge_attributes, edge_invar_pxs, clusters)
    #Calculate the averaged node attributes
    Ave_k, Ave_b = Average_KB(node_names, node_attributes, clusters)
    #Keep the process of residual optmization in record
    edge_attributes_record = [valid_edge_id]
    
    #Optmization channel by channel
    Harm_k =  np.ones((len(node_names), len(channels)))
    Harm_b = np.zeros((len(node_names), len(channels)))
    for c in range(len(channels)):
        edge_attributes_record_c = []
        Error_record = [Error_t[c]]
        while(len(Error_record) < last_n) or ((Error_record[-last_n] - Error_record[-1]) >= (Error_record[0] - Error_record[-1])*impro_ratio):
            edge_attributes_record_c.append(np.copy(edge_attributes[:,c]))
            #[1] Identify the node with greatest improvable radiometric inconsistency with surroudings.
            invalid_nodes = np.where(clusters != main_cluster_id)[0]
            candidate_nodes = np.argsort(Error_n_cor[:,c])
            node_id = candidate_nodes[~np.isin(candidate_nodes, invalid_nodes)][-1]
            #[2] Calculate the optimal parameter adjustment Δξkn and Δξbn
            delta_Harm_k = 1/Ave_k[node_id,c]
            delta_Harm_b = delta_Harm_k*(-1)*Ave_b[node_id,c]
            #[3] Adjust the harmonization parameters for the identified strip
            Harm_k[node_id,c] = Harm_k[node_id,c]*delta_Harm_k
            Harm_b[node_id,c] = Harm_b[node_id,c]*delta_Harm_k + delta_Harm_b
            #[4] Update node and edge attributes accordingly
            relate_edge_id = np.where(edge_names == node_names[node_id])#Edges related to the current node
            for i in range(len(node_attributes[0][node_id])):
                #[1] First, update node attributes for the identified node. (the second and third demisions of 【node attributes】)
                #k'nm = knm × Δξkn
                node_attributes[1][node_id][i][c] = node_attributes[1][node_id][i][c]*delta_Harm_k
                #b'nm = bnm × Δξkn + Δξbn
                node_attributes[2][node_id][i][c] = node_attributes[2][node_id][i][c]*delta_Harm_k + delta_Harm_b
                #[2] Then, update node attributes for the connected node of the identified node.
                conn_node_id = np.where(node_names == node_attributes[0][node_id][i])[0][0]
                #current node is the jth connected node of its ith connected node
                j = np.where(np.array(node_attributes[0][conn_node_id]) == node_names[node_id])[0][0]
                #k'mn = kmn / Δξkn
                node_attributes[1][conn_node_id][j][c] = node_attributes[1][conn_node_id][j][c]/delta_Harm_k
                #b'mn = bmn - Δξbn×(kmn / Δξkn)
                node_attributes[2][conn_node_id][j][c] = node_attributes[2][conn_node_id][j][c] - delta_Harm_b*node_attributes[1][conn_node_id][j][c]
                #[3] Update edge attributes accordingly(update invariant pxs by applying delta )
                edge_invar_pxs[relate_edge_id[0][i]][relate_edge_id[1][i]][c,:] = edge_invar_pxs[relate_edge_id[0][i]][relate_edge_id[1][i]][c,:]*delta_Harm_k + delta_Harm_b
                edge_attributes[relate_edge_id[0][i]][c] = np.mean(edge_invar_pxs[relate_edge_id[0][i]][0][c,:] - edge_invar_pxs[relate_edge_id[0][i]][1][c,:])
                #[4] Update error for connected node
                Error_n_cor[conn_node_id,c] = Error_Node_Correctable(node_names, edge_names, edge_attributes, edge_invar_pxs, clusters, c, conn_node_id)
                #[5] Update averaged node attributes for connected node
                Ave_k[conn_node_id,c], Ave_b[conn_node_id,c] = Average_KB(node_names, node_attributes, clusters, c, conn_node_id)
            #[6] Update Total error, Node error and  averaged node attributes (based on the updated node and edge attributes)            
            Error_t[c] = Error_Total(edge_attributes, valid_edge_id, c)
            Error_n_cor[node_id,c] = Error_Node_Correctable(node_names, edge_names, edge_attributes, edge_invar_pxs, clusters, c, node_id)
            Ave_k[node_id,c], Ave_b[node_id,c] = Average_KB(node_names, node_attributes, clusters, c, node_id)
            Error_record.append(Error_t[c])
            #Reloop until exiting condition is met.
        edge_attributes_record.append(edge_attributes_record_c)
        print("graph optimization: ", c/len(channels)*100, " %")
    
    return Harm_k, Harm_b, edge_attributes_record, clusters


#Classify nodes into: 1) nodes from main cluster. 2) nodes from sub-cluster. 3) isolated nodes.
def classify_nodes(node_names, edge_names):
    visited_nodes = np.zeros((len(node_names)))
    clusters = np.zeros((len(node_names)))
    cluster_count = 0
    while(len(np.where(visited_nodes == 0)[0]) != 0):
        visited_nodes_pre = np.copy(visited_nodes)
        intinal_node = node_names[np.where(visited_nodes == 0)[0][0]]
        visited_nodes[np.where(node_names == intinal_node)] = 1
        #If the intinal node is an isolated node
        if(len(np.where(edge_names == intinal_node)[0]) == 0):
            clusters[np.where(node_names == intinal_node)] = -1
        else:
            detect_cluster(intinal_node, visited_nodes, node_names, edge_names)
            clusters[np.where((visited_nodes - visited_nodes_pre) == 1)] = cluster_count
            cluster_count += 1
    return clusters


#For a given node, identify the complete cluster of its connected nodes.
def detect_cluster(intinal_node, visited_nodes, node_names, edge_names):
    #Traverse each neighboring node connected to the【intinal_node】.
    flipped_array = ~np.where(edge_names == intinal_node)[1].astype(bool)
    neighbour_nodes = edge_names[(np.where(edge_names == intinal_node)[0],flipped_array.astype(int))]
    for node in neighbour_nodes:
        if(visited_nodes[np.where(node_names == node)] == 0):
            #If any neighboring node remains unvisited, proceed with a more in-depth search.
            visited_nodes[np.where(node_names == node)] = 1
            detect_cluster(node, visited_nodes, node_names, edge_names)


#main_cluster_id, main_cluster_node_count, other_clusters_node_count, isolated_node_count
def cluster_statistic(clusters):
    clusters = clusters[clusters!=-1]
    return mode(clusters)[0],len(clusters[np.where(clusters == mode(clusters)[0])]),\
           len(clusters) - len(clusters[np.where(clusters == mode(clusters)[0])]) - len(np.where(clusters == -1)[0]),\
           len(np.where(clusters == -1)[0])


#Calculate the total loss
def Error_Total(edge_attributes, valid_edge_id, channel=-1):
    if channel == -1:
        return np.mean(np.abs(edge_attributes[valid_edge_id]),axis = 0)
    else:
        return np.mean(np.abs(edge_attributes[valid_edge_id][:,channel]))


#Calculate the correctable loss for each node
def Error_Node_Correctable(node_names, edge_names, edge_attributes, edge_invar_pxs, clusters, channel = -1, n_id = -1):
    #Cluster statistic
    main_cluster_id, main_cluster_node_count, other_clusters_node_count, isolated_node_count = cluster_statistic(clusters)
    if channel == -1 and n_id == -1:
        error_cor_nodes = np.zeros((len(node_names),len(channels)))
        for i in range(len(node_names)):
            if(clusters[i] == main_cluster_id):
                #Statistic though all related edges
                corrected_error = []
                relate_edge_id = np.array(np.where(edge_names == node_names[i])).T
                for edge in relate_edge_id:
                    cur_corrected_error = []
                    cur_edge_attributes = np.copy(edge_attributes[edge[0]])
                    cur_edge_invar_pxs = np.copy(edge_invar_pxs[edge[0]])
                    for c in range(len(channels)):
                        #Calculate the optimal parameter adjustment Δξkn and Δξbn for the current node
                        Ave_k,Ave_b = Average_KB(node_names, node_attributes, clusters, c, i)
                        delta_Harm_k = 1/Ave_k
                        delta_Harm_b = delta_Harm_k*(-1)*Ave_b
                        #Calculate the error improvment by applying harmonization paramenters.
                        cur_edge_invar_pxs[edge[1]][c,:] = cur_edge_invar_pxs[edge[1]][c,:]*delta_Harm_k + delta_Harm_b
                        cur_edge_attributes[c] = np.mean(cur_edge_invar_pxs[0][c,:] - cur_edge_invar_pxs[1][c,:])
                    corrected_error.append(np.abs(np.copy(edge_attributes[edge[0]])) - np.abs(cur_edge_attributes)) #before - after
                error_cor_nodes[i] = np.mean(corrected_error,axis=0)
        return error_cor_nodes
    else:
        #Statistic though all related edges
        corrected_error = []
        relate_edge_id = np.array(np.where(edge_names == node_names[n_id])).T
        for edge in relate_edge_id:
            cur_edge_invar_pxs = np.copy(edge_invar_pxs[edge[0]])[:,channel]
            #Calculate the optimal parameter adjustment Δξkn and Δξbn for the current node
            Ave_k,Ave_b = Average_KB(node_names, node_attributes, clusters, channel, n_id)
            delta_Harm_k = 1/Ave_k
            delta_Harm_b = delta_Harm_k*(-1)*Ave_b
            #Calculate the error improvment by applying harmonization paramenters.
            cur_edge_invar_pxs[edge[1]] = cur_edge_invar_pxs[edge[1]]*delta_Harm_k + delta_Harm_b
            cur_edge_attributes = np.mean(cur_edge_invar_pxs[0]- cur_edge_invar_pxs[1])
            corrected_error.append(abs(edge_attributes[edge[0]][channel]) - abs(cur_edge_attributes))#before - after
        return np.mean(corrected_error)
        

#Calculate the average K and B for each node
def Average_KB(node_names, node_attributes, clusters, channel = -1, n_id = -1):
    #Cluster statistic
    main_cluster_id, main_cluster_node_count, other_clusters_node_count, isolated_node_count = cluster_statistic(clusters)
    if channel == -1 and n_id == -1:
        average_k = np.zeros((len(node_names),len(channels)))
        average_b = np.zeros((len(node_names),len(channels)))
        for i in range(len(node_names)):
            if(clusters[i] == main_cluster_id):
                average_b[i] = np.mean(node_attributes[2][i],axis=0)
                average_k[i] = np.mean(node_attributes[1][i],axis=0)
    else:
        average_b = np.mean(np.array(node_attributes[2][n_id])[:,channel])
        average_k = np.mean(np.array(node_attributes[1][n_id])[:,channel])
    return average_k, average_b


#Export Harmonization Parameters
def export_harmonization_parameters(path, node_names, clusters, Harm_k, Harm_b):
    main_cluster_id, main_cluster_node_count, other_clusters_node_count, isolated_node_count = cluster_statistic(clusters)
    status = np.copy(clusters.astype(str))
    status[np.where(status != str(main_cluster_id))] = "fail"
    status[np.where(status == str(main_cluster_id))] = "success"
    #Export harmonization parameters
    node_names = node_names.astype(str)
    harmonization_parameters = np.concatenate((node_names.reshape(1,-1), status.reshape(1,-1), Harm_k.T, Harm_b.T), axis = 0)
    harmonization_parameters_col = ["node_names", "status"]
    for i in range(2):
        for c in channels:
            if(i == 0):
                harmonization_parameters_col.append(c + "_"  + "b")
            elif(i == 1):
                harmonization_parameters_col.append(c + "_"  + "k")
    pd.DataFrame(harmonization_parameters.T,columns = harmonization_parameters_col).to_csv(path + "\\" + "harmonization_parameters.csv", index=False)
    print("harmonization parameters have been exported")


#Export Harmonized Images
def export_harmonized_image(harmimg_folder, imagefile_folder, node_names, clusters, Harm_k, Harm_b):
    if not os.path.exists(harmimg_folder):
            os.makedirs(harmimg_folder)
    image_files, footprint_names, footprint_objects = get_files(imagefile_folder)
    main_cluster_id, main_cluster_node_count, other_clusters_node_count, isolated_node_count = cluster_statistic(clusters)
    for i in range(len(image_files)):
        #If the node can be harmonized
        node_name = footprint_names[i]
        if(clusters[np.where(node_names == node_name)][0] != main_cluster_id):
            continue
        #Harmonization parameters
        Cur_Harm_k = Harm_k[np.where(node_names == node_name)[0][0]]
        Cur_Harm_b = Harm_b[np.where(node_names == node_name)[0][0]]
        #Link to input image
        img_file = image_files[i]
        img_ds = gdal.Open(img_file)
        Cur_Rows = img_ds.GetRasterBand(1).YSize
        Cur_Columns = img_ds.GetRasterBand(1).XSize
        Cur_GeoTrans = img_ds.GetGeoTransform()
        Cur_Projection = img_ds.GetProjection()
        #Link to output image
        img_out_path = harmimg_folder + "\\" + node_name+"_harmonized.tif"
        img_out_ds = gtiff_driver.Create(img_out_path,Cur_Columns,Cur_Rows,len(channels),gdal.GDT_UInt32)
        img_out_ds.SetProjection(Cur_Projection)
        img_out_ds.SetGeoTransform(Cur_GeoTrans)
        #Processing by block
        for x in range(0, Cur_Columns, Block_size): 
            if (x + Block_size < Cur_Columns): 
                cols = Block_size 
            else: 
                cols = Cur_Columns - x 
            for y in range(0, Cur_Rows, Block_size): 
                if (y + Block_size < Cur_Rows): 
                    rows = Block_size 
                else: 
                    rows = Cur_Rows - y
                #Read in & write out in block
                image_data = np.zeros((rows,cols,len(channels)))
                c_id = 0
                for c in channels:
                    if c == "Coastal Blue":
                        c = 1
                    elif c == "Blue":
                        c = 2
                    elif c == "Green I":
                        c = 3
                    elif c == "Green II":
                        c = 4
                    elif c == "Yellow":
                        c = 5
                    elif c == "Red":
                        c = 6
                    elif c == "Red-Edge":
                        c = 7
                    elif c == "NIR":
                        c = 8
                    image_data[:,:,c_id] = img_ds.GetRasterBand(c).ReadAsArray(x, y, cols, rows)
                    image_data[:,:,c_id][np.where(image_data[:,:,c_id]!=0)] *= Cur_Harm_k[c_id]
                    image_data[:,:,c_id][np.where(image_data[:,:,c_id]!=0)] += Cur_Harm_b[c_id]
                    img_out_band = img_out_ds.GetRasterBand(c_id+1)
                    img_out_band.WriteArray(image_data[:,:,c_id], x, y)
                    c_id += 1
        del img_out_ds
        print(node_name," has been exported")




'''=========================================================================================================='''
'''==============================================Parameter Setting============================================'''
'''=========================================================================================================='''
'''************[1] Graph Construction & Initialization***************'''
# Define the folder containing the images, organized according to the required format
imagefile_folder = r"H:\Planet_Data\Beijing\Test_Data"
# Export the constructed graph files (i.e., node, edge and invariant pixels) to the designated path.
# If a graph has already been constructed, provide the paths to the required reconstruction files.
nodefile_path = r"H:\Planet_Data\Beijing\node_x.csv"        # Path to the node data file
edgefile_path = r"H:\Planet_Data\Beijing\edge_x.csv"        # Path to the edge data file
invarpx_path = r"H:\Planet_Data\Beijing\invar_x.csv"        # Path to the invariant pixel data file
# Specify the selected channels to be harmonized.
channels = ["Coastal Blue", "Blue", "Green I", "Green II", "Yellow", "Red", "Red-Edge", "NIR"]
# Exclude edges where the acquisition dates of the two strips differ by more than "max_span_days".
max_span_days = 30
# Discard edges with RIE r^2 values below this threshold.
R2_threshold = 0.7
# Exclude sample plots where the proportion of shared valid pixels between strips is below "clear_threshold" (e.g., 0.5 = 50% shared valid pixels). 
clear_threshold = 0.5
# Define the significance level for identifying invariant pixels using the Multivariate Alteration Detection (MAD) method.
significance_threshold = 0.001
# Define the size and density of sample plots (measured in degrees) used to identify invariant pixels for regression fitting.
sample_size = 0.1  # Plot size
sample_interval_y = 0.01  # Sampling interval in the Y direction
sample_interval_x = 0.01  # Sampling interval in the X direction
# Maximum number of invariant pixels per edge (limits storage size).
maximum_invar_px = 2000

'''*******************[2] Graph Optimization*********************'''
# Specify the folder to store the generated optimization results.
optfile_folder = r"H:\Planet_Data\Beijing"
# Specify the folder to store the harmonized images.
harmimg_folder = r"H:\Planet_Data\Beijing\Test_corrected"
# Stop the optimization process when the average loss over the last "last_n" iterations improves by less than "impro_ratio".
last_n = 200          # Number of iterations to monitor
impro_ratio = 0.0001  # Minimum improvement ratio
# Generate harmonized images in a block-wise manner.
Block_size = 8000     # Size of each block for processing



'''=========================================================================================================='''
'''===============================================Main Function==============================================='''
'''=========================================================================================================='''
'''************ [1] Graph Construction & Initialization ***************'''
# Step 1: Read in image and footprint files
# - Enumerate image files along with their corresponding footprint objects.
image_files, footprint_names, footprint_objects = get_files(imagefile_folder)

# Step 2: Graph construction
# - Construct a graph structure based on the extracted footprint objects.
node_names, edge_names, edge_objects = construct_graph(footprint_names,footprint_objects)

# Step 3: Graph initialization
# - Identify invariant pixels and calculate attributes for nodes and edges.
edge_names, edge_attributes, edge_invar_pxs, node_attributes = initialize_graph(node_names, edge_names, edge_objects, image_files)

# Step 4: Optional - Export the constructed graph
# - Save graph details (nodes, edges, and invariant pixels) to disk for future use.
export_graph(nodefile_path, edgefile_path, invarpx_path, node_names, node_attributes, edge_names, edge_attributes, edge_invar_pxs)

# Alternative: If a pre-existing graph is available, import it instead.
# - Load graph details (nodes, edges, attributes, and harmonization parameters) from previously saved files.
#node_names, node_attributes, edge_names, edge_attributes, edge_invar_pxs, Harm_k, Harm_b = import_graph(nodefile_path, edgefile_path, invarpx_path)

'''******************* [2] Graph Optimization *********************'''
# Step 1: Perform graph optimization
# - Optimize the graph to calculate harmonization parameters (Harm_k and Harm_b).
# - Track edge attribute updates, node modifications, and cluster grouping.
Harm_k, Harm_b, edge_attributes_record, clusters = optmize_graph(node_names, node_attributes, edge_names, edge_attributes, edge_invar_pxs)

# Step 2: Export harmonization parameters (strip-specific)
# - Save the harmonization parameters (k and b) for each strip to the output folder.
export_harmonization_parameters(optfile_folder, node_names, clusters, Harm_k, Harm_b)

# Step 3: Optional -  Export harmonized images (may be VERY time-consuming)
# - Generate and save harmonized images for the given dataset using the computed parameters.
export_harmonized_image(harmimg_folder, imagefile_folder, node_names, clusters, Harm_k, Harm_b)

# Step 4: Optional -  Cluster statistic
#main_cluster_id, main_cluster_node_count, other_clusters_node_count, isolated_node_count = cluster_statistic(clusters)
