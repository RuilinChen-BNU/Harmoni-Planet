# Harmoni-Planet
Harmoni-Planet: A Holistic Harmonization Method for PlanetScope Constellation Imagery Leveraging a Graph-Based Greedy Optimization Strategy

🌟 Features

	1. End-to-end support for PlanetScope imagery harmonization in a holistic, scalable and self-contained manner.
	2. Option to output only the optimized harmonization coefficients.
	3. Supports storing and importing of graph structures. 
	4. Supports both the strip-based and scene-based implementations. 
 
📋 Prerequisites

	1. Python version: 3.8 or higher
	2. Package manager: pip
	3. Required Python libraries: [os, re, csv, json, math, copy, glob2, numpy, pandas, osgeo, scipy, datetime, sklearn]

📂 Data Preparation

	1. Log in to the Planet data ordering platform (i.e., Planet Explorer) and select the desired imagery by clicking “Order Scenes”.
	2. Under Delivery options, choose “Direct download” and select “GeoTIFF or TIFF + RPC”.
	3. Assign a name to your order.
	4. Under Select assets, choose “Surface reflectance” for either 4-band or 8-band imagery.
	5. In Tools & review, enable the following options:
	  a) Clip items to AOI.
        b) Composite items by strip (if scene-based implementation is needed, skip this).
        c) Do NOT select the Harmonize option.
   
	The downloaded data will be packaged as a compressed file titled [order_name]_psscene_[asset_name](e.g., Beijing_20230705_psscene_analytic_8b_sr_udm2). 
	The package will include multiple strip images (in .tif format) along with their corresponding udm files (in .tif) and metadata files (in .json). Example filenames are:
 	  "2023-07-05_strip_6625124_composite.tif"
        "2023-07-05_strip_6625124_composite_udm2.tif"
        "2023-07-05_strip_6625124_composite_metadata.json"

🔧 Parameter Setting

	Parameters for Graph Construction & Initialization:
	1. "imagefile_folder": Specify the folder containing the images, organized as per the required format (see Data Preparation).
	2. "nodefile_path", "edgefile_path" and "invarpx_path": Export the constructed graph files (i.e., node, edge and invariant pixels) to the designated path. If a graph has already been constructed, provide the paths to the required reconstruction files.
	3. "channels": Specify the selected channels to harmonize.
	4. "max_span_days": Exclude edges where the acquisition dates of two strips differ by more than the "max_span_days".
	5. "R2_threshold": Discard edges with RIE r^2 values below this threshold.
 	6. "clear_threshold": Exclude sample plots with a shared valid pixel proportion between strips below  "clear_threshold" (e.g., 0.5 = 50% shared valid pixels). 
	7. "significance_threshold": Set the significance level for identifying invariant pixels using the Multivariate Alteration Detection (MAD) method.
	8. "sample_size", "sample_interval_y", and "sample_interval_x": Define the size and density of sample plots (measured in degrees) used to identify invariant pixels for regression fitting.
 	9. "maximum_invar_px": Specify the maximum number of invariant pixels per edge to control storage size.
	
	Parameters for Graph Optimization:
 	1. "optfile_folder": Directory to store the generated optimization results.
	2. "harmimg_folder": Directory to store the harmonized images.
	3. "last_n" and "impro_ratio": Halts optimization when the average loss improvement over the "last_n" iterations is below the specified "impro_ratio".
 	4. "Block_size": Enables processing harmonized images in a block-wise manner.
  


⚙️ Main Function

	Functions for Graph Construction & Initialization:
	1. get_files(): Reads image and footprint files.
	2. construct_graph(): Graph construction --- Construct a graph structure based on the extracted footprint objects.
	3. initialize_graph(): Graph initialization --- Identify invariant pixels and calculate attributes for nodes and edges.
	4. export_graph(): Export the constructed graph---Save graph details (nodes, edges, and invariant pixels) to disk for future use (Optional).
	5. import_graph(): Load graph details (nodes, edges, attributes, and harmonization parameters) from previously saved files (Optional).
	
	Functions for Graph Optimization:
 	1. optmize_graph(): Graph optimization --- Optimize the graph to calculate harmonization parameters (Harm_k and Harm_b)
	2. export_harmonization_parameters(): Export harmonization parameters --- Save the harmonization parameters (k and b) for each strip to the output folder.
	3. export_harmonized_image(): Export harmonized images --- Generate and save harmonized images for the given dataset using the optimized parameters.


🖼️ Program Output

	The program produces three key outputs:
	1. Constructed graph structure, saved in the following files: "node.csv", "edge.csv" and "invarpx.csv".
	2. Optimized harmonization parameters for each strip image, saved in: "harmonization_parameter.csv".
	3. Harmonized images for the dataset, delivered as output files.
	For additional insights into the cluster statistics of the constructed graph (e.g., main cluster node IDs, count of main/secondary cluster nodes, isolated nodes), apply the cluster_statistic() method on the clusters returned by optimize_graph().
 

✅ Testing

	Testing data:
	The Test_Data folder contains 37 pre-processed and downscaled PlanetScope strips, their corresponding UDM2 files, and metadata files.
 
	Disclaimer:
	Due to licensing restrictions, the dataset has been downscaled to 90m resolution, and metadata irrelevant to image footprints has been removed.
	To use the testing data: 1. Set "imagefile_folder" to the Test_Data folder.
	                         2. Configure output file paths accordingly.

	Visual evaluation:
	To visually evaluate the algorithm's performance on the test dataset, apply the following PlanetScope strip imagery with identical stretching methods:
  	202306276606884.tif | 202306276606795.tif | 202306276606742.tif | 202306276606997.tif | 202306276606764.tif
	202306296611713.tif | 202306296611316.tif | 202306296611662.tif | 202306296611282.tif | 202306296611929.tif
	These images were acquired over a two-day period and exhibit significant radiometric inconsistencies between strips, making them suitable for visualizing the algorithm's performance.


📜 License

	This project is released under the MIT license. See the LICENSE file for details.
 
🙌 Acknowledgements

	Created by Ruilin Chen. Looking forward to your feedback and suggestions! If you have any questions or encounter issues, please feel free to submit an Issue.
