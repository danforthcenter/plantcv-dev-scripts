#!/usr/bin/python
import sys, traceback
import cv2
import numpy as np
import argparse
import string
import plantcv as pcv


### Parse command-line arguments ###
def options():
	parser = argparse.ArgumentParser(description="Imaging processing with opencv")
	parser.add_argument("-i", "--image", help="Input image file.", required=True)
	parser.add_argument("-m", "--roi", help="Input region of interest file.", required=False)
	parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=True)
	parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", action="store_true")
	args = parser.parse_args()
	return args


### Begin main pipeline here ####
def main():
	args = options()
	
	# Read image 
  	img, path, filename = pcv.readimage(args.image)

 	# Pipeline step
 	device = 0

	# Convert RGB to HSV and extract the saturation channel
	device, s = pcv.rgb2gray_hsv(img, 's', device, args.debug)

	# Find a threshold for the threshold image. 
	thres = (np.amax(s)/2)*0.8

	# Threshold the value image
	device, s_thresh = pcv.binary_threshold(s, thres, 255, 'light', device, args.debug)

	# Apply Mask (for vis images, mask_color=white)
	device, masked = pcv.apply_mask(img, s_thresh, 'white', device, args.debug)

	# Identify objects
	device, id_objects, obj_hierarchy = pcv.find_objects(masked, s_thresh, device, args.debug)

	# Define ROI
	device, roi1, roi_hierarchy= pcv.define_roi(masked,'rectangle', device, None, 'default', args.debug ,adjust=False, x_adj=0, y_adj=0, w_adj=0, h_adj=0)

	# Decide which objects to keep
	device,roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img,'partial',roi1,roi_hierarchy,id_objects,obj_hierarchy,device, args.debug)

	# Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
  	device, color_header,color_data,norm_slice= pcv.analyze_color(img, args.image, kept_mask, 256, device, args.debug,'all','rgb','v','img',300,args.outdir+'/'+filename)
 
  	# Output color data
  	# pcv.print_results(args.image, color_header, color_data)



# When using the script directely as python spirodela_pipeline.py, the code
# will run main but when importing it to another script, main will not be run
# unless it is called upon by the other script
if __name__ == '__main__':
	main()

