#!/usr/bin/env python

import sys, traceback
import cv2
import os
import re
import numpy as np
import argparse
import string
import plantcv as pcv


def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=False)
    parser.add_argument("-r","--result", help="result file.", required= False )
    parser.add_argument("-w","--writeimg", help="write out images.", default=False, action="store_true")
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    args = parser.parse_args()
    return args

### Main pipeline
def main():
  # Get options
  args = options()
  
  # Read image
  img, path, filename = pcv.readimage(args.image)
    
  # Pipeline step
  device = 0
  
  # Convert RGB to HSV and extract the Value channel
  device, img_gray_val = pcv.rgb2gray_hsv(img, 'v', device, args.debug)

  # Convert RGB to HSV and extract the Saturation channel
  device, img_gray_sat = pcv.rgb2gray_hsv(img, 's', device, args.debug)
  img_gray_sat = 255-img_gray_sat 
  
  #Corrects image brightness using inverted Saturation image
  #Original histogram
  sat_hist = cv2.calcHist(tuple(img_gray_sat), [0], None, [256], [0, 256])
  
  # Calculates index of maximum of histogram and finds alpha based on the peak
  sat_hmax = np.argmax(sat_hist)
  sat_alpha = 255 / float(sat_hmax)
  
  # Converts image and plots to screen
  sat_img2 = (img_gray_sat)
  sat_img2 = np.asarray(np.where(sat_img2 <= sat_hmax, np.multiply(sat_alpha,sat_img2), 255), np.uint8)
  
  #Reinverts Saturation Image
  sat_img2 = 255-sat_img2
  
  #Corrects image brightness using inverted Value image
  #Original histogram
  val_hist = cv2.calcHist(tuple(img_gray_val), [0], None, [256], [0, 256])

  # Calculates index of maximum of histogram and finds alpha based on the peak
  val_hmax = np.argmax(val_hist)
  val_alpha = 255 / float(val_hmax)

  # Converts image and plots to screen
  val_img2 = (img_gray_val)
  val_img2 = np.asarray(np.where(val_img2 <= val_hmax, np.multiply(val_alpha,val_img2), 255), np.uint8)

  #Reinverts Value Image
  val_img2 = 255-val_img2
  
  # Threshold the Saturation image
  device, sat_img_binary = pcv.binary_threshold(sat_img2, 35, 255, 'light', device, args.debug)
  
  # Threshold the Value image
  device, val_img_binary = pcv.binary_threshold(val_img2, 35, 255, 'light', device, args.debug)
  
  #Combines Saturation and Value Images
  img_binary = np.where(sat_img_binary<255, val_img_binary,sat_img_binary)
  
  # Fills in speckles using fill
  mask = np.copy(img_binary)
  device, fill_image= pcv.fill(img_binary, mask, 200, device, args.debug)

  # Identify objects
  device, id_objects, obj_hierarchy = pcv.find_objects(img, fill_image, device, args.debug)
  
  # Define ROI
  device, roi, roi_hierarchy = pcv.define_roi(img, 'rectangle', device, None, 'default', args.debug, True, 2000, 500, -500, -7500)
  
  # Decide which objects to keep
  device, roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi, roi_hierarchy, id_objects, obj_hierarchy, device,args.debug)

  # Object combine kept objects
  device, obj, mask = pcv.object_composition(img, roi_objects, roi_obj_hierarchy, device, args.debug)
  
  # Masked image
  device, masked = pcv.apply_mask(img, mask, 'white', device, args.debug)
  
  # Identify infected tissue
  device, img_lab = pcv.rgb2gray_lab(masked, 'a', device, args.debug)
  device, img_hsv = pcv.rgb2gray_hsv(masked, 'v', device, args.debug)

  device, disease_a = pcv.binary_threshold(img_lab, 35, 255, 'light', device, args.debug)
  device, disease_b = pcv.binary_threshold(img_hsv, 35, 255, 'light', device, args.debug)
  

   
  # ############## VIS Analysis ################
  
  outfile=False
  if args.writeimg==True:
    outfile=args.outdir+"/"+filename
  
  # Find shape properties, output shape image (optional)
  device, shape_header,shape_data,shape_img = pcv.analyze_object(img, args.image, obj, mask, device,args.debug,outfile)
  
  # Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
  device, color_header,color_data,color_img= pcv.analyze_color(img, args.image, mask, 256, device, args.debug,None,'v','img',300,outfile)
  
  # Determine color properties: Histograms, Color Slices and Pseudocolored Images, output color analyzed images (optional)
  device, marker_header,marker_data,marker_img= pcv.report_size_marker_area(img,'rectangle',device,args.debug,'detect',3800, 8500,-4700,-4500,'black','light','v',150,outfile)
  print(marker_data)
  #Output shape and color data
  
  result=open(args.result,"a")
  result.write('\t'.join(map(str,shape_header)))
  result.write("\n")
  result.write('\t'.join(map(str,shape_data)))
  result.write("\n")
  for row in shape_img:
      result.write('\t'.join(map(str,row)))
      result.write("\n")
  result.write('\t'.join(map(str,color_header)))
  result.write("\n")
  result.write('\t'.join(map(str,color_data)))
  result.write("\n")
  for row in color_img:
    result.write('\t'.join(map(str,row)))
    result.write("\n")
  result.write('\t'.join(map(str,marker_header)))
  result.write("\n")
  result.write('\t'.join(map(str,marker_data)))
  result.write("\n")
  for row in marker_img:
    result.write('\t'.join(map(str,row)))
    result.write("\n")
  result.close()
  
if __name__ == '__main__':
  main()

