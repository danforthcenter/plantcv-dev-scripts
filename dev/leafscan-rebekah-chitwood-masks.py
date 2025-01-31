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

  # Convert RGB to HSV and extract the Saturation channel
  device, h = pcv.rgb2gray_hsv(img, 'h', device, args.debug)
  
  # Threshold the Saturation image
  device, h_thresh = pcv.binary_threshold(h, 70, 255, 'dark', device, args.debug)
  device, h_cnt = pcv.binary_threshold(h, 70, 255, 'dark', device, args.debug)
  
  # Fill small objects
  device, img_fill = pcv.fill(h_thresh, h_cnt, 600, device, args.debug)
  
  # Dilate image to fill speckles in image
  device,dilate=pcv.dilate(img_fill, 3, 3, device, args.debug)
  
  # Erode image to reduce dilation back
  device, er_img=pcv.erode(dilate, 3, 3, device, args.debug)
  
  # find objects
  device, id_objects, obj_hierarchy = pcv.find_objects(img, er_img, device, args.debug)
  
  # Define ROI
  device, roi, roi_hierarchy = pcv.define_roi(img, 'rectangle', device, None, 'default', args.debug, True, 
                                             100, 200, -100, -150)  
  # Decide which objects to keep
  device,roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(img,'partial',roi,roi_hierarchy,id_objects,obj_hierarchy,device, args.debug)
  
  for i in range(0, len(roi_objects)):
    # Object combine kept objects
    device, obj, mask = pcv.object_composition(img, [roi_objects[i]], np.array([[roi_obj_hierarchy[0][i]]]), 
                                            device, args.debug)

    y=i+1
    maskname=(str(args.outdir)+"/"+str(filename[:-4])+"-mask-"+str(y)+".jpg")
    pcv.print_image(mask,maskname)
  
if __name__ == '__main__':
  main()

