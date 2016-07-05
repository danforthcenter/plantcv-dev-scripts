#!/usr/bin/env python

import argparse
import sys, os
import re
import shutil
import numpy as np

### Parse command-line arguments
def options():
  parser = argparse.ArgumentParser(description="Get file names to run cufflinks over")
  parser.add_argument("-d", "--directory", help="directory to run script over.")
  parser.add_argument("-p", "--pipeline", help="directory to run script over.")
  parser.add_argument("-o", "--outdir", help="directory to run script over.")
  args = parser.parse_args()
  return args

def build_leafscan(directory,pipeline,outdir):
    dirs=os.listdir(directory)
    files=[]
    path=[]
    
    for x in dirs:
      if re.search('.jpg',x):
        b=str(directory)+str(x)
        path.append(b)
            
    for i,a in enumerate(path):
      b=(str(pipeline)+" -i "+str(a)+" -o "+str(outdir))      
      files.append(b)
      print(b)
            
    files1='\n'.join(files)
    
    shellname=str(outdir)+"/leafscan.sh"
    fwrite=os.open(shellname,os.O_RDWR|os.O_CREAT)
    os.write(fwrite,files1)
    os.close(fwrite)
    

### Main pipeline
def main():
  # Get options
  args = options()
  
  build_leafscan(args.directory,args.pipeline,args.outdir)
  

if __name__ == '__main__':
  main()