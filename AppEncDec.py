# The implementation of GDN is inherited from
# https://github.com/jorge-pessoa/pytorch-gdn,
# under the MIT License. The source code is 
# also related to an implementation
# of the arithmetic coding by Nayuki from
# https://github.com/nayuki/Reference-arithmetic-coding
# under the MIT License.
#
# This file is being made available under the BSD License.  
# Copyright (c) 2021 Yueyu Hu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob

import numpy as np
import pickle

import math
import time

from networks import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "command", choices=["compress", "decompress"],
      help="What to do? Choose from `compress` and `decompress`")
  parser.add_argument(
      "input", nargs="?",
      help="Input filename.")
  parser.add_argument(
      "output", nargs="?",
      help="Output filename.")
  parser.add_argument(
      "--qp", default=1, type=int,
      help="Quality parameter, choose from [1~7] (model0) or [1~8] (model1)"
  )
  parser.add_argument(
      "--model_type", default=0, type=int,
      help="Model type, choose from 0:PSNR 1:MS-SSIM"
  )
  parser.add_argument(
      "--save_recon", default=0, type=int,
      help="Whether to save reconstructed image in the encoding process."
  )

  parser.add_argument(
      "--device", default='cpu', type=str,
      help="Which device does the network run on?"
  )

  args = parser.parse_args()

  if args.command == "compress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for compression.")
    compress_low(args)
  elif args.command == "decompress":
    if args.input is None or args.output is None:
      raise ValueError("Need input and output filename for decompression.")
    decompress_low(args)
