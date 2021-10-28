import sys

sys.path.append("D:/Coding/Deep Learning/G2Wave_Classification/src/pip_installs_required/EfficientNet-PyTorch-master/")
sys.path.append("pip_installs_required/timm_master/")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations
import torch.nn as nn
import timm
import torch.nn.functional as F
import gc
import cv2
import warnings

from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from Lib import copy 
from sklearn import model_selection
