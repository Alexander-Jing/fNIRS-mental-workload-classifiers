import os
import sys
import numpy as np
import torch
import torch.nn as nn

import argparse
import time
import pandas as pd

from easydict import EasyDict as edict
from tqdm import trange
from torchsummary import summary
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns

YOUR_PATH = "/home/jyt/workspace/fNIRS_models/code_data_tufts"
sys.path.insert(0, YOUR_PATH + '/fNIRS-mental-workload-classifiers/helpers')
import models
import brain_data
from utils import generic_GetTrainValTestSubjects, seed_everything, makedir_if_not_exist, plot_confusion_matrix, save_pickle, train_one_epoch, eval_model, save_training_curves_FixedTrainValSplit, write_performance_info_FixedTrainValSplit, write_program_time, write_inference_time
from utils import LabelSmoothing, train_one_epoch_fNIRS_T, eval_model_fNIRST, train_one_epoch_Ours_T, eval_model_OursT
from utils import EarlyStopping


#%matplotlib inline

x=[1,2,3,4,5,6]
y = [73.21, 70.52, 73.07, 72.86, 70.06, 69.88]
std_err = [17.24, 16.87, 16.18, 18.05, 17.05, 17.77]

error_params = dict(elinewidth=2,ecolor='black',capsize=4)

plt.figure(figsize=(11, 5))
#plt.style.use('default')
plt.rcParams["font.family"] = 'Times'
plt.bar(x,y,color=['pink','xkcd:light red','xkcd:marigold','xkcd:orange','xkcd:faded blue','xkcd:sea blue'],yerr=std_err,error_kw=error_params,\
                    width=0.7)
plt.xticks(x, ['Ours','Ours+CNN','Ours\n+MixUp','Ours\n+MixUp\n+Real',\
                        'Ours+CNN\n+MixUp','Ours+CNN\n+MixUp\n+Real'], fontproperties = 'Times', fontsize=16)
plt.grid(True,axis='y',ls=':',color='black',alpha=0.6)
plt.ylabel("Average \n accuracy(%)",fontdict={'family' : 'Times', 'size'   : 16})
plt.rcParams['pdf.fonttype'] = 42  # modify the image type for paper submission
plt.show()
plt.savefig('mixup_pdf_6.pdf', bbox_inches='tight')
