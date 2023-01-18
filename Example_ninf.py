import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
import os 
import pandas as pd

Root_Folder = "/content/drive/MyDrive"
code_parent_folder = "/SCRaPL_software"
os.chdir('/content/drive/My Drive/' + code_parent_folder)

#Load Error models
from Models import SCRaPL
from Error_Models import *
from Fit_Model import *
from Post_analysis import *

tfd = tfp.distributions
tfb = tfp.bijectors
tfde = tfp.experimental.distributions

with open(Root_Folder + code_parent_folder + "/Synth/" + "Synth_Met.pickle", 'rb') as handle:
      yy_met = pickle.load(handle)
with open(Root_Folder + code_parent_folder + "/Synth/" + "Synth_CpG.pickle", 'rb') as handle:
      CpG = pickle.load(handle)
with open(Root_Folder + code_parent_folder + "/Synth/" + "Synth_Exp.pickle", 'rb') as handle:
      yy_exp = pickle.load(handle)
with open(Root_Folder + code_parent_folder + "/Synth/" + "Synth_Nrm.pickle", 'rb') as handle:
      nrm = pickle.load(handle)

SCRaPL_model = SCRaPL([yy_met,CpG,yy_exp,nrm],[Bin_dist,Poiss_dist],inflation="none",save_link=Root_Folder + code_parent_folder + "/")
SCRaPL_model.fit("nuts",num_chains = 2)
