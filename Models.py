import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from natsort import natsorted
import pickle
import os
from timeit import default_timer as timer

from Error_Models import *
from Fit_Model import *
from Post_analysis import *
from Modules import SCRaPL_class

tfb = tfp.bijectors
tfd = tfp.distributions


class SCRaPL():

              def __init__(self,data, error_models,inflation,save_link=None, **kwargs):
                        super(SCRaPL, self).__init__(**kwargs)       
                        self.error_model1,self.error_model2 = error_models
                        self.inflation = inflation
                        self._gen_joint_distr(data)
                        if save_link != None:
                            self.save_link = save_link
                            

              def  _load_meta(self):

                        dir = self.save_link + "Meta_files/"
                        meta_file_list = natsorted(os.listdir(dir))

                        fit_alg_list = list(set([file.split('_')[2] for file in meta_file_list if file.split('_')[0]!="DIC" & file.split('_')[0]!="WAIC" ]))
                        fit_alg_list = [file.split('.')[0] for file in fit_alg_list]

                        if len(fit_alg_list) ==0:
                              print("No iference metadata have been found. Please infer parameters using fit funnction.")
                        elif len(fit_alg_list) == 1:
                             
                             fit_alg_list = fit_alg_list[0]
                             meta_file_list = [file for file in meta_file_list if file.endswith(fit_alg_list +".pickle")]
                        else:

                            fit_alg_list = input("Enter type of Metadata from :"+ ' '.join(fit_alg_list)+ " " )
                            meta_file_list = [file for file in meta_file_list if file.endswith(fit_alg_list +".pickle")] 

                        return meta_file_list, fit_alg_list
                                           

              def _synth_data_gen(self,err_mdl):

                            if err_mdl == Bin_dist:
                                    CpG_vls = tf.math.round(tf.random.uniform(shape=(self.n_genes,self.n_cells),minval=1,maxval=3,dtype=tf.float32))
                                    CpG = tf.math.round(tf.random.uniform(shape=(self.n_genes,self.n_cells),minval=50,maxval=500,dtype=tf.float32))

                                    return  tf.where(tf.equal(CpG_vls, 0), tf.zeros_like(CpG),CpG)

                            elif err_mdl == Poiss_dist:
                                    
                                    return tf.repeat(tf.random.uniform(shape=(1,self.n_cells),minval=0.5,maxval=1.5,dtype=tf.float32),repeats=tf.cast(self.n_genes,dtype=tf.int32),axis=0)
                            else: 
                                    raise Exception("For error model correct labels are Bin_dist or Poiss_dist")                      

              def _gen_joint_distr(self,data):

                        if len(data) == 4:
                                self.data_lay1,self.data_lay1_sinf,self.data_lay2,self.data_lay2_sinf = data
                                self.n_genes,self.n_cells = tf.shape(self.data_lay1)
                        elif len(data) == 2:
                                self.n_genes,self.n_cells = data

                                self.data_lay1_sinf = self._synth_data_gen(self.error_model1)
                                self.data_lay2_sinf = self._synth_data_gen(self.error_model2)                             
                                
                        else:
                                raise Exception("Please inpute list of multi-omics data or number of genes/cells to generate synthetic dataset")
                        
                        self.model_class_full = SCRaPL_class(self.n_genes,self.n_cells,self.error_model1,self.error_model2,self.data_lay1_sinf,self.data_lay2_sinf,self.inflation)
                        self.model_class_full.inflation_inf()
                        self.mdl_full = self.model_class_full.call()

                        if len(data) == 2:
                                self.data_lay1,self.data_lay2 = self.sample_from_model()[-2:]

                        return self.mdl_full
                        
              def sample_from_model(self):                               
                    
                    return self.mdl_full.sample() 


              def fit(self,algo,num_chains,prt=None):

                    start_all = timer()
                    if self.save_link!=None:
                          if os.path.exists(self.save_link)!= True :
                              raise Exception("Please choose an existing path to save posterior samples.")

                    dataset = (self.data_lay1,self.data_lay1_sinf,self.data_lay2,self.data_lay2_sinf)
                    if algo == 'nuts' or algo =='hmc':
                          
                            fit_MCMC(self.model_class_full,dataset,3000,0.751,15000,num_chains,prt,est_IC=True,save_lnk=self.save_link,kernel=algo)
                    
                    elif algo == 'mle':

                            fit_MLE(self.model_class_full,dataset,1000,0.05,save_lnk=self.save_link)

                    elif algo == 'vi':
                        
                            fit_VI(self.model_class_full,dataset,2000,0.015,save_lnk=self.save_link)

                    else:
                        pass
                    
                    end_all = timer()
                    print("Total fit mcmc time",end_all-start_all)

                    

              def build_post(self):

                      meta_file_list, fit_alg = self._load_meta()
                      var_ind = 0
                      dir = self.save_link + "Meta_files/"
                      if fit_alg == "MCMC":

                            WAIC_ind = 0
                            var_all = []
                            if meta_file_list[0].split('_')[0] == 'WAIC':
                                    with open(dir + meta_file_list[0], 'rb') as handle:
                                        WAIC = pickle.load(handle)
                                        cr = tf.math.argmin(WAIC,axis=0).numpy()
                                    WAIC_ind+=1

                            for fl in meta_file_list[WAIC_ind:]:

                                    with open(dir + fl, 'rb') as handle:
                                        var = pickle.load(handle)
                                        var_all.append(tf.reduce_mean(chain_QC(var,cr),axis=0))
                                    var_ind +=1 
                            var_all = tuple(var_all)
                            mdl = self.model_class_full.SCRaPL_post_predict(*(var_all[:-2]))
                            

                      elif fit_alg == "MLE":
                            #Not very Accurate.
                            var_all = []

                            for fl in meta_file_list:

                                    if var_ind % 2 == 0:
                                        with open(dir + fl, 'rb') as handle:
                                            var = pickle.load(handle)
                                            var_all.append(var)
                                    var_ind +=1 

                            var_all = tuple(var_all)
                            mdl = self.model_class_full.SCRaPL_post_predict(*(var_all[:-2]))

                      else:
                            #Not very Accurate. Later versions will use VI as proposal distribution.
                            mdl = tfp.experimental.vi.build_factored_surrogate_posterior( event_shape= self.mdl_full.event_shape_tensor()[:-2])
                            for fl in meta_file_list:

                                      with open(dir + fl, 'rb') as handle:
                                            mdl.variables[var_ind].assign(pickle.load(handle))

                                      var_ind+=1

                      return mdl          

              def lt_param_analysis(self,thres):

                      meta_file_list, fit_alg = self._load_meta()
                      dir = self.save_link + "Meta_files/"
                      if fit_alg != "MCMC":
                          raise Exception("Analysis is only available for MCMC fitting algorithm.")

                      DIC_ind = 0
                      var_stats_col = []
                      if meta_file_list[0].split('_')[0] == 'DIC':
                              with open(dir + meta_file_list[0], 'rb') as handle:
                                  DIC = pickle.load(handle)
                                  cr = tf.math.argmin(DIC,axis=0).numpy()
                              DIC_ind+=1

                      for fl in meta_file_list[DIC_ind:DIC_ind+1]:

                            with open(dir + fl, 'rb') as handle:
                                    var = pickle.load(handle)
                            
                            var = chain_QC(var,cr)      
                            var = np.squeeze(var.numpy())
                            density = Est_smooth_1D_density(var)
                            alpha = np.linspace(0.95,0.5)
                            for al in alpha:
                                  EFDR, num_features,feature_ind, p_bay = EFDR_cor(density,al,thres)
                                  if EFDR<0.1 or np.isnan(EFDR.numpy())==True:
                                            break

                            
                            var_stats_col.append([EFDR, num_features,feature_ind, p_bay,density])

                      return var_stats_col   
                  