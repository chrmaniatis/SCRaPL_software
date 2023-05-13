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

tfb = tfp.bijectors
tfd = tfp.distributions

class SCRaPL_class():
        def __init__(self,n_genes,n_cells,error_model1,error_model2,model1_sinf,model2_sinf,inflation, **kwargs):
        
                super(SCRaPL_class, self).__init__(**kwargs)       
                
                self.n_genes = n_genes
                self.n_cells = n_cells
                
                
                self.aff =  tfb.Chain([tfb.Shift(-1.),tfb.Scale(scale=2.)])
                self.aff_inv = tfb.Invert(self.aff)
                self.exp = tfb.Exp()
                self.log = tfb.Invert(self.exp)
                self.tanh = tfb.Tanh()
                self.tanh_inv = tfb.Invert(self.tanh)
                self.sigm = tfb.Sigmoid()
                self.sigm_inv = tfb.Invert(self.sigm)
                self.cor_trsf = tfb.Chain([self.aff_inv,self.tanh,tfb.Scale(scale=0.5)])
                self.cor_trsf_inv = tfb.Invert(self.cor_trsf)   
                
                
                self.cor_bij = tfb.Chain([self.tanh,tfb.Scale(scale=0.5)])
                self.std_bij = tfb.Chain([self.exp,tfb.Scale(scale=-1.0)]) 


                self.err_mdl1 = error_model1
                self.err_mdl2 = error_model2
                self.err_mdl_1_sinf = model1_sinf
                self.err_mdl_2_sinf = model2_sinf
                
                self.infl = inflation
                
        def SCRaPL_main_model(self):
            
                Root = tfd.JointDistributionCoroutine.Root
                cor_lt = yield Root(tfd.Independent(tfd.TransformedDistribution( distribution = tfd.Beta( concentration0 = 15.0*tf.ones([self.n_genes,1]), concentration1=15.0*tf.ones([self.n_genes,1])), bijector= self.cor_trsf_inv, name = "cor_lt" ),reinterpreted_batch_ndims=2))
                m_1_lt = yield Root(tfd.Independent(tfd.Normal(loc=tf.ones([self.n_genes,1]),scale=tf.ones([self.n_genes,1]), name = "m_1_lt"),reinterpreted_batch_ndims=2))
                m_2_lt = yield Root(tfd.Independent(tfd.Normal(loc=4*tf.ones([self.n_genes,1]),scale=tf.ones([self.n_genes,1]), name = "m_exp_lt"),reinterpreted_batch_ndims=2))
                s_1_lt = yield Root(tfd.Independent(tfd.TransformedDistribution( distribution = tfd.InverseGamma(concentration=2.5*tf.ones([self.n_genes,1]),scale=4.5*tf.ones([self.n_genes,1])),bijector= self.log, name = "s_1_lt" ),reinterpreted_batch_ndims=2))
                s_2_lt = yield Root(tfd.Independent(tfd.TransformedDistribution( distribution = tfd.InverseGamma(concentration=2.5*tf.ones([self.n_genes,1]),scale=4.5*tf.ones([self.n_genes,1])),bijector= self.log , name = "s_2_lt"),reinterpreted_batch_ndims=2))
                
                cor = self.cor_bij.forward(cor_lt)
                s_1 = self.std_bij.forward(s_1_lt)
                s_2 = self.std_bij.forward(s_2_lt)
                
                m_1 = tf.math.multiply( m_1_lt,tf.ones([self.n_genes,self.n_cells]))
                s_1 = tf.math.multiply( s_1   ,tf.ones([self.n_genes,self.n_cells]))
                
                
                if self.infl == "lay1":
                
                            infl_lay_1_lt = yield Root(tfd.Independent(tfd.TransformedDistribution( distribution = tfd.Beta( concentration0 =8.0*tf.ones([self.n_genes,1]),concentration1=2.0*tf.ones([self.n_genes,1])), bijector= self.sigm_inv, name = "infl_lay_1" ) ,reinterpreted_batch_ndims=2))                                        
                            infl_lay_1 = tf.math.multiply(self.sigm.forward(infl_lay_1_lt),tf.ones([self.n_genes,self.n_cells]))
                            
                            x_1 = yield tfd.Independent(tfd.Normal(loc = m_1, scale = s_1,name="x_1"),reinterpreted_batch_ndims=2)
                            m_cnd_2 = m_2_lt+tf.math.multiply(tf.math.divide(tf.math.multiply(s_2,x_1-m_1),s_1),cor)
                            s_cnd_2 = tf.math.sqrt(tf.math.multiply(1-tf.math.square(cor),tf.math.square(s_2)))
                            x_2 = yield tfd.Independent(tfd.Normal(loc = m_cnd_2, scale = s_cnd_2,name="x_2"),reinterpreted_batch_ndims=2)
                            
                            y_1 = yield tfd.Independent(ZINF_dist(self.err_mdl1(self.err_mdl_1_sinf,x_1),infl_lay_1),reinterpreted_batch_ndims=2)
                            y_2 = yield tfd.Independent(self.err_mdl2(self.err_mdl_2_sinf,x_2),reinterpreted_batch_ndims=2)
                
                elif self.infl == "lay2":
                
                            infl_lay_2_lt = yield Root(tfd.Independent( tfd.TransformedDistribution( distribution = tfd.Beta( concentration0 =8.0*tf.ones([self.n_genes,1]), concentration1=2.0*tf.ones([self.n_genes,1])), bijector= self.sigm_inv, name = "infl_lay_2" ),reinterpreted_batch_ndims=2))     
                            infl_lay_2 = tf.math.multiply(self.sigm.forward(infl_lay_2_lt),tf.ones([self.n_genes,self.n_cells]))
                            
                            x_1 = yield tfd.Independent(tfd.Normal(loc = m_1, scale = s_1,name="x_1"),reinterpreted_batch_ndims=2)
                            m_cnd_2 = m_2_lt+tf.math.multiply(tf.math.divide(tf.math.multiply(s_2,x_1-m_1),s_1),cor)
                            s_cnd_2 = tf.math.sqrt(tf.math.multiply(1-tf.math.square(cor),tf.math.square(s_2)))
                            x_2 = yield tfd.Independent(tfd.Normal(loc = m_cnd_2, scale = s_cnd_2,name="x_2"),reinterpreted_batch_ndims=2)
                            
                            y_1 = yield tfd.Independent(self.err_mdl1(self.err_mdl_1_sinf,x_1),reinterpreted_batch_ndims=2)
                            y_2 = yield tfd.Independent(ZINF_dist(self.err_mdl2(self.err_mdl_2_sinf,x_2),infl_lay_2),reinterpreted_batch_ndims=2)
                
                elif self.infl == "both":
                
                            infl_lay_1_lt = yield Root(tfd.Independent(tfd.TransformedDistribution( distribution = tfd.Beta( concentration0 =8.0*tf.ones([self.n_genes,1]), concentration1=2.0*tf.ones([self.n_genes,1])), bijector= self.sigm_inv, name = "infl_lay_1" ),reinterpreted_batch_ndims=2))
                            infl_lay_2_lt = yield Root(tfd.Independent(tfd.TransformedDistribution( distribution = tfd.Beta( concentration0 =8.0*tf.ones([self.n_genes,1]), concentration1=2.0*tf.ones([self.n_genes,1])), bijector= self.sigm_inv, name = "infl_lay_2" ),reinterpreted_batch_ndims=2))
                            
                            infl_lay_1 = tf.math.multiply(self.sigm.forward(infl_lay_1_lt),tf.ones([self.n_genes,self.n_cells]))
                            infl_lay_2 = tf.math.multiply(self.sigm.forward(infl_lay_2_lt),tf.ones([self.n_genes,self.n_cells]))
                            
                            x_1 = yield tfd.Independent(tfd.Normal(loc = m_1, scale = s_1,name="x_1"),reinterpreted_batch_ndims=2)
                            m_cnd_2 = m_2_lt+tf.math.multiply(tf.math.divide(tf.math.multiply(s_2,x_1-m_1),s_1),cor)
                            s_cnd_2 = tf.math.sqrt(tf.math.multiply(1-tf.math.square(cor),tf.math.square(s_2)))
                            x_2 = yield tfd.Independent(tfd.Normal(loc = m_cnd_2, scale = s_cnd_2,name="x_2"),reinterpreted_batch_ndims=2)
                            
                            y_1 = yield tfd.Independent(ZINF_dist(self.err_mdl1(self.err_mdl_1_sinf,x_1),infl_lay_1),reinterpreted_batch_ndims=2)
                            y_2 = yield tfd.Independent(ZINF_dist(self.err_mdl2(self.err_mdl_2_sinf,x_2),infl_lay_2),reinterpreted_batch_ndims=2)
                
                elif self.infl == "none":
                
                            x_1 = yield tfd.Independent(tfd.Normal(loc = m_1, scale = s_1,name="x_1"),reinterpreted_batch_ndims=2)
                            m_cnd_2 = m_2_lt+tf.math.multiply(tf.math.divide(tf.math.multiply(s_2,x_1-m_1),s_1),cor)
                            s_cnd_2 = tf.math.sqrt(tf.math.multiply(1-tf.math.square(cor),tf.math.square(s_2)))
                            x_2 = yield tfd.Independent(tfd.Normal(loc = m_cnd_2, scale = s_cnd_2,name="x_2"),reinterpreted_batch_ndims=2)
                            
                            y_1 = yield tfd.Independent(self.err_mdl1(self.err_mdl_1_sinf,x_1),reinterpreted_batch_ndims=2)
                            y_2 = yield tfd.Independent(self.err_mdl2(self.err_mdl_2_sinf,x_2),reinterpreted_batch_ndims=2)
                else:
                            raise Exception("Please add choose from available labels (lay1,lay2,both or none).")

        def SCRaPL_post_predict(self,*param):
            
                def prob_model():
                
                        Root = tfd.JointDistributionCoroutine.Root
        
                        if self.infl == "none":
                            
                                x_1,x_2 = param[5:]
                                y_1 = yield tfd.Independent(self.err_mdl1(self.err_mdl_1_sinf,x_1),reinterpreted_batch_ndims=0)
                                y_2 = yield tfd.Independent(self.err_mdl2(self.err_mdl_2_sinf,x_2),reinterpreted_batch_ndims=0)
        
                        elif self.infl == "lay1":
                        
                                infl_lay_1_lt,x_1,x_2 = param[5:]
                                infl_lay_11 = tf.math.multiply(self.sigm.forward(infl_lay_1_lt),tf.ones([self.n_genes,self.n_cells]))
                                
                                y_1 = yield tfd.Independent(ZINF_dist(self.err_mdl1(self.err_mdl_1_sinf,x_1),infl_lay_11),reinterpreted_batch_ndims=0)
                                y_2 = yield tfd.Independent(self.err_mdl2(self.err_mdl_2_sinf,x_2),reinterpreted_batch_ndims=0)
                        
                        elif self.infl == "lay2":
                        
                                infl_lay_2_lt,x_1,x_2 = param[5:]
                                infl_lay_22 = tf.math.multiply(self.sigm.forward(infl_lay_2_lt),tf.ones([self.n_genes,self.n_cells]))
                                
                                y_1 = yield tfd.Independent(self.err_mdl1(self.err_mdl_1_sinf,x_1),reinterpreted_batch_ndims=0)
                                y_2 = yield tfd.Independent(ZINF_dist(self.err_mdl2(self.err_mdl_2_sinf,x_2),infl_lay_22),reinterpreted_batch_ndims=0)
                                
                        else:
                        
                                infl_lay_1_lt,infl_lay_2_lt,x_1,x_2 = param[5:]
                                infl_lay_11 = tf.math.multiply(self.sigm.forward(infl_lay_1_lt),tf.ones([self.n_genes,self.n_cells]))
                                infl_lay_22 = tf.math.multiply(self.sigm.forward(infl_lay_2_lt),tf.ones([self.n_genes,self.n_cells]))
                                
                                y_1 = yield tfd.Independent(ZINF_dist(self.err_mdl1(self.err_mdl_1_sinf,x_1),infl_lay_11),reinterpreted_batch_ndims=0)
                                y_2 = yield tfd.Independent(ZINF_dist(self.err_mdl2(self.err_mdl_2_sinf,x_2),infl_lay_22),reinterpreted_batch_ndims=0)
                                
                return tfd.JointDistributionCoroutine(prob_model)                        
        
        def inflation_inf(self):
        
                if self.infl == "lay1":
                
                        print("Added inflation to molecular layer 1.")
                
                elif self.infl == "lay2":
                
                        print("Added inflation to molecular layer 2.")
                
                elif self.infl == "both":
                
                        print("Added inflation to both molecular layers.")
                
                else:
                        print("No inflation added.")
        
        
        def mdl_param(self,genes):
                
                genes = tf.constant(genes,dtype=tf.int32) 
                prms = 5*genes+2*genes*self.n_cells
                if self.infl == "lay1":
                        
                        prms += genes
                elif self.infl == "lay2":
                        
                        prms += genes
                
                elif self.infl == "both":
                    
                        prms += 2*genes
                
                else:
                        pass
                
                return prms

        def call(self):
        
                return tfd.JointDistributionCoroutine(self.SCRaPL_main_model)
        
        def sample_latent(self):
                
                mdl = self.call()
                xx = mdl.sample()
                
                smps = []
                for ii in range(len(xx)-2):
                    smps.append(xx[ii])

                return smps
