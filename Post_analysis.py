import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import pickle
from sklearn.neighbors import KernelDensity

tfb = tfp.bijectors

def integral(y, x):
    dx = (x[:,-1] - x[:,0]) / (int(x.shape[1]) - 1)
    return  tf.multiply((y[:,0] + y[:,-1])/2+tf.reduce_sum(y[:,1:-1],axis=1) , dx)

def Est_smooth_1D_density(smps):

      k_density = np.apply_along_axis(lambda x: KernelDensity(bandwidth=0.1, kernel='gaussian').fit(x[:,None]),0, smps)
      return k_density
      
def EFDR_cor(k_density,alpha,gam):

      x_d = np.linspace(-3, 3, 400)
      p_bay  = np.zeros(len(k_density))

      for ii in range(p_bay.shape[0]):
          pp =  np.exp(k_density[ii].score_samples(x_d[:,None]))
          p_bay[ii] =  integral(pp[None,np.abs(x_d)<2*np.math.atanh(gam)], x_d[None,np.abs(x_d)<2*np.math.atanh(gam)])

      pp1 = 1-p_bay
      zz1 = tf.greater(pp1,alpha)
      zz2 = tf.less_equal(pp1,alpha)  

      EFDR = tf.reduce_mean(tf.gather(p_bay,tf.where(zz1==True)))

      num_features = tf.shape(tf.where(zz1==True))[0]
      feature_ind = tf.cast(zz1,dtype=tf.int16)

      return EFDR, num_features,feature_ind, p_bay