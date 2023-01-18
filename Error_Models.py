#This file contains the error models. 
import tensorflow_probability as tfp
import tensorflow as tf

tfb = tfp.bijectors
eps=0.001
bin_bij = tfb.Chain([tfb.Shift(eps/2.0),tfb.Scale(scale=1.0-eps),tfb.NormalCDF()])

#Binomial distribution error model.
def Bin_dist(total_cpgs,p_lt):
        p = bin_bij.forward(p_lt)
        Bin_distr = tfp.distributions.Binomial(total_count=total_cpgs,probs=p,name='Binomial')
        
        return Bin_distr

#Poisson distribution error model.    
def Poiss_dist(norm_const,log_rt):
    
        norm_const_lg = tf.math.log(eps+norm_const)
        Poiss_distr = tfp.distributions.Poisson(log_rate=norm_const+log_rt,name='Poisson')
        
        return Poiss_distr
        
#Zero inflated error model.
def ZINF_dist(dist,infl):
            
            pp = tf.stack([infl,1-infl],axis=-1)
            
            if dist.parameters['name']=='Poisson':
                    log_rt = dist.parameters['log_rate']
                    xx = tf.stack([-20*tf.ones_like(log_rt),log_rt],axis=-1)
                    ZINF_dist = tfp.distributions.MixtureSameFamily(
                                                          mixture_distribution = tfp.distributions.Categorical(probs=pp),
                                                          components_distribution = tfp.distributions.Poisson(log_rate=xx)
                                                          )
                                                          
            elif dist.parameters['name']=='Binomial':
                    prb = dist.parameters['probs']
                    tot_cnts = dist.parameters['total_count']
                    
                    tot_cnts_conct = tf.stack([tf.zeros_like(tot_cnts),tot_cnts],axis=-1)
                    xx = tf.stack([tf.zeros_like(prb),prb],axis=-1) 
                    
                    ZINF_dist =  tfp.distributions.MixtureSameFamily(
                                                          mixture_distribution = tfp.distributions.Categorical(probs=pp),
                                                          components_distribution = tfp.distributions.Binomial(total_count=tot_cnts_conct,probs=xx)
                                                          )
                                                          
            else:
                    Exception("Mix_dist uses Poisson or Binomial error models. Please make sure you use correct error models with correctly defind names.")
                    
            return ZINF_dist    