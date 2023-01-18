import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
import pickle
import os 
from natsort import natsorted 
from Modules import SCRaPL_class
import gc
import humanize
import psutil
from google.colab import runtime

tfb = tfp.bijectors

#Initialization functions
def sample_init_MCMC(mdl,num_chains,obs):
        num_obs = len(obs)
        dummy_init = mdl.sample()[:-num_obs]
        xx = mdl.sample(num_chains)[:-num_obs]
        ssmps = mdl.sample(value=(*xx,*obs))[:-num_obs],mdl.sample(value=(*dummy_init,*obs))[-num_obs:]
        return ssmps

def sample_init_MLE(mdl,num_obs):
        latent_init = mdl.sample()[:-num_obs]
        var = []
        for ii in range(len(latent_init)):
              var.append(tf.Variable(latent_init[ii]))
        return var

#Helper functions
def est_batch(mdl,up_bdn):

            if mdl.mdl_param(mdl.n_genes)<=up_bdn:
            
                batch_size = tf.cast(mdl.n_genes,dtype=tf.int32) 
            else:
                uu = tf.math.floor(up_bdn/mdl.mdl_param(1))
                batch_size = tf.cast(uu,dtype=tf.int32) 
            

            return batch_size.numpy()   

def perm_lt(samples):
  
          x_1_prm = []
          x_2_prm = []
          x_1_tmp = tf.transpose(samples[-2], perm=[3,1,2,0])
          x_2_tmp = tf.transpose(samples[-1], perm=[3,1,2,0])

          for ii in range(x_1_tmp.shape[2]):

                x_1_prm.append(tf.random.shuffle(x_1_tmp[:,:,ii,:]))
                x_2_prm.append(tf.random.shuffle(x_2_tmp[:,:,ii,:]))

          x_1_prm = tf.stack(x_1_prm,axis=-2)
          x_2_prm = tf.stack(x_2_prm,axis=-2)

          x_1_prm =  tf.transpose(x_1_prm, perm=[3,1,2,0])
          x_2_prm =  tf.transpose(x_2_prm, perm=[3,1,2,0])

          return x_1_prm,x_2_prm

def IC_estimate(mdl,smps,obs):
    
        num_obs = len(obs)

        model_dic_smps = mdl.SCRaPL_post_predict(*(smps))
        model_dic_smps = model_dic_smps.log_prob(obs)

        smps_mean = tuple([tf.reduce_mean(xx,axis=0) for xx in smps])
        model_dic_mns = mdl.SCRaPL_post_predict(*(smps_mean))
        model_dic_mns = model_dic_mns.log_prob(obs)
       
        log_post_mn = tf.reduce_sum(model_dic_mns,axis=-1)
        lppd = tf.reduce_sum(tf.reduce_logsumexp(model_dic_smps,axis=0)-tf.math.log(tf.cast(model_dic_smps.shape[0],dtype=tf.float32)),axis=-1)
        pdic = 2.0*tf.reduce_sum(np.var(model_dic_smps.numpy(), axis=0,ddof=1),axis=-1)
        pwaic = tf.reduce_sum(np.var(model_dic_smps.numpy(), axis=0,ddof=1),axis=-1)

        WAIC = -2.0*(lppd-pwaic)
        DIC = -2.0*(log_post_mn-pdic)
        del model_dic_smps,model_dic_mns, lppd,log_post_mn, pdic,pwaic
        return WAIC,DIC

 
def chain_QC(xx,cr):

      num_genes = len(cr)
      xx = tf.stack([xx[:,cr[ii],ii,:] for ii in range(num_genes)],axis=1)
      return xx

def impute_val(x,ax=0):

        msk =  tf.cast(tf.math.is_finite(x) == True, tf.float32)
        x_msk =tf.where(msk ==0, tf.zeros_like(msk), x)
        imp_mean =  tf.math.divide(tf.reduce_sum(x_msk,axis=ax),tf.reduce_sum(msk,axis=ax))
        try:
              x_msk = tf.math.multiply(1.0-msk,imp_mean)
        except:
              x_msk = tf.math.multiply(1.0-msk,imp_mean[:,tf.newaxis])

        return x_msk

def std_approx(mdl,mle,obs,sv_lnk=None):

        with tf.GradientTape(persistent=True) as hess_tape:
                      hess_tape.watch(mle)
                      with tf.GradientTape() as grad_tape:
                            grad_tape.watch(mle)
                            y = -mdl.log_prob(*(*mle,*obs))
                      grad = grad_tape.gradient(y, mle)
                      hess = []
                      for ii in range(len(mle)):
                            grad_grads = hess_tape.gradient(grad[ii], mle)
                            var = 1.0/grad_grads[ii]
                            dim1,dim2 = var.shape
                            if dim2>1:
                                ax = 1
                            else:
                                ax = 0
                            
                            imp_val = impute_val(var,ax)
                            var = tf.math.sqrt(tf.where( tf.math.logical_or(tf.less(var, 0.0),tf.math.is_nan(var)==True),imp_val,var))
                            hess.append(var)
        return hess


def agr_prts(root_dir,temp,perm, ft_tp):

    dir_temp = root_dir+temp
    dir_perm = root_dir+perm

    files_list = sorted(os.listdir(dir_temp))
    files_smp_list = natsorted([ file for file in files_list if file.endswith(ft_tp+".pickle") & file.startswith("smp_") ])
    files_WAIC_list = natsorted([ file for file in files_list if file.endswith(ft_tp+".pickle") & file.startswith("WAIC_") ])
    files_DIC_list = natsorted([ file for file in files_list if file.endswith(ft_tp+".pickle") & file.startswith("DIC_") ])
    files_ng_cor_list = natsorted([ file for file in files_list if file.endswith(ft_tp+".pickle") & file.startswith("Neg_cor_") ])
    smp_num =  len(set([ file.split('_')[1] for file in files_smp_list]))

    for smp in range(smp_num):
          files_list_tmp = [file for file in files_smp_list if file.split('_')[1] == str(smp)  ]
          smp_col = []
          
          for prt in range(len(files_list_tmp)):

              with open(dir_temp + files_list_tmp[prt], 'rb') as handle:
                    smp_col.append(pickle.load(handle))

          if ft_tp == "MCMC":

            smp_col = tf.concat(smp_col,axis=-2)
            
          else:

            smp_col = tf.concat(smp_col,axis=0)

          if os.path.exists(dir_perm)!= True :
                      os.mkdir(dir_perm)     


          with open(dir_perm+'smp_' + str(smp) + '_' + ft_tp + '.pickle', 'wb') as handle:
                      pickle.dump(smp_col, handle)      


    if len(files_WAIC_list)>0:

          WAIC_num =  len(set([ file.split('_')[2] for file in files_WAIC_list]))
          files_list_tmp = [file for file in files_WAIC_list ]
          WAIC_col = []
          for prt in range(len(files_list_tmp)):

              with open(dir_temp + files_list_tmp[prt], 'rb') as handle:
                    WAIC_col.append(pickle.load(handle))
                    
          WAIC_col = tf.concat(WAIC_col,axis=-1)      

          with open(dir_perm+'WAIC_' + ft_tp + '.pickle', 'wb') as handle:
                        pickle.dump(WAIC_col, handle)  

    if len(files_DIC_list)>0:

          DIC_num =  len(set([ file.split('_')[2] for file in files_DIC_list]))
          files_list_tmp = [file for file in files_DIC_list ]
          DIC_col = []
          for prt in range(len(files_list_tmp)):

              with open(dir_temp + files_list_tmp[prt], 'rb') as handle:
                    DIC_col.append(pickle.load(handle))
                    
          DIC_col = tf.concat(DIC_col,axis=-1)      

          with open(dir_perm+'DIC_' + ft_tp + '.pickle', 'wb') as handle:
                        pickle.dump(DIC_col, handle)  

    if len(files_ng_cor_list)>0:

          Neg_cor_num =  len(set([ file.split('_')[4] for file in files_ng_cor_list]))
          Neg_col = []
          files_list_tmp = [file for file in files_ng_cor_list ]                
          for prt in files_list_tmp:

              with open(dir_temp + prt, 'rb') as handle:
                    Neg_col.append(pickle.load(handle))
                    
          Neg_col = tf.concat(Neg_col,axis=-2)
                       
          with open(dir_perm+'Neg_col_' + ft_tp + '.pickle', 'wb') as handle:
                        pickle.dump(Neg_col, handle)   


#Fit functions
def fit_MCMC(mdl,dataset,num_smps,target_rate,num_burnin_smps,num_chns,prt,save_lnk,kernel='nuts',est_IC = True,est_neg_cor = False): #Avoid setting est_neg_cor to true because it estimates a biased negative control correlation.

        batch_size = est_batch(mdl,20000)
        
        if prt== None:
                prt=1
                
        rep_count = 0 
        time = []
        skp_ind = []
        batch_WAIC_time = 0.0

        yy_1,yy_1_sinf,yy_2,yy_2_sinf = dataset
        prts_tot = tf.math.ceil(tf.shape(yy_1).numpy()[0]/batch_size).numpy()
        
        temp = 'Temp_files/'
        perm = 'Meta_files/'
        fit_type = "MCMC"
        print("Inference will be split into "+str(prts_tot)+" batches of at most "+str(batch_size)+" features to effectively explore posterior space.")
        if kernel == 'nuts':
                sampler = sample_nuts
        elif kernel == 'hmc':
                sampler = sample_hmc
        else:
                raise ValueError('Please implement kernel.')

        while prt < prts_tot+1:                

              yy_1_prt = yy_1[(prt-1)*batch_size:prt*batch_size,:]
              yy_1_sinf_prt = yy_1_sinf[(prt-1)*batch_size:prt*batch_size,:]
              yy_2_prt = yy_2[(prt-1)*batch_size:prt*batch_size,:]
              yy_2_sinf_prt = yy_2_sinf[(prt-1)*batch_size:prt*batch_size,:]

              batch_ft = tf.shape(yy_1_prt)[0]
              model = SCRaPL_class(batch_ft,mdl.n_cells,mdl.err_mdl1,mdl.err_mdl2,yy_1_sinf_prt,yy_2_sinf_prt,mdl.infl)
              model_dist = model.call()
              #model_cor_dist = model.Neg_cor_model()

              init_x,obs = sample_init_MCMC(model_dist,num_chns,(yy_1_prt,yy_2_prt))

              start = timer()
              samples, sampler_stat = sampler(model_dist,obs,init_x,num_smps,num_burnin_smps,target_rate) 
              end = timer()
              batch_time = end-start


              if kernel == 'nuts':                
                    stp_sz = sampler_stat[1]
              else:
                    stp_sz = 0.05
              
              stp_sz = tf.reduce_mean(stp_sz)
              p_accept = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(sampler_stat[0], 0.)))

              qc_acc_rt = tf.math.logical_and(p_accept<0.9,p_accept>0.4)
              qc_stp_sz = stp_sz>0.00001              

              if tf.math.logical_and(qc_acc_rt,qc_stp_sz) == True:

                      #if est_neg_cor == True:

                      #        x_1_prm,x_2_prm = perm_lt(samples)
                              
                      #        #init_cor,obs_cor = sample_init_MCMC(model_cor_dist,(num_smps,num_chns),(*(samples[1:-2]),x_1_prm,x_2_prm))
                      #        #start_cor = timer()
                      #        #samples_neg_cor, sampler_stat_neg_cor = sample_nuts(model_cor_dist,obs_cor,init_cor,1,400,0.65)
                      #        #end_cor = timer()
                      #        #batch_cor_time = end_cor-start_cor

                      #        #with open(save_lnk + temp + 'Neg_cor_exp_prt_'+ str(prt) + '_MCMC.pickle', 'wb') as handle:
                      #        #    pickle.dump(samples_neg_cor[0], handle)  
                      
                      if est_IC ==True:
                          
                              start = timer()    
                              WAIC,DIC = IC_estimate(model,samples,obs)
                              end = timer()
                              batch_WAIC_time = end-start
                              with open(save_lnk + temp + 'WAIC_prt_' + str(prt) + '_MCMC.pickle', 'wb') as handle:
                                  pickle.dump(WAIC, handle) 
                              with open(save_lnk + temp + 'DIC_prt_' + str(prt) + '_MCMC.pickle', 'wb') as handle:
                                  pickle.dump(DIC, handle)
                              
                      time.append(batch_time+ batch_WAIC_time)

                      if save_lnk!=None:

                            if os.path.exists(save_lnk + temp)!= True :
                                    os.mkdir(save_lnk + temp) 

                            for smps in range(len(samples)):
                              
                                with open(save_lnk + temp +'smp_'+str(smps)+'_prt_'+str(prt)+'_MCMC.pickle', 'wb') as handle:
                                    pickle.dump(samples[smps], handle)

                            with open(save_lnk + temp +'stat_smp_prt_'+str(prt)+'_MCMC.pickle', 'wb') as handle:
                                    pickle.dump(sampler_stat, handle)
                            
                            del samples,sampler_stat, model, yy_1_prt, yy_1_sinf_prt, yy_2_prt, yy_2_sinf_prt, batch_ft, model_dist, init_x, obs, stp_sz, qc_stp_sz,  qc_acc_rt, smps, handle, start,end,WAIC
                            gc.collect()
                            tf.keras.backend.clear_session()
                            tf.compat.v1.reset_default_graph()

                      if float(humanize.naturalsize( psutil.virtual_memory().available ).split(' ')[0])<13.0:
                            runtime.unassign()
                      avg_time = tf.reduce_mean(tf.stack(time))
                      rem_time = (tf.cast(prts_tot,dtype=tf.float32)-tf.cast(prt,dtype=tf.float32)) *avg_time
                      print("Completed batch: "+str(prt)+" ({:0.2f}%) ".format(100.0*tf.cast(prt,dtype=tf.float32)/tf.cast(prts_tot,dtype=tf.float32))+"| Remainig time: {:0.2f}s".format(rem_time) )
                      prt+=1
                      rep_count = 0.0
              else:
                      time.append(batch_time)
                      
                      avg_time = tf.reduce_mean(tf.stack(time))
                      rem_time = (tf.cast(prts_tot,dtype=tf.float32)-tf.cast(prt-1,dtype=tf.float32)) *avg_time
                      print("Re-attempting batch " +str(prt)+ " due to failed QC step. Remainig time: {:0.2f}s".format(rem_time) )
                      rep_count+=1
                      
                      if rep_count>3.0:
                          skp_ind.append(prt)
                          print("Skipping batch "+str(prt)+" because it failed QC multiple times")
                          prt+=1
                          rep_count=0.0
                          
                      if float(humanize.naturalsize( psutil.virtual_memory().available ).split(' ')[0])<13.0:
                            runtime.unassign()

        if len(skp_ind) == 0 and save_lnk!=None:
        
                agr_prts(save_lnk,temp,perm, "MCMC")
                    
        elif len(skp_ind) > 0 and save_lnk==None:
            
                print("Batches that failed:",skp_ind)
                
        elif len(skp_ind) > 0 and save_lnk!=None:
            
                print("Batch aggregation was not performed becuase batches ", skp_ind," failed")  
        
        else:
                
                print("Batch aggregation was not performed becuase save link was not provided")
                          
def fit_MLE(mdl,dataset,num_smps,lr,save_lnk):


        batch_size = est_batch(mdl,5000000) 
        
        
        prt = 1
        rep_count = 0 
        time = []
        skp_ind = []
        
        yy_1,yy_1_sinf,yy_2,yy_2_sinf = dataset
        prts_tot = tf.math.ceil(tf.shape(yy_1).numpy()[0]/batch_size).numpy()
        print("Inference will be split into "+str(prts_tot)+" batches of at most "+str(batch_size)+" features to effectively explore posterior space.")
        
        
        while prt < prts_tot+1:   
              
              yy_1_prt = yy_1[(prt-1)*batch_size:prt*batch_size,:]
              yy_1_sinf_prt = yy_1_sinf[(prt-1)*batch_size:prt*batch_size,:]
              yy_2_prt = yy_2[(prt-1)*batch_size:prt*batch_size,:]
              yy_2_sinf_prt = yy_2_sinf[(prt-1)*batch_size:prt*batch_size,:]
        
              batch_ft = tf.shape(yy_1_prt)[0]
              model = SCRaPL_class(batch_ft,mdl.n_cells,mdl.err_mdl1,mdl.err_mdl2,yy_1_sinf_prt,yy_2_sinf_prt,mdl.infl)
              model_dist = model.call()
        
        
              obs = (yy_1_prt,yy_2_prt)
              start = timer()
              samples_MLE,losses =  MLE(model_dist,obs,num_smps,lr) 
              end = timer()
              batch_time = end-start
        
              time.append(batch_time)
        
              if tf.math.is_nan(losses[-1]) != True:
        
                    std = std_approx(model_dist,samples_MLE,obs)
                    
                    temp = 'Temp_files/'
                    perm = 'Meta_files/'
                    fit_type = "MLE"
                    
                    if save_lnk!=None:
        
                              if os.path.exists(save_lnk + temp)!= True :
                                    os.mkdir(save_lnk + temp ) 
        
                              for smps in range(len(samples_MLE)):
                                  
                                        with open(save_lnk + temp +'smp_'+str(2*smps)+'_prt_'+str(prt)+'_MLE.pickle', 'wb') as handle:
                                              pickle.dump(samples_MLE[smps], handle)
        
                                        with open(save_lnk + temp +'smp_'+str(2*smps+1)+'_prt_'+str(prt)+'_MLE.pickle', 'wb') as handle:
                                              pickle.dump(tfb.Softplus().inverse(std[smps]), handle)                                                
                                        
        
                              del samples_MLE,smps,std
        
                    avg_time = tf.reduce_mean(tf.stack(time))
                    rem_time = (tf.cast(prts_tot,dtype=tf.float32)-tf.cast(prt,dtype=tf.float32)) *avg_time
                    print("Completed batch: "+str(prt)+" ({:0.2f}%) ".format(100.0*tf.cast(prt,dtype=tf.float32)/tf.cast(prts_tot,dtype=tf.float32))+"| Remainig time: {:0.2f}s".format(rem_time) )                            
                    prt+=1
                    rep_count = 0.0
              
              else:
        
                    time.append(batch_time)
        
                    avg_time = tf.reduce_mean(tf.stack(time))
                    rem_time = (tf.cast(prts_tot,dtype=tf.float32)-tf.cast(prt-1,dtype=tf.float32)) *avg_time
                    print("Re-attempting batch " +str(prt)+ " due to failed QC step. Remainig time: {:0.2f}s".format(rem_time) )
                    rep_count+=1
        
                    if rep_count>3.0:
                              skp_ind.append(prt)
                              print("Skipping batch "+str(prt)+" because it failed QC multiple times")
                              prt+=1
                              rep_count=0.0

        if len(skp_ind) == 0 and save_lnk!=None:
        
                agr_prts(save_lnk,temp,perm, "MLE")
                    
        elif len(skp_ind) > 0 and save_lnk==None:
            
                print("Batches that failed:",skp_ind)
                
        elif len(skp_ind) > 0 and save_lnk!=None:
            
                print("Batch aggregation was not performed becuase batches ", skp_ind," failed")  
                
        else:
                print("Batch aggregation was not performed becuase save link was not provided")


def fit_VI(mdl,dataset,num_smps,lr,save_lnk):

        batch_size = est_batch(mdl,2500000) 
        
        prt = 1
        rep_count = 0 
        time = []
        skp_ind = []
        
        yy_1,yy_1_sinf,yy_2,yy_2_sinf = dataset
        prts_tot = tf.math.ceil(tf.shape(yy_1).numpy()[0]/batch_size).numpy()
        print("Inference will be split into "+str(prts_tot)+" batches of at most "+str(batch_size)+" features to effectively explore posterior space.")
        
        
        while prt < prts_tot+1:   
              
              yy_1_prt = yy_1[(prt-1)*batch_size:prt*batch_size,:]
              yy_1_sinf_prt = yy_1_sinf[(prt-1)*batch_size:prt*batch_size,:]
              yy_2_prt = yy_2[(prt-1)*batch_size:prt*batch_size,:]
              yy_2_sinf_prt = yy_2_sinf[(prt-1)*batch_size:prt*batch_size,:]
        
              batch_ft = tf.shape(yy_1_prt)[0]
              model = SCRaPL_class(batch_ft,mdl.n_cells,mdl.err_mdl1,mdl.err_mdl2,yy_1_sinf_prt,yy_2_sinf_prt,mdl.infl)
              model_dist = model.call()
        
        
              obs = (yy_1_prt,yy_2_prt)
              start = timer()
              samples_VI,losses =  factored_VI(model_dist,obs,num_smps,lr) 
              end = timer()
              batch_time = end-start
        
              time.append(batch_time)
        
              if tf.math.is_nan(losses[-1]) != True:
        
                    temp = 'Temp_files/'
                    perm = 'Meta_files/'
                    fit_type = "MCMC"
        
                    if save_lnk!=None:
        
                              if os.path.exists(save_lnk + temp)!= True :
                                    os.mkdir(save_lnk + temp) 
        
                              for smps in range(len(samples_VI)):
        
                                        with open(save_lnk + temp +'smp_'+str(smps)+'_prt_'+str(prt)+'_VI.pickle', 'wb') as handle:
                                                      pickle.dump(samples_VI[smps], handle)
        
                              del samples_VI,smps
        
                    avg_time = tf.reduce_mean(tf.stack(time))
                    rem_time = (tf.cast(prts_tot,dtype=tf.float32)-tf.cast(prt,dtype=tf.float32)) *avg_time
                    print("Completed batch: "+str(prt)+" ({:0.2f}%) ".format(100.0*tf.cast(prt,dtype=tf.float32)/tf.cast(prts_tot,dtype=tf.float32))+"| Remainig time: {:0.2f}s".format(rem_time) )                            
                    prt+=1
                    rep_count = 0.0
              
              else:
        
                    time.append(batch_time)
                    
                    avg_time = tf.reduce_mean(tf.stack(time))
                    rem_time = (tf.cast(prts_tot,dtype=tf.float32)-tf.cast(prt-1,dtype=tf.float32)) *avg_time
                    print("Re-attempting batch " +str(prt)+ " due to failed QC step. Remainig time: {:0.2f}s".format(rem_time) )
                    rep_count+=1
        
                    if rep_count>3.0:
                              skp_ind.append(prt)
                              print("Skipping batch "+str(prt)+" because it failed QC multiple times")
                              prt+=1
                              rep_count=0.0

        if len(skp_ind) == 0 and save_lnk!=None:
        
                agr_prts(save_lnk,temp,perm, "VI")
                    
        elif len(skp_ind) > 0 and save_lnk==None:
            
                print("Batches that failed:",skp_ind)
                
        elif len(skp_ind) > 0 and save_lnk!=None:
            
                print("Batch aggregation was not performed becuase batches ", skp_ind," failed")  
                
        else:
                print("Batch aggregation was not performed becuase save link was not provided")
              
#Core fitting modules

def factored_VI(mdl,obs,nm_steps,lr):

    num_obs = len(obs)
    target_log_prob_fn = lambda *latent : mdl.log_prob(*(*latent,*obs))
    mdl_app = tfp.experimental.vi.build_factored_surrogate_posterior( event_shape=mdl.event_shape_tensor()[:-num_obs])

    opt =  tf.optimizers.Adam(learning_rate=lr)
    losses = tfp.vi.fit_surrogate_posterior( target_log_prob_fn, surrogate_posterior=mdl_app, optimizer=opt, num_steps=nm_steps ,jit_compile=True)

    return mdl_app.trainable_variables,losses

def MLE(mdl,obs,nm_steps,lr):

    num_obs = len(obs)
    latent = sample_init_MLE(mdl,num_obs)
    target_log_prob_fn = lambda *latent : -mdl.log_prob(*(*latent,*obs))

    opt =  tf.optimizers.Adam(learning_rate=lr)
    conv = tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=1.0, window_size=10)
    losses = tfp.math.minimize( lambda: target_log_prob_fn(*latent), optimizer=opt,num_steps=nm_steps,jit_compile=True,  convergence_criterion = conv)
    return latent, losses

@tf.function(autograph=False, jit_compile=True)   
def sample_nuts(model,obs,init_x,num_chain_iter,num_burnin_iter,target_accept_rate):
        
        
        num_warmup_iter = int(0.8*num_burnin_iter)

        unconstrained_bijectors = [tfb.Identity()]*len(init_x)

        log_post = lambda *latent : model.log_prob(*(latent+obs))
        def trace_fn(_, pkr):
            return (
                pkr.inner_results.inner_results.log_accept_ratio,
                pkr.inner_results.inner_results.step_size,
                pkr.inner_results.inner_results.leapfrogs_taken,
                pkr.inner_results.inner_results.target_log_prob,
                pkr.inner_results.inner_results.has_divergence
                  )

        nuts= tfp.mcmc.NoUTurnSampler(
                                target_log_prob_fn=log_post,
                                step_size=0.05,
                                max_tree_depth=6
                                    ) 
        ttk = tfp.mcmc.TransformedTransitionKernel(
                                inner_kernel=nuts,
                                bijector=unconstrained_bijectors
                                                    )
        adapted_kernel=tfp.mcmc.DualAveragingStepSizeAdaptation(
                                inner_kernel=ttk,
                                num_adaptation_steps=num_warmup_iter,
                                target_accept_prob= target_accept_rate
                                )#
        
        states , sampler_stat =tfp.mcmc.sample_chain(
                        num_results=num_chain_iter,
                        num_burnin_steps=num_burnin_iter,
                        current_state=init_x,
                        kernel=adapted_kernel,
                        trace_fn=trace_fn) 
        
        return states, sampler_stat

@tf.function(autograph=False, jit_compile=True)   
def sample_hmc(model,obs,init_x,num_chain_iter,num_burnin_iter,target_accept_rate):
        
        
        num_warmup_iter = int(0.8*num_burnin_iter)

        unconstrained_bijectors = [tfb.Identity()]*len(init_x)
        
        log_post = lambda *latent : model.log_prob(*(latent+obs))
        
        def trace_fn(_, pkr):
            return (
                pkr.inner_results.inner_results.log_accept_ratio,
                  )
      
        hmc= tfp.mcmc.HamiltonianMonteCarlo(
                                target_log_prob_fn=log_post,
                                step_size=0.05,
                                num_leapfrog_steps=4
                                    ) 
        ttk = tfp.mcmc.TransformedTransitionKernel(
                                inner_kernel=hmc,
                                bijector=unconstrained_bijectors
                                                    )
        adapted_kernel=tfp.mcmc.DualAveragingStepSizeAdaptation(
                                inner_kernel=ttk,
                                num_adaptation_steps=num_warmup_iter,
                                target_accept_prob= target_accept_rate)
        
        states , sampler_stat =tfp.mcmc.sample_chain(
                        num_results=num_chain_iter,
                        num_burnin_steps=num_burnin_iter,
                        current_state=init_x,
                        kernel=adapted_kernel,
                        trace_fn=trace_fn) 
        
        return states, sampler_stat