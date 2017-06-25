import mxnet as mx
import numpy as np
import logging
from utils import *
from math import sqrt
from autoencoder import AutoEncoderModel
import os

if __name__ == '__main__':
    lambda_u = 1 # lambda_u in CDL
    lambda_v = 10 # lambda_v in CDL
    K = 8 
    p = 4 # result will be save in 'cdl%d' % p
    num_iter = 100 # default = 34000?
    batch_size = 256

    np.random.seed(1126) # set seed
    lv = 1e-2 # lambda_v/lambda_n in CDL
    dir_save = 'cdl%d' % p
    if not os.path.isdir(dir_save):
        os.system('mkdir %s' % dir_save)
    print('p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d' % (p,lambda_v,lambda_u,lv,K) )   
    with open(dir_save+'/cdl.log','w') as fp:
        fp.write('p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d\n' % (p,lambda_v,lambda_u,lv,K))
    
    X = loaFeatureData('data/ml-1m/item.txt') # feature matrix
    R = loadRatingData('data/ml-1m/train.txt') # rating matrix
    # set to INFO to see less information during training
    logging.basicConfig(level=logging.DEBUG)
    #ae_model = AutoEncoderModel(mx.gpu(0), [784,500,500,2000,10], pt_dropout=0.2,
    #    internal_act='relu', output_act='relu')
    ae_model = AutoEncoderModel(mx.cpu(2), [X.shape[1],100,K], pt_dropout=0.2, internal_act='relu', output_act='relu')

    train_X = X
    #ae_model.layerwise_pretrain(train_X, 256, 50000, 'sgd', l_rate=0.1, decay=0.0,
    #                         lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    #V = np.zeros((train_X.shape[0],10))
    V = np.random.rand(train_X.shape[0],K) / 10
    lambda_v_rt = np.ones((train_X.shape[0],K))*sqrt(lv)
    U, V, theta, BCD_loss = ae_model.finetune(train_X, R, V, lambda_v_rt, lambda_u,
            lambda_v, dir_save, batch_size,
            num_iter, 'sgd', l_rate=0.1, decay=0.0,
            lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
    #ae_model.save('cdl_pt.arg')
    np.savetxt(dir_save+'/final-U.dat',U,fmt='%.5f',comments='')
    np.savetxt(dir_save+'/final-V.dat',V,fmt='%.5f',comments='')
    np.savetxt(dir_save+'/final-theta.dat',theta,fmt='%.5f',comments='')

    #ae_model.load('cdl_pt.arg')
    Recon_loss = lambda_v/lv*ae_model.eval(train_X,V,lambda_v_rt)
    print("Training error: %.3f" % (BCD_loss+Recon_loss))
    with open(dir_save+'/cdl.log','a') as fp:
        fp.write("Training error: %.3f\n" % (BCD_loss+Recon_loss))
    #print "Validation error:", ae_model.eval(val_X)

    rmse = RMSE(p, 'data/ml-1m/train.txt')
    print('RMSE: {:.4f}'.format(rmse))
    with open(dir_save+'/cdl.log','a') as fp:
	fp.write('RMSE: {:.4f}'.format(rmse))
