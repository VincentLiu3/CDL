import mxnet as mx
import numpy as np
import logging
from utils import *
from math import sqrt
from autoencoder import AutoEncoderModel
import os
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description = 'Tensor Factorization')
	parser.add_argument('--train' , type = str, default = '', help = 'Training file')
	parser.add_argument('--test' , type = str, default = '', help = 'Testing file')
	parser.add_argument('--feat' , type = str, default = '', help = 'Item feature file')
	parser.add_argument('--p', type = str, default = '1', help = 'result will be save in cdl{p}')

	parser.add_argument('--lambda_u', type = int, default = 1, help = 'lambda_u in CDL')
	parser.add_argument('--lambda_v', type = int, default = 10, help = 'lambda_v in CDL')
	parser.add_argument('--K', type = int, default = 8, help = 'Dimension of latent fectors')
	parser.add_argument('--iter', type = float, default = 100, help = 'Number of iteration')
	parser.add_argument('--batchsize', type = int, default = 256, help = 'Batchsize')
	
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()

	'''
    lambda_u = 1 # lambda_u in CDL
    lambda_v = 10 # lambda_v in CDL
    K = 8
    p = 4 # result will be save in 'cdl%d' % p
    num_iter = 100 # default = 34000?
    batch_size = 256
    train_file = 'data/ml-1m/train.txt'
    test_file = 'data/ml-1m/test.txt'
    feat_file = 'data/ml-1m/item.txt'
    '''
	# set to INFO to see less information during training
	logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
	np.random.seed(1126) # set seed

	lv = 1e-2 # lambda_v/lambda_n in CDL
	dir_save = 'cdl{}'.format(args.p)
	if not os.path.isdir(dir_save):
		os.mkdir (dir_save)
	logging.info('{}: lambda_v/lambda_u/ratio/K: {}/{}/{}/{}'.format(dir_save, args.lambda_v, args.lambda_u,lv, args.K) )  
	with open(dir_save+'/cdl.log','w') as fp:
		fp.write('{}: lambda_v/lambda_u/ratio/K: {}/{}/{}/{}\n'.format(dir_save, args.lambda_v, args.lambda_u,lv, args.K))

	X = loaFeatureData(args.feat) # feature matrix
	R = loadRatingData(args.train) # rating matrix

	#ae_model = AutoEncoderModel(mx.gpu(0), [784,500,500,2000,10], pt_dropout=0.2, internal_act='relu', output_act='relu')
	ae_model = AutoEncoderModel(mx.cpu(2), [X.shape[1], 100, args.K], pt_dropout=0.2, internal_act='relu', output_act='relu')

	train_X = X
	#ae_model.layerwise_pretrain(train_X, 256, 50000, 'sgd', l_rate=0.1, decay=0.0, lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
	#V = np.zeros((train_X.shape[0],10))
	V = np.random.rand(train_X.shape[0], args.K) / 10
	lambda_v_rt = np.ones((train_X.shape[0], args.K))*sqrt(lv)
	U, V, theta, BCD_loss = ae_model.finetune(train_X, R, V, lambda_v_rt, args.lambda_u,
			args.lambda_v, dir_save, args.batchsize,
			args.iter, 'sgd', l_rate=0.1, decay=0.0,
			lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
	#ae_model.save('cdl_pt.arg')
	np.savetxt(dir_save+'/final-U.dat',U,fmt='%.5f',comments='')
	np.savetxt(dir_save+'/final-V.dat',V,fmt='%.5f',comments='')
	np.savetxt(dir_save+'/final-theta.dat',theta,fmt='%.5f',comments='')

	#ae_model.load('cdl_pt.arg')
	Recon_loss = args.lambda_v/lv*ae_model.eval(train_X,V,lambda_v_rt)
	logging.info("Training error: {:.4f}".format(BCD_loss+Recon_loss))
	with open(dir_save+'/cdl.log','a') as fp:
		fp.write("Training error: {:.4f}\n".format(BCD_loss+Recon_loss))
	#print "Validation error:", ae_model.eval(val_X)

	rmse = RMSE(dir_save, args.test)
	logging.info('RMSE: {:.4f}'.format(rmse))
	with open(dir_save+'/cdl.log','a') as fp:
		fp.write('RMSE: {:.4f}\n'.format(rmse))
