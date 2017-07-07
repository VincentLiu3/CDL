import numpy as np
import scipy.sparse

def read_mult(f_in='mult.dat',D=8000):
    fp = open(f_in)
    lines = fp.readlines()
    X = np.zeros((len(lines),D))
    for i,line in enumerate(lines):
        strs = line.strip().split(' ')[1:]
        for strr in strs:
            segs = strr.split(':')
            X[i,int(segs[0])] = float(segs[1])
    arr_max = np.amax(X,axis=1)
    X = (X.T/arr_max).T
    return X

def get_mult():
	X = read_mult('mult.dat',8000).astype(np.float32)
	return X

def get_dummy_mult():
	X = np.random.rand(1000,100)
	X[X<0.9] = 0
	return X

def read_user(f_in='cf-train-1-users.dat',num_u=5551,num_v=16980):
	fp = open(f_in)
	R = np.mat(np.zeros((num_u,num_v)))
	for i,line in enumerate(fp):
		segs = line.strip().split(' ')[1:]
		for seg in segs:
			R[i,int(seg)] = 1
	return R

def read_dummy_user():
	R = np.mat(np.random.rand(2000,1000))
	R[R<0.9] = 0
	R[R>0.8] = 1
	return R

def loaFeatureData(filename):
	'''
	laod triple data (row, column, value) to np array
	'''
	fData = np.loadtxt(filename, delimiter=',')
	fData = fData.T
	fData = scipy.sparse.coo_matrix((fData[2],(fData[0],fData[1])))
	R = fData.toarray()
	return R

def loadRatingData(filename='data/ml-1m/train.txt'):
	'''
	laod triple data (row, column, value) to matrx
	'''
	fData = np.loadtxt(filename, delimiter=',')
	fData = fData.T
	fData = scipy.sparse.coo_matrix((fData[2],(fData[0],fData[1])))
	R = fData.todense()
	return R

def RMSE(p, test_data):
	X_test = loadRatingData(test_data)
	X_test = scipy.sparse.coo_matrix(X_test)
        # read learned U and V
	dir_save = 'cdl{}/'.format(p)
	U = np.loadtxt(dir_save+'final-U.dat')
	V = np.loadtxt(dir_save+'final-V.dat')
	UV = np.dot(U, V.T)
	X_prediton = scipy.sparse.coo_matrix((UV[X_test.row, X_test.col], (X_test.row, X_test.col)), shape=X_test.shape)
	assert( np.all(X_test.row == X_prediton.row) and np.all(X_test.col == X_prediton.col) ), 'predition and origin value must have the same length'
	rmse = np.sqrt( sum( pow(X_test.data-X_prediton.data, 2) ) / len(X_test.data) )
	return rmse

