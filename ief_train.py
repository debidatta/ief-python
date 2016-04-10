from __future__ import division
import sys
import argparse
from shutil import copyfile

caffe_root = '/home/debidatd/parsenet/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np
from find_errors import find_errors
from add_errors import add_errors

parser = argparse.ArgumentParser()
parser.add_argument("solver", help="path to solver prototxt")
parser.add_argument("weights", help="Initial weights")
parser.add_argument("--runs", help="Number of runs", default=4, type=int)
parser.add_argument("log_train", help="File to log train error")
parser.add_argument("log_test", help="File to log test error")
parser.add_argument("vgg_net", help="Train prototxt path of network from where RGB weights will be copied")
parser.add_argument("vgg_weights", help="Caffemodel path of network from where RGB weights will be copied")
parser.add_argument("caffemodel", help="Prefix of caffemodels will be stored after every run. This is independent of solver snapshotting.")
parser.add_argument("--gpu", help="GPU ID", default=0, type=int)
parser.add_argument("--resume", help="Flag to know if training is resuming or being done from scratch", action="store_true")
parser.add_argument("--train_iter", help="Number of train iterations per run", default=750, type=int)
parser.add_argument("--test_iter", help="Number of test iterations", default=50, type=int)
parser.add_argument("--test_interval", help="Test interval or run test after test_interval number of training iterations", default=50, type=int)
parser.add_argument("--kps_folder", help="Folder to store keypoints after each run and error in keypoint", default='curr_kps/', type=int)
args = parser.parse_args()

# caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(args.gpu)
solver = caffe.SGDSolver(args.solver)
solver.net.copy_from(args.weights)

# We want to copy RGB weights from a pretrained network when we start training
if not args.resume:
    net_vgg = caffe.Net(args.vgg_net, args.vgg_weights, caffe.TEST) 
    solver.net.params['conv1d'][0].data[:,:3,:,:] = np.copy(net_vgg.params['conv1'][0].data[:,:3,:,:])
    solver.test_nets[0].params['conv1d'][0].data[:,:3,:,:] = np.copy(net_vgg.params['conv1'][0].data[:,:3,:,:])
    del net_vgg

train_loss = np.zeros(args.train_iter)
test_loss = np.zeros(args.train_iter)
f = open(args.log_train, 'w')
g = open(args.log_test, 'w')

copyfile('curr_kps_train.txt', os.path.join(args.kps_folder, 'curr_kps_train_0.txt')
copyfile('curr_kps_test.txt', os.path.join(args.kps_folder, 'curr_kps_test_0.txt')
train_net = 'prototxts/error_train.txt'
test_net = 'prototxts/error_test.txt'
test_niter = 25
train_iter = 317

for n in xrange(args.runs):
    f.write("Begin run: %s\n"%n)
    g.write("Begin run: %s\n"%n)
    if n > 0:
        solver = caffe.SGDSolver(args.solver)
        solver.net.copy_from(last_run_snapshot)    
    for i in xrange(args.train_iter):
        solver.step(1)
        train_loss[i] = solver.net.blobs['loss'].data
        f.write('{} {}\n'.format(i, np.sqrt(train_loss[i])))
        f.flush()
        if i%args.test_interval  == 0:
            for j in xrange(test_iter):
                solver.test_nets[0].forward()
                test_loss[i] += solver.test_nets[0].blobs['loss'].data
            test_loss[i] = test_loss[i]/test_iter
            g.write('{} {}\n'.format(i, np.sqrt(test_loss[i])))
            g.flush()
    f.write("End run: %s\n*********\n"%n)
    g.write("End run: %s\n*********\n"%n)
    last_run_snapshot = '%s_run%s.caffemodel'%(args.caffemodels, n)
    solver.net.save(last_run_snapshot)
    del solver
   
    error_kps_file = os.path.join(args.kps_folder, 'error_train_%s.txt'%(n))
    find_errors(train_net, last_run_snapshot, error_kps_file, train_niter, 40)
    curr_kps_train_file = os.path.join(args.kps_folder, 'curr_kps_train_%s.txt'%n)
    add_errors(curr_kps_train_file, error_kps_file, 'curr_kps_train.txt')
    copyfile('curr_kps_train.txt', os.path.join(args.kps_folder, 'curr_kps_train_%s.txt'%(n+1)))

    error_kps_file = os.path.join(args.kps_folder, 'error_test_%s.txt'%(n))
    find_errors(test_net, last_run_snapshot, error_kps_file, test_niter, 40)
    curr_kps_test_file = os.path.join(args.kps_folder, 'curr_kps_test_%s.txt'%n)
    add_errors(curr_kps_test_file, error_kps_file, 'curr_kps_test.txt')
    copyfile('curr_kps_test.txt', os.path.join(args.kps_folder, 'curr_kps_test_%s.txt'%(n+1))

f.close()
g.close()
