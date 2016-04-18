from __future__ import division
import sys

caffe_root = '/home/debidatd/parsenet/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

def find_errors(prototxt, caffemodel, error_file, batch_count, batch_size, gpu_id):
    # init
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt,
                    caffemodel,
                    caffe.TEST)
    with open(error_file, 'w') as f:
        for i in range(batch_count):
            print i
            net.forward()
            for j in range(batch_size):
                errors = net.blobs['fc8_err_kps'].data[j,:]
                true_labels = net.blobs['error_kps'].data[j,:]
                for k in xrange(42):
                    f.write('%s '%errors[k])
                f.write("\n")
    
if __name__ == "__main__":
    prototxt = sys.argv[1]
    caffemodel = sys.argv[2]
    batch_count = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    error_file = sys.argv[3]
    find_errors(prototxt, caffemodel, error_file, batch_count, batch_size, 1)
