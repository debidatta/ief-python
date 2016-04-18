#import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
sys.path.append('../../python/')
import caffe
import matplotlib.cm as cm
import matplotlib.mlab as ml
from PIL import Image
import os


def get_heatmap_from_kp(cx, cy, w, h):
    x = np.linspace(1, w, w)
    y = np.linspace(1, h, h)
    X, Y = np.meshgrid(x, y)
    Z = ml.bivariate_normal(X, Y, 5, 5, cx, cy)
    if np.max(Z) != 0:
        Z = Z/np.max(Z)
    return Z

def get_kps(im_file, net, size=224):
    imo = cv2.imread(im_file)
    im = cv2.resize(imo, (size, size),
                             interpolation=cv2.INTER_NEAREST)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.load('/home/debidatd/parsenet/examples/keypoint_detection/means/VGG_mean.npy').mean(1).mean(1) #np.array((102.9801, 115.9465, 122.7717))
    in_ = in_.transpose((2,0,1))
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, 24, size, size)
    net.blobs['data'].data[0,:3,:,:] = in_
    # run net and take argmax for prediction
    curr_kps_file = 'curr_kps_bn/curr_kps_test_0.txt'
    with open(curr_kps_file) as f:
         curr_kp = [map(float, l.strip().split()) for l in f.readlines()]
    curr_kp = [kp for kp in curr_kp[0]]
    print len(curr_kp)
    for n in xrange(10):
        print n 
        for i in xrange(0, len(curr_kp), 2):
            x = curr_kp[i]
            y = curr_kp[i+1]
            kp_heatmap =  get_heatmap_from_kp(x, y, size, size)
            if i==0:
                shape = kp_heatmap.shape
                kps = np.zeros((shape[0], shape[1], len(curr_kp)/2), dtype=np.float32)
            kps[:,:,i/2] = kp_heatmap
        net.blobs['data'].data[0,3:,:,:] = kps.transpose((2,0,1))
        net.forward()
        errors = net.blobs['fc8_err_kps'].data[0,:].copy()
        new_kps = []
        for j in xrange(0, len(curr_kp), 2):
            del_x = size*float(errors[j])
            del_y = size*float(errors[j+1])
            #mod = np.sqrt(del_x**2 + del_y**2)
            #if mod <= 20:
            new_kp_x = float(curr_kp[j]) + del_x
            new_kp_y = float(curr_kp[j+1]) + del_y
            #else:
            #    new_kp_x = float(curr_kp[j]) + 20 * del_x / mod
            #    new_kp_y = float(curr_kp[j+1]) + 20 * del_y / mod
            print new_kp_x, new_kp_y,
            new_kps.extend([new_kp_x, new_kp_y])
        curr_kp = new_kps
    kps = (len(curr_kp)/2)*[0]
    h, w, c = imo.shape
    print len(curr_kp)
    for i in xrange(0, len(curr_kp), 2): 
        kps[i/2] = [curr_kp[i]/224.0*w, curr_kp[i+1]/224.0*h]
    return im, kps


deploy_net = sys.argv[1]
caffemodel = sys.argv[2]
im_file_1 = sys.argv[3]
im_file_2 = sys.argv[4]
caffe.set_mode_cpu()
# load net
net = caffe.Net(deploy_net, caffemodel, caffe.TEST)
# shape for input (data blob is N x C x H x W), set data

im1, kps_1 = get_kps(im_file_1, net)
#im2, kps_2 = get_kps(im_file_2, net)
label_file = '/media/dey/debidatd/keypoint_data/labels/' + os.path.basename(im_file_2)[:-3]+'txt'
kp_file = 'kp_list_manual.txt'
with open(kp_file) as f:
    kp_list = [int(x.strip()) for x in f.readlines()]
with open(label_file) as f:
    kps = [map(int, x.strip().split()[1:]) for x in f.readlines()]

num_kps = len(kps_1)
color=iter(cm.rainbow(np.linspace(0,1,50)))
im = Image.open(im_file_1)
arr = np.asarray(im)
f = plt.figure()
ax1 = f.add_subplot(121)
ax1.imshow(arr)
ax2 = f.add_subplot(122)
im = Image.open(im_file_2)
arr = np.asarray(im)
ax2.imshow(arr)
color = cm.rainbow(np.linspace(0,1,num_kps))
for i in xrange(num_kps):
    col = color[i] 
    x1,y1 = kps_1[i][0], kps_1[i][1]
    x2,y2 = kps[kp_list[i]][0], kps[kp_list[i]][1] #kps_2[i][0], kps_2[i][1]
    if x2 != -1:
        ax1.plot(x1, y1, 'o', markerfacecolor=col, markeredgecolor='k')
        ax2.plot(x2, y2, 'o', markerfacecolor=col, markeredgecolor='k')
f.tight_layout()
plt.show()
