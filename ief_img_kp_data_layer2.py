import os
import sys
sys.path.append("/home/debidatd/parsenet/python")
import caffe
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib.cm as cm
import os
from PIL import Image

def getboundingbox(kps):
    x = min(kps[:,0])
    y = min(kps[:,1])
    w = max(kps[:,0]) - x
    h = max(kps[:,1]) - y 
    return x, y, w, h

def get_heatmap_from_kp(cx, cy, w, h):
    x = np.linspace(1, w, w)
    y = np.linspace(1, h, h)
    X, Y = np.meshgrid(x, y)
    Z = ml.bivariate_normal(X, Y, 2.5, 2.5, cx, cy)
    if np.max(Z) != 0:
        Z = Z/np.max(Z)
    return Z

class ImageKPDataLayer(caffe.Layer):
    """Data layer used for preparing input images and keypoint heatmaps."""
    def augment_data(self, im, kps, curr_kps):
        # image cropping
        kps = kps.reshape((len(kps)/2, 2))
        curr_kps = curr_kps.reshape((len(curr_kps)/2, 2))
        x, y, w, h = getboundingbox(kps)
        crop_pad_inf = 1.5
        crop_pad_sup = 2.0
        shift = 5
        # bounding rect extending
        inf, sup = crop_pad_inf, crop_pad_sup
        r = sup - inf
        pad_w_r = np.random.rand() * r + inf  # inf~sup
        pad_h_r = np.random.rand() * r + inf  # inf~sup
        x -= (w * pad_w_r - w) / 2
        y -= (h * pad_h_r - h) / 2
        w *= pad_w_r
        h *= pad_h_r
    
        # shifting
        x += np.random.rand() * shift * 2 - shift
        y += np.random.rand() * shift * 2 - shift

        # clipping
        x, y, w, h = [int(z) for z in [x, y, w, h]]
        x = np.clip(x, 0, im.shape[1] - 1)
        y = np.clip(y, 0, im.shape[0] - 1)
        w = np.clip(w, 1, im.shape[1] - (x + 1))
        h = np.clip(h, 1, im.shape[0] - (y + 1))
        im = im[y:y + h, x:x + w]

        # joint shifting
        kps = np.asarray([(j[0] - x, j[1] - y) for j in kps])
        kps = kps.flatten()
        curr_kps = np.asarray([(j[0] - x, j[1] - y) for j in curr_kps])
        curr_kps = curr_kps.flatten()

        return im, kps, curr_kps

    def im_kps_list_to_blob(self, ims, kps):
        """Convert a list of keypoint heatmaps into a network input.
        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        num_images = len(kps)
        blob = np.zeros((num_images, self._resize, self._resize, 3+self._num_keypoints),
                        dtype=np.float32)
        for i in xrange(num_images):
            blob[i, :, :, :3] = ims[i]
            blob[i, :, :, 3:] = kps[i]
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

    def error_kps_list_to_blob(self, error_kps):
        num_images = len(error_kps)
        blob = np.zeros((num_images, 2*self._num_keypoints),
                        dtype=np.float32)
        for i in xrange(num_images):
            blob[i, :] = error_kps[i]
        return blob

    def prep_im_for_blob(self, im):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im -= np.load(self._mean_file).mean(1).mean(1)
        #im_shape = im.shape
        #im_size_min = np.min(im_shape[0:2])
        #im_size_max = np.max(im_shape[0:2])
        #im_scale = float(target_size) / float(im_size_min)
        #im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
        #            interpolation=cv2.INTER_LINEAR)

        return im

    def get_true_kps(self, kps_file, kps_list):
        with open(os.path.join(self._dataset_root, kps_file)) as f:
            kps_all = [map(float, x.strip().split()) for x in f.readlines()]
        kp_list = []
        for i,kp_id in enumerate(kps_list):
            x = kps_all[kp_id][1]
            y = kps_all[kp_id][2]
            kp_list.append(x)
            kp_list.append(y)
        return np.array(kp_list)

    def prep_error_kps_for_blob(self, curr_kps, true_kps):
        return np.array([(true_pt-curr_pt)/self._resize for curr_pt, true_pt in zip(curr_kps, true_kps)])

    def prep_kps_for_blob(self, curr_kp):#, transform_params):
        for i in xrange(0, len(curr_kp), 2):
            x = curr_kp[i]
            y = curr_kp[i+1]
            kp_heatmap =  get_heatmap_from_kp(x, y, self._resize, self._resize)
            if i==0:
                shape = kp_heatmap.shape
                kps = np.zeros((shape[0], shape[1], len(curr_kp)/2), dtype=np.float32)
            kps[:,:,i/2] = kp_heatmap
        return kps    
            
    def get_minibatch(self, minibatch_db):
        num_images = len(minibatch_db)
        processed_ims = []
        processed_kps = []
        processed_error_kps = []
        for i in xrange(num_images):
            im = cv2.imread(os.path.join(self._dataset_root, minibatch_db[i]['im']))
            size = self._resize
            orig_h, orig_w, _ = im.shape
            true_kps = self.get_true_kps(minibatch_db[i]['kp'], self._kps)
            curr_kps = np.array(minibatch_db[i]['curr_kp'])
            curr_kps[0::2] = curr_kps[0::2] / size * float(orig_w)
            curr_kps[1::2] = curr_kps[1::2] / size * float(orig_h)
            if self._phase == 'TRAIN':
                im, true_kps, curr_kps = self.augment_data(im, true_kps, curr_kps)
            true_kps[0::2] = true_kps[0::2] / float(orig_w) * size
            true_kps[1::2] = true_kps[1::2] / float(orig_h) * size
            curr_kps[0::2] = curr_kps[0::2] / float(orig_w) * size
            curr_kps[1::2] = curr_kps[1::2] / float(orig_h) * size
            im = cv2.resize(im, (size, size),
                             interpolation=cv2.INTER_NEAREST)
            im = self.prep_im_for_blob(im)
            kps_heatmaps = self.prep_kps_for_blob(curr_kps)
            error_kps = self.prep_error_kps_for_blob(curr_kps, true_kps) 
            processed_ims.append(im)
            processed_kps.append(kps_heatmaps)
            processed_error_kps.append(error_kps)
        # Create a blob to hold the input images
        im_kps_blob = self.im_kps_list_to_blob(processed_ims, processed_kps)
        error_kps_blob = self.error_kps_list_to_blob(processed_error_kps)

        blobs = {'data': im_kps_blob,
                 'error_kps': error_kps_blob}

        return blobs
 
    def _shuffle_imdb_inds(self):
        """Randomly permute the training roidb."""
        if self._shuffle:
            self._perm = np.random.permutation(np.arange(len(self._imdb)))
        else:
            self._perm = np.arange(len(self._imdb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + self._ims_per_batch >= len(self._imdb):
            self._shuffle_imdb_inds()

        db_inds = self._perm[self._cur:self._cur + self._ims_per_batch]
        self._cur += self._ims_per_batch
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._imdb[i] for i in db_inds]
        return self.get_minibatch(minibatch_db)
     	
    def get_imdb(self, im_list_file, kp_list_file, curr_keypoint_file):
        with open(im_list_file) as f:
            im_kp_files = [x.strip().split() for x in f.readlines()]
        with open(kp_list_file) as f:
            kps = [int(x.strip()) for x in f.readlines()]
        with open(curr_keypoint_file) as f:
            curr_kps_all = [map(float, x.strip().split()) for x in f.readlines()]
        self._kps = kps
        imdb = []
        for i, im_kp in enumerate(im_kp_files):
            imdb_entry = {}
            imdb_entry['im'] = im_kp[0]
            imdb_entry['kp'] = im_kp[1]
            imdb_entry['curr_kp'] = curr_kps_all[i]
            imdb.append(imdb_entry)
        return imdb, len(kps)    
             
    def setup(self, bottom, top):
        """Setup the ImageKPDataLayer."""
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        #self._num_keypoints = 10#layer_params['num_keypoints']
        #self._im_list_file = 'a.txt'#layer_params['im_list']
        #print "as"
        self._keypoint_list = layer_params['kp_list']
        self._im_list_file = layer_params['im_list']
        self._curr_keypoint_file = layer_params['curr_kps']
        self._phase = layer_params['phase'] 
        self._resize = int(layer_params['resize'])
        self._ims_per_batch = int(layer_params['ims_per_batch'])
        self._dataset_root = layer_params['root_dir']
        self._mean_file = layer_params['mean_file']
        self._shuffle = int(layer_params['shuffle'])
        self._name_to_top_map = {
            'data': 0,
            'error_kps': 1}
        
        self._imdb, self._num_keypoints = self.get_imdb(self._im_list_file, self._keypoint_list, self._curr_keypoint_file)
        self._shuffle_imdb_inds()
        # data blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(self._ims_per_batch, 3+self._num_keypoints, self._resize, self._resize)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[1].reshape(self._ims_per_batch, 2*self._num_keypoints)

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        # top[2].reshape(1)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
