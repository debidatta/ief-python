import sys
import os
import numpy as np

root_dir = sys.argv[1]
train_labels_file = sys.argv[2]
kp_list_file = sys.argv[3]
output_kp_list = sys.argv[4]

with open(kp_list_file) as f:
    kps = [int(x.strip()) for x in f.readlines()]

with open(train_labels_file) as f:
    label_files = [x.strip() for x in f.readlines()]

kps_array = np.zeros((len(label_files), len(kps)*2))
for n, label_file in enumerate(label_files):
    with open(os.path.join(root_dir, label_file)) as f:
        kps_coords = [x.strip().split()[1:] for i, x in enumerate(f.readlines()) if i in kps]
        kps_coords = [float(x) for kp_coords in kps_coords for x in kp_coords]
        kps_coords = [x/960*224 if i%2 == 0 else x/540*224 for i,x in enumerate(kps_coords)]
        kps_array[n, :] = kps_coords
            
median_coords = np.median(kps_array, axis=0)

with open(output_kp_list,'w') as f:
    for i in xrange(len(label_files)):
        for j in xrange(len(kps)*2):
            f.write("%s "%median_coords[j])
        f.write("\n")
