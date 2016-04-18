import sys
import os
import numpy as np

def add_errors(curr_kps_file, error_kps_file, output_file):
    with open(curr_kps_file) as f:
        kps = [x.split() for x in f.readlines()]

    with open(error_kps_file) as f:
        errors = [x.split() for x in f.readlines()]

    with open(output_file,'w') as f:
        for i in xrange(len(kps)):
            for j in xrange(0, len(kps[i]), 2):
                del_x = 224.0*float(errors[i][j])
                del_y = 224.0*float(errors[i][j+1])
                new_kp_x = float(kps[i][j]) + del_x 
                new_kp_y = float(kps[i][j+1]) + del_y
                f.write("%s %s "%(new_kp_x, new_kp_y))
            f.write("\n")

if __name__ == '__main__':
    curr_kps_file = sys.argv[1]
    error_kps_file = sys.argv[2]
    output_file = sys.argv[3]
    add_errors(curr_kps_file, error_kps_file, output_file)
