#!/usr/bin/env python

# Script used to process one MRI image at a time (the image and its segmentation), made to be used with wrapper or alone


from __future__ import division, absolute_import
import sys, os
import argparse
from functions_sym_rot import *
import csv
import nibabel as nib
import numpy as np
import time
import math
from scipy.ndimage.filters import gaussian_filter1d

def get_parser():
    parser = argparse.ArgumentParser(description='Script to process a MRI image with its segmentation.')
    parser.add_argument("-i",
                        type=str,
                        required=True,
                        help="File input, e.g. /home/data/cool_T2_MRI.nii.gz")
    parser.add_argument("-iseg",
                        type=str,
                        required=True,
                        help="Segmentation of the input file, e.g. /home/data/cool_T2_MRI_seg_manual.nii.gz")
    parser.add_argument("-o",
                        type=str,
                        required=False,
                        help="Output folder for test results, e.g. path/to/output/folder")
    parser.add_argument("-qc",
                        type=str,
                        required=False,
                        help="The path where the quality control generated content will be saved")

    return parser


def main(args=None):

    # Parser :
    if not args:
        args = sys.argv[1:]
    parser = get_parser()
    arguments = parser.parse_args()
    cwd = os.getcwd()
    fname_image = arguments.i
    fname_seg = arguments.iseg
    if arguments.qc:
        path_qc = arguments.qc
        # creating qc dir if it does not exist
        if not os.path.isdir(path_qc):
            os.mkdir(path_qc)
    if arguments.o:
        output_dir = arguments.o
    else:
        output_dir = os.getcwd()

    print("======> Python processing file : " + fname_image + " with seg : " + fname_seg)
    # Copy input images to output folder and change orientation to LPI with sct_image, then read them with nibabel to have data in numpy arrays, and get the affine and header for later saving results in the same space
    os.system(f"cp {fname_image} {output_dir}")
    os.system(f"cp {fname_seg} {output_dir}")
    sub_and_sequence = (fname_image.split("/")[-1]).split(".nii.gz")[0]

    #image_object = Image(fname_image).change_orientation("LPI")
    fname_image_output = output_dir + "/" + sub_and_sequence + ".nii.gz"
    os.system(f'sct_image -i {fname_image} -setorient LPI -o {fname_image_output}')
    #seg_object = Image(fname_seg).change_orientation("LPI")

    fname_seg_output = output_dir + "/" + sub_and_sequence + "_seg.nii.gz"
    os.system(f'sct_image -i {fname_seg} -setorient LPI -o {fname_seg_output}')
    # Read with nibabel to have data with nibabel:
    data_image = nib.load(fname_image_output).get_fdata()
    data_seg = nib.load(fname_seg_output).get_fdata()

    nx, ny, nz = data_seg.shape
    # get pixel dimensions:
    px, py, pz = nib.load(fname_image_output).header.get_zooms()
    print(px, py, pz)
    min_z = np.min(np.nonzero(data_seg)[2])
    max_z = np.max(np.nonzero(data_seg)[2])

    methods = ["pca", "hog", "auto"]

    angle_range = 40
    conf_score_th_pca = 1.6  # for pca and auto !
    conf_score_th_hog = 1  # only for hog
    smooth = True

    for k, method in enumerate(methods):

        angles = np.zeros(max_z - min_z)
        conf_score = np.zeros(max_z - min_z)
        axes_image = np.zeros((nx, ny, nz))
        start_time = time.time()
        centermass = np.zeros((2, max_z-min_z))

        for z in range(0, max_z-min_z):

            if method == "hog":
                angles[z], conf_score[z], centermass[:, z] = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z], px, py, method, angle_range=angle_range, return_centermass=True)
                if math.isnan(angles[z]) or conf_score[z] is None:
                    raise Exception("this is not supposed to happen, hog is only searching in the angle range, no angle should be outside range")
                if conf_score[z] < conf_score_th_hog:
                    angles[z] = 0
                    conf_score[z] = -5
            elif method == "pca":
                angles[z], conf_score[z], centermass[:, z] = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z], px, py, method, angle_range=angle_range, return_centermass=True)
                if math.isnan(conf_score[z]) or conf_score[z] is None:
                    conf_score[z] = -10
                    angles[z] = 0
                if conf_score[z] < conf_score_th_pca:
                    angles[z] = 0
                    conf_score[z] = -5
            elif method == "auto":
                angles[z], conf_score[z], centermass[:, z] = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z], px, py, "pca", angle_range=angle_range, return_centermass=True)
                if conf_score[z] < conf_score_th_pca or math.isnan(conf_score[z]) or conf_score[z] is None:
                    angles[z], conf_score[z], centermass[:, z] = find_angle(data_image[:, :, min_z + z], data_seg[:, :, min_z + z], px, py, "hog", angle_range=angle_range, return_centermass=True)
            else:
                raise Exception("no method named" + method)

        z_nonzero = range(0, max_z-min_z)

        if smooth:
            # coeffs = np.polyfit(z_nonzero, angles[z_nonzero], polydeg)
            # poly = np.poly1d(coeffs)
            # angles_smoothed = np.polyval(poly, z_nonzero)
            angles_smoothed = gaussian_filter1d(angles, 5)

        for z in range(0, max_z-min_z):
            axes_image[:, :, min_z + z] = generate_2Dimage_line(axes_image[:, :, min_z + z], centermass[0, z], centermass[1, z], angles_smoothed[z] - pi/2, value=k+1)
            # axes_image[int(centermass[0]), int(centermass[1]), min_z + z] = 100000
        # save angle and z in a text file:
        angle_filename = output_dir + "/" + sub_and_sequence + "_angles_" + method + ".txt"
        with open(angle_filename, "w") as f:
            f.write("z_slice\tangle_deg\n")
            for z in z_nonzero:
                f.write(f"{min_z + z}\t{angles_smoothed[z] * 180 / pi}\n")

        print("Time elapsed for method " + method + " (+ generating axes) : " + str(round(time.time() - start_time, 1)) + " seconds")
        print("Max angle is : " + str(max(angles) * 180/pi) + ", min is : " + str(min(angles) * 180/pi) + " and mean is : " + str(np.mean(angles) * 180/pi))

        fname_axes = output_dir + "/" + sub_and_sequence + "_axes_" + method + ".nii.gz"
        nib.save(nib.Nifti1Image(axes_image, nib.load(fname_image_output).affine, nib.load(fname_image_output).header), fname_axes)
        nib.save(nib.Nifti1Image(data_image, nib.load(fname_image_output).affine, nib.load(fname_image_output).header), fname_image_output)
        nib.save(nib.Nifti1Image(data_seg, nib.load(fname_seg_output).affine, nib.load(fname_seg_output).header), fname_seg_output)

        # if method is "pca":
        #     cmap = 'PRGn'
        # elif method is "hog":
        #     cmap = 'Wistia'
        # else:
        #     cmap = 'winter'
        # plt.figure(figsize=(6.4 * 2, 4.8 * 2))
        # plt.scatter(z_nonzero, angles * 180 / pi, c=conf_score, cmap=cmap)
        # if smooth:
        #     plt.plot(z_nonzero, angles_smoothed * 180/pi, "r-")
        # plt.ylabel("angle in deg")
        # plt.xlabel("z slice")
        # plt.colorbar().ax.set_ylabel("conf score " + method)
        # plt.savefig(output_dir + "/" + fname_image.split("/")[-1] + "_" + sub_and_sequence + method + "_angle_conf_score_z.png")  # reliable file name ?

        angles_qc = np.zeros(data_image.shape[2])
        angles_qc[:] = np.nan
        angles_qc[min_z:max_z] = -angles_smoothed

        #generate_qc(fname_in1=fname_image_output, fname_seg=fname_seg_output, angle_line=angles_qc[::-1], args=[method], path_qc=path_qc, dataset=None, subject=None, process="rotation")
        #os.system(f'sct_qc  -i fname_image_output {fname_image_output} -qc {path_qc}')
    print("fsleyes " + fname_image_output + " " + fname_seg_output + " -cm red" + " " + output_dir + "/" + sub_and_sequence + "_axes_pca.nii.gz -cm blue " + output_dir + "/" + sub_and_sequence + "_axes_hog.nii.gz -cm green " + output_dir + "/" + sub_and_sequence + "_axes_auto.nii.gz -cm yellow")
    # fsleyes /home/nicolas/unf_test/unf_spineGeneric/sub-01/anat/sub-01_T1w.nii.gz /home/nicolas/test_single_rot/sub-01_T1w_axes_pca.nii.gz -cm blue /home/nicolas/test_single_rot/sub-01_T1w_axes_hog.nii.gz -cm green

def memory_limit():
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 / 2, hard))

def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory

if __name__ == '__main__':

    # if sys.gettrace() is None:
        #sct.init_sct()
        # call main function
    main()
    # else:
    #     memory_limit()  # Limitates maximun memory usage to half
    #     try:
    #         sct.init_sct()
    #         call main function
            # main()
        # except MemoryError:
        #     sys.stderr.write('\n\nERROR: Memory Exception\n')
        #     sys.exit(1)
