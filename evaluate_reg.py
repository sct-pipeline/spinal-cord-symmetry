#!/usr/bin/env python

# Script used to process one MRI image at a time (the image and its segmentation), made to be used with wrapper or alone

from __future__ import division, absolute_import
import sys, os
import argparse
from functions_sym_rot import *
import csv
import nibabel as nib


def get_parser():
    parser = argparse.ArgumentParser(
        description='Script to process a MRI image with its segmentation.'
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="File input, e.g. /home/data/cool_T2_MRI.nii.gz"
    )
    parser.add_argument(
        "-iseg",
        type=str,
        required=True,
        help="Segmentation of the input file, e.g. /home/data/cool_T2_MRI_seg_manual.nii.gz"
    )
    parser.add_argument(
        "-sct-dir",
        type=str,
        required=True,
        help="Path to SCT directory, e.g. /home/data/spinalcordtoolbox"
    )
    parser.add_argument(
        "-o",
        type=str,
        required=False,
        help="Output folder for test results, e.g. path/to/output/folder"
    )
    parser.add_argument(
        "-discs",
        type=str,
        required=False,
        help="Output folder for test results, e.g. path/to/output/folder"
    )
    parser.add_argument(
        "-qc",
        type=str,
        required=False,
        help="The path where the quality control generated content will be saved"
    )
    return parser

def main(args=None):
    # Parser :
    if not args:
        args = sys.argv[1:]
    parser = get_parser()
    arguments = parser.parse_args()
    fname_image = arguments.i
    fname_seg = arguments.iseg
    sct_path = arguments.sct_dir
    fname_discs = arguments.discs
    if arguments.discs:
        fname_discs = arguments.discs
    else:
        fname_discs = None
    if arguments.qc:
        path_qc = arguments.qc
        # creating qc dir if it does not exist
        if not os.path.isdir(path_qc):
            os.mkdir(path_qc)
    if arguments.o:
        output_dir = arguments.o
        # creating output dir if it does not exist
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

    methods = ["NoRot", "pca", "hog", "pcahog"]

    fname_seg_template = os.path.join(sct_path, 'data/PAM50/template/PAM50_cord.nii.gz')

    print("Python processing file : " + fname_image + " with seg : " + fname_seg)

    # Determining contrast :
    if ("T1w" in fname_image) or ("t1w" in fname_image) or ("MToff" in fname_image):
        contrast, contrast_label = "t1", "t1"
        fname_image_template = os.path.join(sct_path, "data/PAM50/template/PAM50_t1.nii.gz")
    elif ("T2w" in fname_image) or ("t2w" in fname_image)or ("MTon" in fname_image):
        contrast, contrast_label = "t2", "t2"
        fname_image_template = os.path.join(sct_path, "data/PAM50/template/PAM50_t2.nii.gz")
    elif ("T2s" in fname_image) or ("t2s" in fname_image):
        contrast, contrast_label = "t2s", "t2"
        fname_image_template = os.path.join(sct_path, "data/PAM50/template/PAM50_t2s.nii.gz")
    else:
        print("Contrast not supported yet for file : " + fname_image)
        return

    # Labelling vertebrae :
    if fname_discs:
        # Copying discs file in output dir
        os.system(f'cp {fname_discs} {output_dir}')
    else:
        # Label discs:
        os.system(
            f'sct_label_vertebrae -i {fname_image} -s {fname_seg} -c {contrast_label} -ofolder {output_dir} -v 0'
        )
        fname_discs = output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_labeled_discs.nii.gz" ## TODO: validate this name

    # Applying same process but for different methods :

    for method in methods:

        print("\n\n Registration with " + method)

        # Registration
        if method == "NoRot":
            os.system(
                f'sct_register_to_template -i {fname_image} -s {fname_seg} -c {contrast} -ldisc {fname_discs} -ofolder {output_dir} -param "step=1,type=seg,algo=centermass,poly=0,slicewise=1" -v 0 -qc {path_qc}'
            )
        else:
            if method == "hog" or method == "pcahog":
                type_im="imseg"
            else:
                type_im="seg"
            os.system(
                f'sct_register_to_template -i {fname_image} -s {fname_seg} -c {contrast} -ldisc {fname_discs} -ofolder {output_dir} -param "step=1,type={type_im},algo=centermassrot,poly=0,slicewise=1,rot_method={method}" -v 0 -qc {path_qc}'
            )

        # Applying warping field to segmentation
        os.system(
            f'sct_apply_transfo -i {fname_seg} -d {fname_seg_template} -w {output_dir}/warp_anat2template.nii.gz -o {output_dir}/{(fname_seg.split("/")[-1]).split(".nii.gz")[0]}_reg.nii.gz -v 0'
        )
        os.system(
            f'sct_maths -i {output_dir}/{(fname_seg.split("/")[-1]).split(".nii.gz")[0]}_reg.nii.gz -bin 0.5 -o {output_dir}/{(fname_seg.split("/")[-1]).split(".nii.gz")[0]}_reg_tresh.nii.gz -v 0'
        )

        # Opening registered segmentation
        fname_seg_reg = output_dir + "/" + (fname_seg.split("/")[-1]).split(".nii.gz")[0] + "_reg_tresh.nii.gz"
        data_seg_reg = nib.load(fname_seg_reg).get_fdata()
        data_seg_template = nib.load(fname_seg_template).get_fdata()
        min_z = np.min(np.nonzero(data_seg_reg)[2])
        max_z = np.max(np.nonzero(data_seg_reg)[2])

        # Computing Dice metrics
        dice_slice = []
        dice_glob = compute_similarity_metric(data_seg_reg[:, :, min_z:max_z], data_seg_template[:, :, min_z:max_z], metric="Dice")

        for z in range(min_z, max_z):
            dice_slice.append(compute_similarity_metric(data_seg_reg[:, :, z], data_seg_template[:, :, z], metric="Dice"))

        hausdorff_slice = []
        #hausdorff_glob = compute_similarity_metric(data_seg_reg[:, :, min_z:max_z], data_seg_template[:, :, min_z:max_z], metric="Hausdorff")
        hausdorff_glob = 0
        for z in range(min_z, max_z):
            hausdorff_slice.append(compute_similarity_metric(data_seg_reg[:, :, z], data_seg_template[:, :, z], metric="Hausdorff"))
        
        jacquard_distance_slice = []
        jacquard_distance_glob = compute_similarity_metric(data_seg_reg[:, :, min_z:max_z], data_seg_template[:, :, min_z:max_z], metric="Jaccard")

        for z in range(min_z, max_z):
            jacquard_distance_slice.append(compute_similarity_metric(data_seg_reg[:, :, z], data_seg_template[:, :, z], metric="Jaccard"))

        # Writing out metrics in csv files
        cwd = os.getcwd()
        os.chdir(output_dir)
        with open((fname_image.split("/")[-1]).split(".nii")[0] + "_dice_" + method + ".csv", 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["dice_global", dice_glob])
            filewriter.writerow(["dice_mean", np.mean(dice_slice)])
            filewriter.writerow(["dice_min", min(dice_slice)])
            filewriter.writerow(["dice_max", max(dice_slice)])
            filewriter.writerow(["dice_std", np.std(dice_slice)])
        
        with open((fname_image.split("/")[-1]).split(".nii")[0] + "_hausdorff_" + method + ".csv", 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["hausdorff_global", hausdorff_glob])
            filewriter.writerow(["hausdorff_mean", np.mean(hausdorff_slice)])
            filewriter.writerow(["hausdorff_min", min(hausdorff_slice)])
            filewriter.writerow(["hausdorff_max", max(hausdorff_slice)])
            filewriter.writerow(["hausdorff_std", np.std(hausdorff_slice)])
        
        with open((fname_image.split("/")[-1]).split(".nii")[0] + "_jaccard_" + method + ".csv", 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["jaccard_global", jacquard_distance_glob])
            filewriter.writerow(["jaccard_mean", np.mean(jacquard_distance_slice)])
            filewriter.writerow(["jaccard_min", min(jacquard_distance_slice)])
            filewriter.writerow(["jaccard_max", max(jacquard_distance_slice)])
            filewriter.writerow(["jaccard_std", np.std(jacquard_distance_slice)])
        
        os.chdir(cwd)

        # generate_qc(fname_in1=fname_image, fname_in2=output_dir + "/template2anat.nii.gz", fname_seg=fname_seg, args=args,
        #             path_qc=path_qc, dataset=None, subject=None,
        #             process='sct_register_to_template')


if __name__ == "__main__":
    
    # call main function
    main()
