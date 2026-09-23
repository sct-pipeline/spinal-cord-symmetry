#!/bin/bash
#
# Process data of different neck positions (extension, flexion and straight).
#
# Usage:
#   ./process_data.sh <SUBJECT>
# 
# Manual segmentations and labels (discs, PMJ, nerve rootlets) should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/anat/
#
# Authors: Sandrine Bédard

set -x
# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1
PATH_SCRIPTS=$PWD
# Save script path
PATH_DERIVATIVES="${PATH_DATA}/derivatives/labels"

# get starting time:
start=`date +%s`

# FUNCTIONS
# ==============================================================================

# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  # Update global variable with segmentation file name
  # Remove space other in filename 
  #suffix="_space-other"
  FILELABEL="${file//$suffix/}_label-disc"
  FILELABELMANUAL="${PATH_DERIVATIVES}/${SUBJECT}/anat/${FILELABEL}.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_image -i ${file}.nii.gz -set-sform-to-qform
    sct_image -i ${file_seg}.nii.gz -set-sform-to-qform
    sct_image -i ${FILELABEL}.nii.gz -set-sform-to-qform
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Check if manual segmentation already exists. If it does, copy it locally. If it does not, perform segmentation.
segment_if_does_not_exist(){
  local file="$1"
  folder_contrast="anat"

  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels_softseg_bin/${SUBJECT}/${folder_contrast}/${file}_seg.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg spinalcord -i ${file}.nii.gz -qc ${PATH_QC} -largest 1 -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
  fi
}

# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# split subject is and session:
sub_id=$(echo $SUBJECT | cut -d'/' -f1)
ses_id=$(echo $SUBJECT | cut -d'/' -f2)

# Copy source images
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${sub_id}_*T2w.* .
# Go to anat folder where all structural data are located
cd ${SUBJECT}/anat/
file_t2="${SUBJECT//\//_}_T2w"

# 1. Segment spinal cord
segment_if_does_not_exist $file_t2

# 2. Create C2-C3 disc label in the cord
label_if_does_not_exist $file_t2 ${file_t2}_seg
# Create mid-vertebrae labels
#sct_label_utils -i ${file_t2}_seg_labeled.nii.gz -vert-body 2,7 -o ${file_t2}_seg_labeled_vertbody_27.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}

python ${PATH_SCRIPTS}/evaluate_reg.py -i ${file_t2}.nii.gz -iseg ${file_t2}_seg.nii.gz -sct-dir $SCT_DIR -o ${PATH_DATA_PROCESSED}/${SUBJECT}/anat -qc ${PATH_QC} -discs ${file_t2}_seg_labeled_discs.nii.gz

sct_process_segmentation -i ${file_t2}_seg.nii.gz -anat ${file_t2}.nii.gz -perslice 1 -normalize-PAM50 1 -discfile ${file_t2}_seg_labeled_discs.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT} -o $PATH_RESULTS/${file_t2}_PAM50.csv
sct_process_segmentation -i ${file_t2}_seg.nii.gz -anat ${file_t2}.nii.gz -perslice 1 -normalize-PAM50 0 -discfile ${file_t2}_seg_labeled_discs.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT} -v 2 -o $PATH_RESULTS/${file_t2}.csv

end=`date +%s`
runtime_final=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime_final / 3600))hrs $((($runtime_final / 60) % 60))min $(($runtime_final % 60))sec"
echo "~~~"
