#!/usr/bin/env python
# -*- coding: utf-8

# Analyses CSA morphometry of using rootlets vs disc-based registration
# Example command: python analyse_ascor.py -i <input_directory> -o <output_directory> -exclude <exclude_file>   
# Author: Sandrine Bédard

import os
import logging
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import scipy.stats as stats
import yaml
from scipy.signal import find_peaks

METRICS = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)', 'MEAN(eccentricity)',
           'MEAN(solidity)', 'MEAN(symmetry_dice_RL)', 'MEAN(symmetry_hausdorff_RL)', 'MEAN(symmetry_difference_RL)',
           'MEAN(symmetry_dice_AP)', 'MEAN(symmetry_hausdorff_AP)', 'MEAN(symmetry_difference_AP)']

METRICS_DTYPE = {
    'MEAN(diameter_AP)': 'float64',
    'MEAN(area)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(eccentricity)': 'float64',
    'MEAN(solidity)': 'float64',
    'aSCOR':'float64',
    'MEAN(symmetry_dice_RL)': 'float64',
    'MEAN(symmetry_hausdorff_RL)': 'float64',
    'MEAN(symmetry_difference_RL)': 'float64',
    'MEAN(symmetry_dice_AP)': 'float64', 
    'MEAN(symmetry_hausdorff_AP)': 'float64', 
    'MEAN(symmetry_difference_AP)': 'float64'
}

METRIC_TO_TITLE = {
    'MEAN(diameter_AP)': 'AP Diameter',
    'MEAN(area)': 'Cross-Sectional Area',
    'MEAN(diameter_RL)': 'Transverse Diameter',
    'MEAN(eccentricity)': 'Eccentricity',
    'MEAN(solidity)': 'Solidity',
    'MEAN(compression_ratio)': 'Compression Ratio',
    'aSCOR':'aSCOR',
    'MEAN(symmetry_dice_RL)': 'Symmetry Dice RL',
    'MEAN(symmetry_hausdorff_RL)': 'Symmetry Hausdorff RL',
    'MEAN(symmetry_difference_RL)': 'Symmetry Difference RL',
    'MEAN(symmetry_dice_AP)': 'Symmetry Dice AP',
    'MEAN(symmetry_hausdorff_AP)': 'Symmetry Hausdorff AP',
    'MEAN(symmetry_difference_AP)': 'Symmetry Difference AP'
}

METRIC_TO_AXIS = {
    'MEAN(diameter_AP)': 'AP Diameter [mm]',
    'MEAN(area)': 'Cross-Sectional Area [mm²]',
    'MEAN(diameter_RL)': 'Transverse Diameter [mm]',
    'MEAN(eccentricity)': 'Eccentricity [a.u.]',
    'MEAN(solidity)': 'Solidity [%]',
    'MEAN(compression_ratio)': 'Compression Ratio [a.u.]',
    'aSCOR':'aSCOR [%]',
    'MEAN(symmetry_dice_RL)': 'Symmetry Dice RL [a.u.]',
    'MEAN(symmetry_hausdorff_RL)': 'Symmetry Hausdorff RL [mm]',
    'MEAN(symmetry_difference_RL)': 'Symmetry Difference RL [mm²]',
    'MEAN(symmetry_dice_AP)': 'Symmetry Dice AP [a.u.]',
    'MEAN(symmetry_hausdorff_AP)': 'Symmetry Hausdorff AP [mm]',
    'MEAN(symmetry_difference_AP)': 'Symmetry Difference AP [mm²]'
}

DEMOGRAPHIC_TO_AXIS = {
    'age': 'Age [years]',
    'BMI': 'BMI [kg/m²]',
    'height': 'Height [cm]',
    'weight': 'Weight [kg]',
}

# ylim max offset (used for showing text)
METRICS_TO_YLIM_OFFSET = {
    'MEAN(diameter_AP)': 0.4,
    'MEAN(area)': 6,
    'MEAN(diameter_RL)': 0.7,
    'MEAN(eccentricity)': 0.03,
    'MEAN(solidity)': 1,
    'MEAN(compression_ratio)': 0.03,
    'aSCOR': 1,
    'MEAN(symmetry_dice_RL)': 0.02,
    'MEAN(symmetry_hausdorff_RL)': 0.02,
    'MEAN(symmetry_difference_RL)': 0.02,
    'MEAN(symmetry_dice_AP)': 0.02,
    'MEAN(symmetry_hausdorff_AP)': 0.02,
    'MEAN(symmetry_difference_AP)': 0.02
    
}

# Set ylim to do not overlap horizontal grid with vertebrae labels
METRICS_TO_YLIM = {
    'MEAN(diameter_AP)': (5.7, 9.3), #(10, 20), TODO: use second value for canal
    'MEAN(area)': (35, 95),  #(135, 300)
    'MEAN(diameter_RL)':(8.5, 14.5), # (15, 35) 
    'MEAN(eccentricity)': (0.51, 0.89),
    'MEAN(solidity)': (91.2, 99.9),
    'MEAN(compression_ratio)': (0.41, 0.84),
    'aSCOR': (20, 50),
    'MEAN(symmetry_dice_RL)': (0.7, 1),
    'MEAN(symmetry_hausdorff_RL)': (0, 2),
    'MEAN(symmetry_difference_RL)': (5, 15),
    'MEAN(symmetry_dice_AP)': (0.7, 1),
    'MEAN(symmetry_hausdorff_AP)': (0, 2),
    'MEAN(symmetry_difference_AP)':  (5, 15),
}


LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12


FNAME_LOG = 'log_stats.txt'

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        required=True,
                        type=str,
                        help="CSV file of single subject with PAM50-based metrics")
    parser.add_argument("-o-folder",
                        type=str,
                        required=True,
                        help="Folder to write results")
    parser.add_argument('-ref-subject', required=False, type=str, default='sub-295461',
                        help="Name of a reference subject to use to get discs levels")
    parser.add_argument("-i-HC",
                        required=False,
                        type=str,
                        help="Path to folder with healthy control")
    parser.add_argument("-exclude",
                        type=str,
                        required=False,
                        help="exclude list")
    return parser


def read_metrics_pam50(file, group, exclude_list=None):
    df = pd.read_csv(file)
    df['participant_id'] = (df['Filename'].str.split('/').str[-3]).astype(str)#.str.replace('_acq-axial_T2w_label-SC_mask.nii.gz', '')
    df['group'] = group
    df = df[['participant_id', 'group', 'VertLevel', 'Slice (I->S)']+ METRICS].drop(0)
    logger.info('Number of subjects BEFORE exclusion:')
    logger.info(len(np.unique(df['participant_id'])))
    if exclude_list:
        for subject in exclude_list:
            logger.info(f'dropping {subject}')
            df = df.drop(df[df['participant_id'] == subject].index, axis=0)
    logger.info('Number of subjects AFTER exclusion:')
    logger.info(len(np.unique(df['participant_id'])))
    return df


def get_vert_indices(df, vertlevel='VertLevel'):
    """
    Get indices of slices corresponding to mid-vertebrae
    Args:
        df (pd.dataFrame): dataframe with CSA values
    Returns:
        vert (pd.Series): vertebrae levels across slices
        ind_vert (np.array): indices of slices corresponding to the beginning of each level (=intervertebral disc)
        ind_vert_mid (np.array): indices of slices corresponding to mid-levels
    """
    # Get vert levels for one certain subject
    vert = df[(df['participant_id'] == ref) & (df['group'] == 'dcm')][vertlevel] # 'sub-amu01' TODO: add argument for example subject
    # Get indexes of where array changes value
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    # Get the beginning of C1
    ind_vert = np.append(ind_vert, vert.index.values[-1])
    ind_vert_mid = []
    # Get indexes of mid-vertebrae
    for i in range(len(ind_vert)-1):
        ind_vert_mid.append(int(ind_vert[i:i+2].mean()))

    return vert, ind_vert, ind_vert_mid


def plot_ind_sub(df, group, metric, path_out, filename, hue='participant_id', hc=None):
    fig_size = (7, 6)
    font_size = LABELS_FONT_SIZE

    # Plot individual subject lines
    plt.figure()
    fig, ax = plt.subplots(figsize=fig_size)
    sns.lineplot(ax=ax, data=df.loc[df['group'] == group], x="Slice (I->S)", y=metric, hue=hue, linewidth=2, zorder=1, alpha=0.8)
    if hc is not None:
        sns.lineplot(ax=ax, data=hc, x="Slice (I->S)", y=metric, color='orange', linewidth=2, zorder=3, label='HC mean', errorbar='sd')
        #sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue=hue, linewidth=2,  color='red')
    #ax.set_ylim(METRICS_TO_YLIM[metric][0], METRICS_TO_YLIM[metric][1])
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin + 15, xmax - 15)
    ymin, ymax = ax.get_ylim()
    ax.get_legend().remove()

    # Get indices of slices corresponding vertebral levels
    vert, ind_vert, ind_vert_mid = get_vert_indices(df, vertlevel='VertLevel')
    for idx, x in enumerate(ind_vert[1:-1]):
        ax.axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)
    for idx, x in enumerate(ind_vert_mid, 0):
        if vert[x] > 7:
            level = 'T' + str(vert[x] - 7)
        else:
            level = 'C' + str(int(vert[x]))
        ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                verticalalignment='bottom', color='black', fontsize=font_size)

    ax.invert_xaxis()
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)
    ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
    ax.set_xlabel('Axial Slice #', fontsize=font_size)

    path_filename = os.path.join(path_out, filename)
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    logger.info('Figure saved: ' + path_filename)


def create_lineplot(df, metric=METRICS, hue=None, filename=None):
    """
    Create lineplot for individual metrics per vertebral levels.
    Note: we are ploting slices not levels to avoid averaging across levels.
    Args:
        df (pd.dataFrame): dataframe with metric values
        hue (str): column name of the dataframe to use for grouping; if None, no grouping is applied
    """

    #mpl.rcParams['font.family'] = 'Arial'

    fig, axes = plt.subplots(3, 4, figsize=(5, 5))
    axs = [axes]#.ravel()

    # Loop across metrics
    for index, metric in enumerate(metric):
        # Note: we are ploting slices not levels to avoid averaging across levels
        if hue == 'sex' or hue == 'group':
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue=hue, linewidth=2)
            if index == 0:
                axs[index].legend(loc='upper right', fontsize=TICKS_FONT_SIZE)
            else:
                axs[index].get_legend().remove()
        else:
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue=hue, linewidth=2)

        axs[index].set_ylim(METRICS_TO_YLIM[metric][0], METRICS_TO_YLIM[metric][1])
        ymin, ymax = axs[index].get_ylim()

        # Add labels
        axs[index].set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        axs[index].set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
        # Increase xticks and yticks font size
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        # Remove spines
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        # Insert a vertical line for each intervertebral disc
        for idx, x in enumerate(ind_vert[:-1]):
            axs[index].axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)
        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Deal with T1 label (C8 -> T1)
            if vert[x] > 7:
                level = 'T' + str(int(vert[x]) - 7)
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            else:
                level = 'C' + str(int(vert[x]))
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        # Invert x-axis
        axs[index].invert_xaxis()
        # Add only horizontal grid lines
        axs[index].yaxis.grid(True)
        # Move grid to background (i.e. behind other elements)
        axs[index].set_axisbelow(True)

    # Save figure
    if filename is None:
        if hue:
            filename = 'lineplot_per' + hue + '.png'
        else:
            filename = 'lineplot.png'
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    logger.info('Figure saved: ' + filename)


def read_csv_files_HC(path_HC, participant_file=None):
    # Initialize pandas dataframe where data across all subjects will be stored
    print(f'Reading {path_HC}')
    df = pd.DataFrame()
    # Loop through .csv files of healthy controls
    for file in os.listdir(path_HC):
        if 'PAM50.csv' in file:
            # Read csv file as pandas dataframe for given subject
            df_subject = pd.read_csv(os.path.join(path_HC, file))

            # Concatenate DataFrame objects
            df = pd.concat([df, df_subject], axis=0, ignore_index=True)
    # Get sub-id (e.g., sub-amu01) from Filename column and insert it as a new column called participant_id
    # Subject ID is the first characters of the filename till slash
    df.insert(0, 'participant_id', df['Filename'].str.split('/').str[-3])
    # Get number of unique subjects (unique strings under Filename column)
    subjects = df['Filename'].unique()
    # If a participants.tsv file is provided, insert columns sex, age and manufacturer from df_participants into df
    df_participants = pd.DataFrame()
    if participant_file:
        df_participants = pd.read_csv(participant_file, sep='\t')
        df = df.merge(df_participants[["age", "sex", "height", "weight", "manufacturer", "participant_id"]],
                    on='participant_id')
    # Print number of subjects
    print(f'Number of subjects: {str(len(subjects))}\n')
    return df, df_participants, subjects


def main():

    args = get_parser().parse_args()
    # Get input argments
    input_file = os.path.abspath(args.i)
    global ref
    ref = args.ref_subject
    output_folder = args.o_folder
    if args.i_HC is not None:
        folder_hc = os.path.abspath(args.i_HC)
    else:
        folder_hc = None
    # Create output folder if does not exist.
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    os.chdir(output_folder)
    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(FNAME_LOG))
    logging.root.addHandler(fh)

    # Create a list with subjects to exclude if input .yml config file is passed
    if args.exclude is not None:
        # Check if input yml file exists
        if os.path.isfile(args.exclude):
            fname_yml = args.exclude
        else:
            sys.exit("ERROR: Input yml file {} does not exist or path is wrong.".format(args.exclude))
        with open(fname_yml, 'r') as stream:
            try:
                exclude = list(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                logger.error(exc)
    else:
        exclude = []
    logger.info(exclude)

    # Read aSCOR data
    df_dcm = read_metrics_pam50(input_file, group='dcm', exclude_list=exclude)

    # Only keep between C2 and T2
    df_dcm = df_dcm[(df_dcm['VertLevel'] >=2) & (df_dcm['VertLevel'] <= 9)]

    # compute mean aSCOR at compression sites
    #create_lineplot(df_dcm, metric=METRICS, hue=None,
    #                    filename=os.path.join(output_folder, 'lineplot_ascor_dcm.png'))
    # for metric in METRICS:
    #     plot_ind_sub(df_dcm, group='dcm', metric=metric,
    #                  path_out=output_folder, filename=f'ascor_dcm_individual_{metric}.png')

    # If HC folder is provided, read HC data and create plots comparing DCM vs HC
    if folder_hc is not None:
        # Read HC data
        df_hc, df_participants, subjects_hc = read_csv_files_HC(folder_hc, participant_file=None)
        # Keep only relevant columns
        df_hc = df_hc[['participant_id', 'VertLevel', 'Slice (I->S)'] + METRICS]
        df_hc['group'] = 'hc'
        # Only keep levels between C1 and T2
        df_hc = df_hc[(df_hc['VertLevel'] >=2) & (df_hc['VertLevel'] <= 9)]
        # Merge DCM and HC dataframes
        df = pd.concat([df_dcm, df_hc], axis=0, ignore_index=True)
        # Create lineplots

        #create_lineplot(df, metric=METRICS, hue='group',
        #                    filename=os.path.join(output_folder, 'lineplot_ascor_dcm_vs_hc.png'))
        for metric in METRICS:
            plot_ind_sub(df, group='dcm', metric=metric,
                         path_out=output_folder, filename=f'ascor_dcm_individual_{metric}_vs_hc.png', hc=df_hc)
        
        # For all individual subject in DCM, plit individual ascor plot vs all the HC (put HC as mean and sd)
        #for subject in np.unique(df_dcm['participant_id']):
        #    ref = subject
        #    df_tmp = df_dcm[df_dcm['participant_id'] == subject]
        #    plot_ind_sub(df_tmp, group='dcm', metric='aSCOR',                 
        #                        path_out=output_folder, filename=f'ascor_{subject}_vs_hc.png', hue='participant_id', hc=df_hc)
if __name__ == "__main__":
    main()
