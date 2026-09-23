#!/usr/bin/env python

# Script that do some cleaning in the output folder and agregate metrics, then plots nice graphs

import sys, os
import argparse
import shutil
import fnmatch
import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def get_parser():

    parser = argparse.ArgumentParser('Aggregate metrics from csv files and plot histograms and tables to compare methods')
    parser.add_argument("-i",
                        type=str,
                        help="Folder with csv files inside to agregate",
                        required=True)
    parser.add_argument("-o", 
                        type=str,
                        help="Output folder for the generated graphs",
                        required=False)
    return parser


def main(args=None):

    # Parser :
    if not args:
        args = sys.argv[1:]
    parser = get_parser()
    arguments = parser.parse_args(args)
    folder = arguments.i
    output_folder = arguments.o
    if output_folder:
        # Create folder if does not exist
        if not os.path.isdir(folder):
            os.mkdir(folder)
    else:
        output_folder = folder
    metrics_type = ["dice", "hausdorff", "jaccard"]
    metrics = ["global", "mean", "min", "max", "std"]
    methods = ["NoRot", "pca", "hog", "pcahog"]
    # init global dic :
    # dictionary for metrics in dictionnary for method containing lists of values
    for type_metric in metrics_type:
        metrics = [f"{type_metric}_global", f"{type_metric}_mean", f"{type_metric}_min", f"{type_metric}_max", f"{type_metric}_std"]
        dice_metrics = {method: {metric: [] for metric in metrics} for method in methods}
        print("Processing metric : " + str(metrics))
        for method in methods:
            
            for root, dirnames, filenames in os.walk(folder):  # searching the given directory
                print("Processing folder : " + root)
                for filename_NoRot in fnmatch.filter(filenames, f"*{type_metric}_NoRot.csv"):
                    filename_pca = os.path.join(root, filename_NoRot.replace("NoRot", "pca"))
                    filename_hog = os.path.join(root, filename_NoRot.replace("NoRot", "hog"))
                    filename_auto = os.path.join(root, filename_NoRot.replace("NoRot", "pcahog"))
                    if not (os.path.isfile(filename_pca) and os.path.isfile(filename_hog) and os.path.isfile(filename_auto)):
                        print("4 csv files not found for file : " + filename_NoRot)
                        continue  # this block makes sure that there is csv file for the 4 methods

                    # Open and verify presence of all metrics
                    with open(os.path.join(root, filename_NoRot), 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        metric_dic_NoRot = {rows[0]: float(rows[1]) for rows in reader}
                        if len(metric_dic_NoRot) != len(metrics):
                            print("all Metrics not present in csv : " + filename_NoRot)
                            continue
                    with open(os.path.join(root, filename_pca), 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        metric_dic_pca = {rows[0]: float(rows[1]) for rows in reader}
                        if len(metric_dic_pca) != len(metrics):
                            print("all Metrics not present in csv : " + filename_pca)
                            continue
                    with open(os.path.join(root, filename_hog), 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        metric_dic_hog = {rows[0]: float(rows[1]) for rows in reader}
                        if len(metric_dic_hog) != len(metrics):
                            print("all Metrics not present in csv : " + filename_hog)
                            continue
                    with open(os.path.join(root, filename_auto), 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        metric_dic_auto = {rows[0]: float(rows[1]) for rows in reader}
                        if len(metric_dic_auto) != len(metrics):
                            print("all Metrics not present in csv : " + filename_hog)
                            continue
                    print(metric_dic_NoRot)
                    # Now append the metrics to the general dic, if program arrives at this step it means that the all .csv exist and have the all metrics inside
                    for metric in metrics:
                        dice_metrics["NoRot"][metric].append(metric_dic_NoRot[metric])
                        dice_metrics["pca"][metric].append(metric_dic_pca[metric])
                        dice_metrics["hog"][metric].append(metric_dic_hog[metric])
                        dice_metrics["pcahog"][metric].append(metric_dic_hog[metric])

            nb_subjects = len(next(iter(next(iter(dice_metrics.values())).values())))  # just to get number of subjects (we access the first element of dic twice)


        # Cleaning everything :
        # shutil.rmtree(folder)
        # os.mkdir(folder)

        # Processing data
        # matplotlib.use('Agg')  # prevent display figure
        fig = plt.figure(figsize=(20*2, 40*2))
        fig.suptitle("Histograms of dataset with " + str(nb_subjects) + " images")
        for k, metric in enumerate(metrics):
            plt.subplot(2, (len(metrics)+1)//2, k + 1)
            if metric == f"{type_metric}_std":
                range_metric = None
                xlabel = "std"
            else:
                range_metric = (0, 1)
                xlabel = f"{type_metric} score"
            plt.hist((dice_metrics["NoRot"][metric], dice_metrics["pca"][metric], dice_metrics["hog"][metric], dice_metrics["pcahog"][metric]), bins=10, range=range_metric)
            plt.ylabel("count")
            plt.xlabel(xlabel)
            plt.title(metric + " histogram")
            plt.legend(methods)

        plt.subplot(2, (len(metrics)+1)//2, len(metrics)+1)
        #Building the np array to plot the table
        data = np.zeros((6, len(methods)))
        for col, metric in enumerate(metrics):
            list_argmax = list(np.argmax((dice_metrics["NoRot"][f"{type_metric}_global"], dice_metrics["pca"][f"{type_metric}_global"], dice_metrics["hog"][f"{type_metric}_global"], dice_metrics["pcahog"][f"{type_metric}_global"]), axis=0))
            nb_NoRot_best = list_argmax.count(0)
            nb_pca_best = list_argmax.count(1)
            nb_hog_best = list_argmax.count(2)
            nb_auto_best = list_argmax.count(3)
            for row, method in enumerate(methods):
                data[col, row] = np.mean(dice_metrics[method][metric])
            data[len(metrics), :] = [nb_NoRot_best, nb_pca_best, nb_hog_best, nb_auto_best]

        # Save the data as a CSV file
        csv_output_path = os.path.join(output_folder, type_metric + "_aggregated_metrics.csv")
        with open(csv_output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["Metric"] + methods
            writer.writerow(header)
            for i, metric in enumerate(metrics):
                writer.writerow([metric] + list(data[i, :]))
            writer.writerow(["No times best"] + list(data[len(metrics), :]))

        #plt.table(cellText=data, colLabels=methods, rowLabels=metrics + ["No times best"], loc="center")

        # Saving everything :
        plt.savefig(output_folder + "/histograms.png")

if __name__ == "__main__":
    # call main function
    main()
