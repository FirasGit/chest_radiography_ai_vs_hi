import os
import pandas as pd
import numpy as np
from evaluation.helper import create_folder_structure


def create_labels_csv(self, label, pred, label_names, acquisition_numbers, cfg=None):
    unique_labels = self.cfg.evaluation.unique_labels
    for sample_idx in range(len(pred)):
        # Get the labels for each sample in batch
        label_dict_pred = {'AcquisitionNumber': str(
            acquisition_numbers[sample_idx].item())}
        label_dict_gt = {'AcquisitionNumber': str(
            acquisition_numbers[sample_idx].item())}

        for unique_label in unique_labels:
            # Store prediction and groundtruth into dataframe

            if cfg.meta.rank_consistent_encoding:
                # Save the rank
                label_dict_gt[unique_label] = [
                    (label[sample_idx] > 0.5).sum().item()]
                label_dict_pred[unique_label] = [
                    (pred[sample_idx] > 0.5).sum().item()]
            else:
                df_gt = pd.DataFrame(
                    np.array([label.tolist()[sample_idx]]), columns=label_names)
                df_pred = pd.DataFrame(
                    np.array([pred.tolist()[sample_idx]]), columns=label_names)

                # Only take prediction and groundtruth that match the wildcard label and use the argmax as the predicted
                # label of the wildcard label. The +1 is needed because the argmax starts at index 0.
                label_dict_gt[unique_label] = [df_gt.loc[:,
                                                         df_gt.columns.str.contains(unique_label)].loc[0].argmax() + 1]
                label_dict_pred[unique_label] = [df_pred.loc[:,
                                                             df_pred.columns.str.contains(unique_label)].loc[0].argmax() + 1]
        df_gt_unique = pd.DataFrame(label_dict_gt)
        df_pred_unique = pd.DataFrame(label_dict_pred)

        destination_dir = os.path.join(self.cfg.evaluation.path_to_evaluation_results_dir,
                                       self.cfg.meta.prefix_name)

        create_folder_structure(destination_dir)
        save_the_dataframes(df_gt_unique, df_pred_unique, destination_dir)


def save_the_dataframes(df_gt_unique, df_pred_unique, destination_dir):
    dataframe_info = {'pred': {'csv_name': 'predictions.csv',
                               'dataframe': df_pred_unique},
                      'gt': {'csv_name': 'groundtruth.csv',
                             'dataframe': df_gt_unique}
                      }

    for key in dataframe_info.keys():
        combined_csv_path = os.path.join(
            destination_dir, dataframe_info[key]['csv_name'])
        if os.path.isfile(combined_csv_path):
            # Append rows to file if exists
            df_combined = pd.read_csv(combined_csv_path)
            df_combined_updated = pd.concat([
                df_combined, dataframe_info[key]['dataframe']])
            df_combined_updated.to_csv(combined_csv_path, index=False)
        else:
            # Create file if it doesn't exist
            dataframe_info[key]['dataframe'].to_csv(
                combined_csv_path, index=False)
