import numpy as np
import torch
from evaluation.radiologists_study_list import radiologists_acquisition_numbers
import cv2
import os
import pandas as pd
from evaluation.helper import create_folder_structure


def create_cam_images(self, data, label, weights, label_names, acquisition_numbers, cfg=None):
    with torch.set_grad_enabled(True):
        pred = self(data, create_cam=True)

        # handle the batch
        for sample_idx, sample_pred in enumerate(pred):
            acquisition_number = str(acquisition_numbers.tolist()[sample_idx])
            destination_dir = get_destination_directory(self.cfg.evaluation.path_to_evaluation_results_dir,
                                                        self.cfg.meta.prefix_name,
                                                        acquisition_number)
            # Create CAM only of images that the other radiologists have annotated
            if acquisition_number in radiologists_acquisition_numbers:
                create_folder_structure(destination_dir)
                for label_num in range(len(sample_pred)):
                    heatmap = generate_heatmap(
                        self, sample_pred, label_num, data, sample_idx)

                    heatmap_resized = get_resized_and_colored_heatmap(
                        heatmap, data)

                    org_image_normalized = min_max_normalization(
                        data[sample_idx][0].cpu())
                    save_superimposed_gradcam(org_image_normalized, heatmap_resized,
                                              destination_dir, label_names, label_num)

                cv2.imwrite(
                    os.path.join(destination_dir, 'original.jpg'), org_image_normalized.numpy())
                df = pd.DataFrame(np.array([self._get_prediction_prob(pred).tolist()[
                                  sample_idx], label.tolist()[sample_idx]]), columns=[label_names])
                df.to_csv(os.path.join(destination_dir, 'data.csv'))


def save_superimposed_gradcam(org_image_normalized, heatmap_resized, destination_dir, label_names, label_num):
    superimposed_img = np.repeat(
        org_image_normalized[:, :, np.newaxis], 3, axis=2) + 0.4 * heatmap_resized
    cv2.imwrite(
        os.path.join(destination_dir, label_names[label_num] + '.jpg'), superimposed_img.numpy())


def get_resized_and_colored_heatmap(heatmap, data):
    heatmap_resized = cv2.resize(
        heatmap.numpy(), (data.shape[-2], data.shape[-1]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_resized = cv2.applyColorMap(
        heatmap_resized, cv2.COLORMAP_JET)
    return heatmap_resized


def generate_heatmap(self, sample_pred, label_num, data, sample_idx):
    sample_pred[label_num].backward(retain_graph=True)
    gradients = self.model.get_gradient()[sample_idx]
    pooled_gradients = torch.mean(gradients, dim=[1, 2])
    activations = self.model.get_activation(data).detach()[sample_idx]
    for j in range(len(activations)):
        activations[j, :, :] *= pooled_gradients[j]
    heatmap = torch.sum(activations, dim=0).squeeze()
    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)
    return heatmap


def min_max_normalization(data, min_val=0, max_val=255):
    min_max_normalized_data = min_val + \
        ((data - data.min()) * (max_val-min_val)) / (data.max() - data.min())
    return min_max_normalized_data


def get_destination_directory(root_path, prefix_name, acquisition_number):
    destination_dir = os.path.join(
        root_path, prefix_name, str(acquisition_number), 'grad_cam')
    return destination_dir
