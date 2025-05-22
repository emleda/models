import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib
import os
import shiny_data
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import cv2
from PIL import Image, ImageFilter
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_augmented_xgboost():
    # Load test sets
    _, _, X_test1, X_test2, X_test3, _, _, y_test1, y_test2, y_test3 = shiny_data.load_split_images()

    # PCA
    pca = joblib.load('Base_pca.joblib')

    # Load model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('xgboost.json')

    # Specify augmentations
    blur_sizes = [0,1,3,5,7,9,19]
    noise_levels = [0,1,3,5,10,20,30,50]

    # Init csv
    csv_file = 'xgboost_augmented_metrics.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'test_set', 'blur_size', 'noise_level', 'accuracy', 'f1',
            # per class precision, recall, f1 (for each of 4 classes)
            'precision_immune', 'precision_other', 'precision_stromal', 'precision_tumour',
            'recall_immune', 'recall_other', 'recall_stromal', 'recall_tumour',
            'f1_immune', 'f1_other', 'f1_stromal', 'f1_tumour',
            # overall average confidence (max prob per sample)
            'confidence_overall',
            # per class average confidence (for samples predicted as that class)
            'confidence_immune_avg', 'confidence_other_avg', 'confidence_stromal_avg', 'confidence_tumour_avg',
            # per class std dev of confidence
            'confidence_immune_std', 'confidence_other_std', 'confidence_stromal_std', 'confidence_tumour_std',
            # counts of predictions per class (how many samples predicted as each class)
            'count_pred_immune', 'count_pred_other', 'count_pred_stromal', 'count_pred_tumour'
        ])

    for test_num, (X_test, y_test_enc) in enumerate([(X_test1, y_test1), (X_test2, y_test2), (X_test3, y_test3)], 1):
        for noise_level in noise_levels:
            for blur_size in blur_sizes:
                # Apply augmentations + flatten + pca
                X_test_aug = shiny_data.apply_noise(shiny_data.apply_blur(X_test, blur_size), std=noise_level)
                X_test_aug_flat = X_test_aug.reshape(X_test_aug.shape[0], -1)
                X_test_aug_pca = pca.transform(X_test_aug_flat)

                # Predict
                y_pred = xgb_model.predict(X_test_aug_pca)
                y_pred_probs = xgb_model.predict_proba(X_test_aug_pca)

                # Metrics
                accuracy = accuracy_score(y_test_enc, y_pred)
                f1_weighted = f1_score(y_test_enc, y_pred, average='weighted')
                precision_per_class = precision_score(y_test_enc, y_pred, average=None, zero_division=0)
                recall_per_class = recall_score(y_test_enc, y_pred, average=None, zero_division=0)
                f1_per_class = f1_score(y_test_enc, y_pred, average=None, zero_division=0)
                confidence_overall = np.mean(np.max(y_pred_probs, axis=1))

                counts_per_class = np.array([np.sum(y_pred == i) for i in range(y_pred_probs.shape[1])])

                conf_avg_per_class = []
                conf_std_per_class = []
                for class_idx in range(y_pred_probs.shape[1]):
                    mask = (y_pred == class_idx)
                    if np.sum(mask) > 0:
                        confs = y_pred_probs[mask, class_idx]
                        conf_avg_per_class.append(np.mean(confs))
                        conf_std_per_class.append(np.std(confs))
                    else:
                        conf_avg_per_class.append(np.nan)
                        conf_std_per_class.append(np.nan)

                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        test_num,
                        blur_size,
                        noise_level,
                        accuracy,
                        f1_weighted,
                        *precision_per_class,
                        *recall_per_class,
                        *f1_per_class,
                        confidence_overall,
                        *conf_avg_per_class,
                        *conf_std_per_class,
                        *counts_per_class
                    ])

                print(f"Test set {test_num}: Blur {blur_size}, Noise Level {noise_level}, Accuracy {accuracy}")


def apply_augmentations(images, blur_size, noise_level):
    '''
    Augmentation function for tensorflow 
    '''
    if blur_size == 0 and noise_level == 0:
        return images
    aug_images = []
    for img in images:
        # Convert to numpy array
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        # Apply blur
        if blur_size > 0:
            img_np = cv2.GaussianBlur(img_np, (blur_size, blur_size), 0)
        
        # Apply noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, img_np.shape).astype(np.float32)
            img_np = np.clip(img_np + noise, 0, 255)
        
        # Convert back to tensor
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()
        aug_images.append(img_tensor)
    
    return torch.stack(aug_images)

def test_augmented_resnet():
# Load datasets
    train_dataset, val_dataset, test_dataset1, test_dataset2, test_dataset3 = shiny_data.get_original()

    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load('resnet50.pt', map_location=device))
    model = model.to(device)
    model.eval()

    # Specify augmentations
    blur_sizes = [0,1,3,5,7,9,19]
    noise_levels = [0,1,3,5,10,20,30]

    # Init csv
    csv_file = 'resnet50_augmented_metrics.csv'
    test_datasets = [test_dataset1, test_dataset2, test_dataset3]
    test_set_names = ['1', '2', '3']
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'test_set', 'blur_size', 'noise_level', 'accuracy', 'f1',
            'precision_immune', 'precision_other', 'precision_stromal', 'precision_tumour',
            'recall_immune', 'recall_other', 'recall_stromal', 'recall_tumour',
            'f1_immune', 'f1_other', 'f1_stromal', 'f1_tumour',
            'confidence_overall',
            'confidence_immune_avg', 'confidence_other_avg', 'confidence_stromal_avg', 'confidence_tumour_avg',
            'confidence_immune_std', 'confidence_other_std', 'confidence_stromal_std', 'confidence_tumour_std',
            'count_pred_immune', 'count_pred_other', 'count_pred_stromal', 'count_pred_tumour'
        ])

    # Init loop
    for test_set_name, test_dataset in zip(test_set_names, test_datasets):
        test_loader = DataLoader(test_dataset, batch_size=128)
        for noise_level in noise_levels:
            for blur_size in blur_sizes:

                all_preds = []
                all_labels = []
                all_probs = []

                # Augment and predict
                with torch.no_grad():
                    for images, labels in test_loader:
                        aug_images = apply_augmentations(images, blur_size, noise_level)
                        aug_images, labels = aug_images.to(device), labels.to(device)
                        outputs = model(aug_images)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(probs, 1)
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs.cpu().numpy())

                # Metrics
                accuracy = accuracy_score(all_labels, all_preds)
                f1_weighted = f1_score(all_labels, all_preds, average='weighted')
                precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
                recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
                f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
                all_probs = np.array(all_probs)
                all_preds = np.array(all_preds)
                all_labels = np.array(all_labels)

                # Confidence metrics
                confidence_overall = np.mean(np.max(all_probs, axis=1))
                counts_pred_per_class = np.array([np.sum(all_preds == i) for i in range(all_probs.shape[1])])
                conf_avg_per_class = []
                conf_std_per_class = []
                for class_idx in range(all_probs.shape[1]):
                    mask = (all_preds == class_idx)
                    if np.sum(mask) > 0:
                        confs = all_probs[mask, class_idx]
                        conf_avg_per_class.append(np.mean(confs))
                        conf_std_per_class.append(np.std(confs))
                    else:
                        conf_avg_per_class.append(np.nan)
                        conf_std_per_class.append(np.nan)

                # Write to csv
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        test_set_name,
                        blur_size,
                        noise_level,
                        accuracy,
                        f1_weighted,
                        *precision_per_class,
                        *recall_per_class,
                        *f1_per_class,
                        confidence_overall,
                        *conf_avg_per_class,
                        *conf_std_per_class,
                        *counts_pred_per_class
                    ])
                print(f"{test_set_name} - Blur {blur_size}, Noise Level {noise_level}, Accuracy {accuracy}")

def test_augmented_cnn():

    #load in test sets
    _, _, X_test1, X_test2, X_test3, _, _, y_test1, y_test2, y_test3 = shiny_data.load_split_images()

    # Load model
    model = load_model('cnn_original.h5')

    # Specify augmentations
    blur_sizes = [0,1,3,5,7,9,19]
    noise_levels = [0,1,3,5,10,20,30]

    # Init csv
    csv_file = 'cnn_original_augmented_metrics.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'test_set', 'blur_size', 'noise_level', 'accuracy', 'f1',
            'precision_immune', 'precision_other', 'precision_stromal', 'precision_tumour',
            'recall_immune', 'recall_other', 'recall_stromal', 'recall_tumour',
            'f1_immune', 'f1_other', 'f1_stromal', 'f1_tumour',
            'confidence_overall',
            'confidence_immune_avg', 'confidence_other_avg', 'confidence_stromal_avg', 'confidence_tumour_avg',
            'confidence_immune_std', 'confidence_other_std', 'confidence_stromal_std', 'confidence_tumour_std',
            'count_pred_immune', 'count_pred_other', 'count_pred_stromal', 'count_pred_tumour'
        ])

    # Init testing loop
    for test_num, (X_test, y_test) in enumerate([(X_test1, y_test1), (X_test2, y_test2), (X_test3, y_test3)], 1):
        for noise_level in noise_levels:
            for blur_size in blur_sizes:

                # Apply augmentations
                X_test_aug = shiny_data.apply_noise(shiny_data.apply_blur(X_test, blur_size), std=noise_level)

                # Predict
                y_pred_probs = model.predict(X_test_aug)
                y_pred = np.argmax(y_pred_probs, axis=1)

                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1_weighted = f1_score(y_test, y_pred, average='weighted')
                precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
                recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
                f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

                # Confidence metrics
                confidence_overall = np.mean(np.max(y_pred_probs, axis=1))
                counts_pred_per_class = np.array([np.sum(y_pred == i) for i in range(y_pred_probs.shape[1])])
                conf_avg_per_class = []
                conf_std_per_class = []
                for class_idx in range(y_pred_probs.shape[1]):
                    mask = (y_pred == class_idx)
                    if np.sum(mask) > 0:
                        confs = y_pred_probs[mask, class_idx]
                        conf_avg_per_class.append(np.mean(confs))
                        conf_std_per_class.append(np.std(confs))
                    else:
                        conf_avg_per_class.append(np.nan)
                        conf_std_per_class.append(np.nan)  

                # Write to csv
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        test_num,
                        blur_size,
                        noise_level,
                        accuracy,
                        f1_weighted,
                        *precision_per_class,
                        *recall_per_class,
                        *f1_per_class,
                        confidence_overall,
                        *conf_avg_per_class,
                        *conf_std_per_class,
                        *counts_pred_per_class
                    ])

                print(f"Fold: {test_num}, Blur: {blur_size}, Noise: {noise_level}, Acc: {accuracy}")

if __name__ == "__main__":
    test_augmented_resnet()
    test_augmented_cnn()