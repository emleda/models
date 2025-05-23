import shiny_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost
import time
import joblib
import xgboost as xgb
import torch
from torchvision import transforms, models
import torch.nn as nn
import evaluation
from tensorflow.keras.models import load_model

def loading(img_path):
    img = shiny_data.load_resize(img_path)
    
    #load pca
    pca = joblib.load('Base_pca.joblib')

    #load xgb model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('xgboost.json')

    #load resnet model
    device = torch.device("cpu")
    rn_model = models.resnet50(pretrained=False)
    num_ftrs = rn_model.fc.in_features
    rn_model.fc = nn.Linear(num_ftrs, 4)
    rn_model.load_state_dict(torch.load('resnet50.pt', map_location=device))
    rn_model = rn_model.to(device)
    rn_model.eval()

    #load cnn model
    cnn_model = load_model('cnn_original.h5')

    return img, pca, xgb_model, rn_model, cnn_model


def predict_xgb(image, pca, model):
    blur_sizes = [0,1,3,5,7,9,19]
    noise_levels = [0,1,3,5,10,20,30,50]
    
    preds = np.empty((len(noise_levels), len(blur_sizes)), dtype=int)

    for i, noise_level in enumerate(noise_levels):
        for j, blur_size in enumerate(blur_sizes):
            img_aug = shiny_data.apply_noise(shiny_data.apply_blur(image[np.newaxis, ...], blur_size), std=noise_level)
            img_aug_flat = img_aug.reshape(1, -1)
            img_pca = pca.transform(img_aug_flat)
            y_pred = model.predict(img_pca)
            preds[i, j] = y_pred[0]

    return preds, blur_sizes, noise_levels

def predict_resnet(img, model=None, device="cpu"):

    blur_sizes = [0,1,3,5,7,9,19]
    noise_levels = [0,1,3,5,10,20,30]
    img = torch.tensor(img).permute(2,0,1).float()

    preds = np.empty((len(noise_levels), len(blur_sizes)), dtype=int)

    with torch.no_grad():
        for i, noise_level in enumerate(noise_levels):
            for j, blur_size in enumerate(blur_sizes):
                # Apply augmentations: image is a tensor (C,H,W), add batch dim
                img_aug = evaluation.apply_augmentations(img.unsqueeze(0), blur_size, noise_level)
                img_aug = img_aug.to(device)
                outputs = model(img_aug)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                preds[i, j] = pred

    return preds, blur_sizes, noise_levels

def predict_cnn(img, model):
    blur_sizes = [0,1,3,5,7,9,19]
    noise_levels = [0,1,3,5,10,20,30]

    preds = np.empty((len(noise_levels), len(blur_sizes)), dtype=int)

    for i, noise_level in enumerate(noise_levels):
        for j, blur_size in enumerate(blur_sizes):
            img_aug = shiny_data.apply_noise(shiny_data.apply_blur(img[np.newaxis, ...], blur_size), std=noise_level)
            y_pred_probs = model.predict(img_aug)
            y_pred = np.argmax(y_pred_probs, axis=1)[0]
            preds[i, j] = y_pred

    return preds, blur_sizes, noise_levels

def plot_prediction_heatmap(preds, blur_sizes, noise_levels):
    class_names = ['immune', 'other', 'stromal', 'tumour']
    plt.figure(figsize=(10, 6))
    cmap = sns.color_palette("tab10", n_colors=4)  # categorical colors for 4 classes
    sns.heatmap(preds, xticklabels=blur_sizes, yticklabels=noise_levels, 
                cmap=cmap, cbar_kws={'ticks': np.arange(4), 'label': 'Predicted Class'})
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticks(np.arange(4) + 0.5)
    cbar.set_ticklabels(class_names)
    plt.xlabel('Blur Size')
    plt.ylabel('Noise Level')
    plt.title('Prediction Heatmap by Augmentation')
    plt.show()

start = time.time()
img, pca, xgb_model, rn_model, cnn_model = loading('data/original_data/B_Cells/B_Cells_2312.png')

#xgboost
xg_preds, blur_sizes, noise_levels = predict_xgb(img, pca, xgb_model) 
plot_prediction_heatmap(xg_preds, blur_sizes, noise_levels)
#resnet
rn_preds, blur_sizes, noise_levels = predict_resnet(img, rn_model) 
plot_prediction_heatmap(rn_preds, blur_sizes, noise_levels)
#cnn
cnn_preds, blur_sizes, noise_levels = predict_cnn(img, cnn_model) 
plot_prediction_heatmap(cnn_preds, blur_sizes, noise_levels)

end = time.time()
duration = end - start
print(f"Render time: {duration}")

