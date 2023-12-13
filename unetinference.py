import os
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping


model1=keras.models.load_model("/home/arnav/Downloads/CV Project/unetDAMModel2.h5")


def load_and_preprocess_data(image_path):
    # Read and decode image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
     # Resize the image to your desired dimensions
    image=cv2.resize(image,(128,128))
    # Normalize image to [0, 1] if needed
    image = image / 255.0

    return image




import numpy as np

def load_and_preprocess_mask(image_path):
    # Convert BGR image to RGB
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(128,128))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_gray=image_gray/255.0
    #one_hot_mask = tf.keras.utils.to_categorical( image_gray/255.0, num_classes=2)


    return image_gray



val_data_dir = '/home/arnav/Downloads/CV Project/DeepCrack/test_img'  # Replace with the path to your validation data directory
val_mask_dir = '/home/arnav/Downloads/CV Project/DeepCrack/test_lab'  # Replace with the path to your validation masks directory

val_data_paths=os.listdir(val_data_dir)
val_mask_paths=os.listdir(val_mask_dir)



# Ensure the order of validation image and mask file paths matches
val_data_paths.sort()
val_mask_paths.sort()


val_img=[load_and_preprocess_data('/home/arnav/Downloads/CV Project/DeepCrack/test_img/'+image_path) for image_path in val_data_paths]
val_mask=[load_and_preprocess_mask('/home/arnav/Downloads/CV Project/DeepCrack/test_lab/'+image_path) for image_path in val_mask_paths]

TestImg =np.array(val_img)
TestMask =np.array(val_mask)


predicted_mask=model1.predict(TestImg)
print(predicted_mask[0].shape)


argmax_masks = np.empty((predicted_mask.shape[0], predicted_mask.shape[1], predicted_mask.shape[2]))

for i in range(predicted_mask.shape[0]):
    argmax_masks[i]=np.argmax(predicted_mask[i],axis=-1)






def calculate_iou(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU) for binary masks.

    Parameters:
    - y_true: Ground truth binary mask
    - y_pred: Predicted binary mask

    Returns:
    - IoU (Intersection over Union)
    """
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_mean_iou(y_true_array, y_pred_array):
    """
    Calculate the mean Intersection over Union (IoU) for arrays of masks.

    Parameters:
    - y_true_array: Array of ground truth binary masks
    - y_pred_array: Array of predicted binary masks

    Returns:
    - Mean IoU
    """
    iou_values = []
    for y_true, y_pred in zip(y_true_array, y_pred_array):
        iou = calculate_iou(y_true, y_pred)
        iou_values.append(iou)

    mean_iou = np.mean(iou_values)
    return mean_iou



print(TestMask[0].shape,argmax_masks[0].shape)

meaniou=calculate_mean_iou(TestMask, argmax_masks)
print(meaniou)



import matplotlib.pyplot as plt 
x=TestMask[40]
x=x*255.0
plt.imshow(x,cmap="gray")
plt.show()
y=argmax_masks[40]*255.0
plt.imshow(y,cmap="gray")
plt.show()
      
