from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


'''
Some times I need normalizad images
'''

def load_images(dir, norm=False):
    images = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        image = cv2.imread(path) 
        if image is not None:
            image = image.astype(np.float32) 
            if norm:
                image = image.astype(np.float32) / 255.0  # Normalize pixel values
            images.append(image)
    return images

def load_masks(dir):
    images = []
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        if image is not None:
            image = image.astype(np.float32) / 255.0  # Normalize pixel values
            images.append(image.reshape(96, 96, 1))
    return images


# just to plot masks 
def plot_img_mask(img, mask, pred):

    pred = np.where(pred <= 0.5, 0, 1)
   
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
   
    axes[0].imshow(img)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Predicted')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def augmented_dataset(image_directory, masks_directory, seed, norm):  
    
    images = load_images(image_directory, norm)
    masks = load_masks(masks_directory)
    
    # no augmentation for the test set!
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=seed)
    
    augmented_X_train = [cv2.flip(i, 1) for i in X_train]
    augmented_y_train = [cv2.flip(i, 1).reshape(96, 96, 1) for i in y_train]

    X_train = np.array(X_train + augmented_X_train)
    y_train = np.array(y_train + augmented_y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


'''
If you want to show some other metrics, just uncomment
'''
def plot_history(history, x_lim, y_lim):
    #val_iou_score = history.history['val_iou_score']
    #iou_score = history.history['iou_score']
    val_f1_score = history.history['val_f1-score']
    #f1_score = history.history['f1-score']
    val_loss = history.history['val_loss']
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)

    # Plotting
    plt.figure(figsize=(12, 6))

    #plt.plot(epochs, val_iou_score, 'b', label='Validation IoU Score')
    #plt.plot(epochs, iou_score, 'g', label='IoU Score')
    #plt.plot(epochs, f1_score, 'r', label='F1 Score')
    plt.plot(epochs, val_f1_score, 'c', label='Validation F1 Score')
    plt.plot(epochs, val_loss, 'm', label='Validation Loss')
    plt.plot(epochs, loss, 'y', label='Training Loss')

    plt.title('Training Metrics', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Score/Loss', fontsize=14)
    plt.legend(fontsize=12)

    plt.grid(True)
    plt.xlim(1, x_lim)
    plt.ylim(0, y_lim)
    plt.show()