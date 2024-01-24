import unet
from util import *
from PIL import Image
from pylab import *
import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *

from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau


x = tf.random.uniform([3, 3])
n_images_to_show = 0
smooth = 1

img_dir = '../data/'
DATA_PATH = '../data/'

folders = ['train_frames/train', 'train_masks/train', 'val_frames/val', 'val_masks/val']

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


#checking gpu is available
fix_gpu()

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())
print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))
print("Device name: {}".format((x.device)))
print(tf.executing_eagerly())




#data augmention part
'''

import albumentations as A
import cv2
transform = A.Compose([
    A.HorizontalFlip(p=0.5),

    A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

    A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
    A.RandomCrop(height=512, width=512, always_apply=True),

    A.IAAAdditiveGaussianNoise(p=0.2),
    A.IAAPerspective(p=0.5),

    A.OneOf(
        [

            A.RandomBrightness(p=1),
            A.RandomGamma(p=1),
        ],
        p=0.9,
    ),

    A.OneOf(
        [
            A.IAASharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ],
        p=0.9,
    ),

    A.OneOf(
        [
            A.RandomContrast(p=1)
        ],
        p=0.9,
    ),
],is_check_shapes=False)

'''



def _read_to_tensor(fname, output_height=512, output_width=512, normalize_data=False):
    '''Function to read images from given image file path, and provide resized images as tensors
        Inputs:
            fname - image file path
            output_height - required output image height
            output_width - required output image width
            normalize_data - if True, normalize data to be centered around 0 (mean 0, range 0 to 1)
        Output: Processed image tensors
    '''

    # Read the image as a tensor
    img_strings = tf.io.read_file(fname)
    imgs_decoded = tf.image.decode_jpeg(img_strings)

    # Resize the image
    output = tf.image.resize(imgs_decoded, [output_height, output_width])

    # Normalize if required
    if normalize_data:
        output = (output - 128) / 128
    return output



# Required image dimensio
def read_images(img_dir):
    '''Function to get all image directories, read images and masks in separate tensors
        Inputs:
            img_dir - file directory
        Outputs
            frame_tensors, masks_tensors, frame files list, mask files list
    '''

    frame_path=img_dir+"train/"
    mask_path = img_dir+"mask/"
    print(os.listdir(frame_path))

    # Get the file names list from provided directory
    frame_list = [f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))]
    mask_list = [f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path, f))]

    # Separate frame and mask files lists, exclude unnecessary files
    frames_list = [file for file in frame_list if ('_L' not in file) and ('png'  in file)]
    masks_list = [file for file in mask_list if ('_L' in file) and ('png'  in file)]

    print('{} frame files found in the provided directory.'.format(len(frames_list)))
    print('{} mask files found in the provided directory.'.format(len(masks_list)))

    # Create file paths from file names
    frames_paths = [os.path.join(frame_path, fname) for fname in frames_list]
    masks_paths = [os.path.join(mask_path, fname) for fname in masks_list]

    # Create dataset of tensors
    frame_data = tf.data.Dataset.from_tensor_slices(frames_paths)
    masks_data = tf.data.Dataset.from_tensor_slices(masks_paths)

    # Read images into the tensor dataset
    frame_tensors = frame_data.map(_read_to_tensor)
    masks_tensors = masks_data.map(_read_to_tensor)

    print('Completed importing {} frame images from the provided directory.'.format(len(frames_list)))
    print('Completed importing {} mask images from the provided directory.'.format(len(masks_list)))

    return frame_tensors, masks_tensors, frames_list, masks_list

def generate_image_folder_structure(frames, masks, frames_list, masks_list):
    '''Function to save images in the appropriate folder directories
        Inputs:
            frames - frame tensor dataset
            masks - mask tensor dataset
            frames_list - frame file paths
            masks_list - mask file paths
    '''
    # Create iterators for frames and masks
    frame_batches = tf.compat.v1.data.make_one_shot_iterator(
        frames)  # outside of TF Eager, we would use make_one_shot_iterator
    mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks)

    # Iterate over the train images while saving the frames and masks in appropriate folders
    dir_name = './train'
    for file in zip(frames_list[:-round(0.2 * len(frames_list))], masks_list[:-round(0.2 * len(masks_list))]):
        # Convert tensors to numpy arrays
        frame = frame_batches.next().numpy().astype(np.uint8)
        mask = mask_batches.next().numpy().astype(np.uint8)

        '''
        transformed = transform(image=frame, mask=mask)
        transformed_image = transformed['image']

        transformed_masks = transformed['mask']

        # Convert numpy arrays to images
        frame = Image.fromarray(transformed_image)
        mask = Image.fromarray(transformed_masks)
'''
        frame = Image.fromarray(frame)
        mask = Image.fromarray(mask)


        # Save frames and masks to correct directories
        frame.save(DATA_PATH + '{}_frames/{}'.format(dir_name, dir_name) + '/' + file[0])
        mask.save(DATA_PATH + '{}_masks/{}'.format(dir_name, dir_name) + '/' + file[1])

    # Iterate over the val images while saving the frames and masks in appropriate folders
    dir_name = 'val'
    for file in zip(frames_list[-round(0.2 * len(frames_list)):], masks_list[-round(0.2 * len(masks_list)):]):
        # Convert tensors to numpy arrays
        frame = frame_batches.next().numpy().astype(np.uint8)
        mask = mask_batches.next().numpy().astype(np.uint8)

        # Convert numpy arrays to images
        frame = Image.fromarray(frame)
        mask = Image.fromarray(mask)

        # Save frames and masks to correct directories
        frame.save(DATA_PATH + '{}_frames/{}'.format(dir_name, dir_name) + '/' + file[0])
        mask.save(DATA_PATH + '{}_masks/{}'.format(dir_name, dir_name) + '/' + file[1])

    print("Saved {} frames to directory {}".format(len(frames_list), DATA_PATH))
    print("Saved {} masks to directory {}".format(len(masks_list), DATA_PATH))









frame_tensors, masks_tensors, frames_list, masks_list = read_images(img_dir)

frame_batches = tf.compat.v1.data.make_one_shot_iterator(
    frame_tensors)  # outside of TF Eager, we would use make_one_shot_iterator
mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks_tensors)



for i in range(n_images_to_show):
    # Get the next image from iterator
    frame = frame_batches.next().numpy().astype(np.uint8)
    mask = mask_batches.next().numpy().astype(np.uint8)

    # Plot the corresponding frames and masks
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(frame)
    fig.add_subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()




# Create folders to hold images and masks




for folder in folders:
    try:
        os.makedirs(DATA_PATH + folder)
    except Exception as e: print(e)





generate_image_folder_structure(frame_tensors, masks_tensors, frames_list, masks_list)

# generate_image_folder_structure(train_frames, train_masks, val_files, 'val')




# Normalizing only frame images, since masks contain label info






model = unet.unet(input_size=(512, 512, 3), n_class=5)







metrics = [dice_coef, iou_coef, 'accuracy']



model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=metrics)



batch_size = 8
validation_steps = 2
num_epochs = 25





earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('unet.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


result = model.fit_generator(TrainAugmentGenerator(), steps_per_epoch=20 ,
                validation_data = ValAugmentGenerator(),
                validation_steps = validation_steps, epochs=num_epochs, callbacks=[ mcp_save, reduce_lr_loss  ,earlyStopping  ])

N = len(result.history['loss'])

#Plot the model evaluation history
plt.style.use("ggplot")
fig = plt.figure(figsize=(20,8))

fig.add_subplot(1,2,1)
plt.title("Training Loss")
plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
plt.ylim(0, 1)

fig.add_subplot(1,2,2)
plt.title("Training Accuracy")
plt.plot(np.arange(0, N), result.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), result.history["val_accuracy"], label="val_accuracy")
plt.ylim(0, 1)

plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()


