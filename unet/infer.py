from util import *
from matplotlib import pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model("unet.h5",custom_objects={"dice_coef":dice_coef,"iou_coef":iou_coef})

testing_gen = ValAugmentGenerator()

batch_img,batch_mask = next(testing_gen)
pred_all= model.predict(batch_img)


for i in range(0, np.shape(pred_all)[0]):
    fig = plt.figure(figsize=(20, 8))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(batch_img[i])
    ax1.title.set_text('Actual frame')


    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Ground truth labels')
    ax2.imshow(onehot_to_rgb(batch_mask[i], id2code))


    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Predicted labels')
    ax3.imshow(onehot_to_rgb(pred_all[i], id2code))


    plt.show()