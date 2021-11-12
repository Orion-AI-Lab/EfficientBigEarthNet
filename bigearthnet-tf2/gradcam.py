import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import Model
from textwrap import wrap


def GradCAM(model, image, y_true, patch_name, interpolant=0.0015):
    labels = [
        "Urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "Agro-forestry areas",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grassland and sparsely vegetated areas",
        "Moors, heathland and sclerophyllous vegetation",
        "Transitional woodland, shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
    ]

    last_conv_layer = next(
        x for x in model.layers[::-1] if isinstance(x, K.layers.Conv2D)
    )
    target_layer = model.get_layer(last_conv_layer.name)

    rgb_image = []
    rgb_image.append(tf.math.add(tf.multiply(image[2], tf.constant(675.88746967)), tf.constant(590.23569706)))
    rgb_image.append(tf.math.add(tf.multiply(image[1], tf.constant(582.87945694)), tf.constant(614.21682446)))
    rgb_image.append(tf.math.add(tf.multiply(image[0], tf.constant(572.41639287)), tf.constant(429.9430203)))
    rgb_image = tf.stack(tf.squeeze(rgb_image)).numpy().transpose(1,2,0)
    rgb_image = tf.maximum(rgb_image, 0) / tf.math.reduce_max(rgb_image).numpy()

    true_label_indexes = np.nonzero(y_true)[0]
    patch_name = patch_name.decode("utf-8")

    # Compute Gradient of Top Predicted Class
    with tf.GradientTape(persistent=True) as tape:
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        tape.watch(gradient_model.get_layer(last_conv_layer.name).variables)
        conv2d_out, prediction = gradient_model(image)
        # Obtain the Prediction Loss

        for i, loss in enumerate(prediction[0]):
            # Gradient() computes the gradient using operations recorded
            # in context of this tape
            gradients = tape.gradient(loss, conv2d_out)

            # Obtain Depthwise Mean
            weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
            heatmap = conv2d_out @ weights[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = heatmap.numpy()

            heatmap = cv2.resize(heatmap, (image[0].shape[1], image[0].shape[1]), fx=0, fy=0)
            jet_heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            jet_heatmap = cv2.applyColorMap(jet_heatmap, cv2.COLORMAP_JET)
            jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)

            # Superimpose the heatmap on original image
            superimposed_img = jet_heatmap * interpolant + rgb_image
            superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

            # Save the superimposed image
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(rgb_image)

            t_label = False
            p_label = False

            true_labels = ''
            for enum, index in enumerate(true_label_indexes):
                true_labels += '[{}] {} '.format(enum, labels[index])

            if i in true_label_indexes:
                t_label = True
                ax1.set_title("\n".join(wrap('True label: ' + true_labels, 45)), fontsize=8, wrap=True, color='g')
            else:
                ax1.set_title("\n".join(wrap('True label: ' + true_labels, 45)), fontsize=8, wrap=True, color='r')
            ax1.axes.get_xaxis().set_visible(False)
            ax1.axes.get_yaxis().set_visible(False)

            ax2.imshow(superimposed_img)
            if loss.numpy() >= 0.5:
                p_label = True
                ax2.set_title("\n".join(wrap('Predicted label: ' + labels[i] + ' [P=' + str(np.around(loss.numpy(),2)) + ']', 45)), fontsize=8, wrap=True, color='g')
            else:
                ax2.set_title("\n".join(wrap('Predicted label: ' + labels[i] + ' [P=' + str(np.around(loss.numpy(),2)) + ']', 45)), fontsize=8, wrap=True, color='r')            
            ax2.axes.get_xaxis().set_visible(False)
            ax2.axes.get_yaxis().set_visible(False)
            
            #plt.tight_layout()
            if all([t_label, p_label]):
                plt.savefig('gradcam/true_positive/' + patch_name + '_' + str(i) + '.png', bbox_inches='tight')
            elif (t_label == True and p_label == False):
                plt.savefig('gradcam/false_negative/' + patch_name + '_' + str(i) + '.png', bbox_inches='tight')
            elif (t_label == False and p_label == True):
                plt.savefig('gradcam/false_positive/' + patch_name + '_' + str(i) + '.png', bbox_inches='tight')
            plt.close('all')
