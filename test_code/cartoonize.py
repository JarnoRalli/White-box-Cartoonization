import os
import cv2
import numpy as np
import tensorflow as tf 
import network
import guided_filter
from tqdm import tqdm
import argparse

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image


def cartoonize(load_folder, save_folder, model_path, rho: float):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=rho, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)

    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            if image is None:
                continue
            image_shape = image.shape
            image = resize_crop(image)
            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output)+1)*127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            output = cv2.resize(output, (image_shape[1], image_shape[0]))
            cv2.imwrite(save_path, output)
        except:
            print('cartoonize {} failed'.format(load_path))

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model_path", help="model path", default="saved_models")
    argParser.add_argument("-i", "--input_folder", help="input folder", default="test_images")
    argParser.add_argument("-o", "--output_folder", help="output folder", default="cartoonized_images")
    argParser.add_argument("-r", "--rho", help="image sharpness", type=float, default=1.0)
    args = argParser.parse_args()

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    cartoonize(args.input_folder, args.output_folder, args.model_path, args.rho)

