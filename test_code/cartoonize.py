import os
import sys
import cv2
import numpy as np
import tensorflow as tf 
from tqdm import tqdm
import argparse
import network
import guided_filter

def path_is_directory(input_string):
    if os.path.isdir(input_string):
        return input_string
    else:
        raise NotADirectoryError(input_string)

def path_exists(input_string):
    if os.path.exists(input_string):
        return input_string
    else:
        raise FileNotFoundError(input_string)

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

def cartoonize(input_path, save_folder, model_path, rho):
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

    file_list = []

    # If the 'input_path' points to a directory, we traverse all the subdirectories and add all the files to be processed later on.
    if os.path.isdir(input_path):
        # Walk through the directory 'input_path' and append the results to a list of dictionaries
        for abs_path, dirs, files in os.walk(input_path):
            for file_ in files:
                # Relative output directory
                rel_path = os.path.relpath(abs_path, input_path)
                # Combined output folder
                combined_output_folder = os.path.join(save_folder, rel_path)
                file_dict = {"input_image": os.path.join(abs_path, file_), "output_image": os.path.join(combined_output_folder, file_)}
                file_list.append(file_dict)
                if not os.path.exists(combined_output_folder):
                    os.makedirs(combined_output_folder)
    # If the 'input_path' points to a file, then we add this file to be processed later on.
    elif os.path.isfile(input_path):
        file_list = [{"input_image": str(input_path), "output_image": os.path.join(save_folder, os.path.basename(input_path))}]
    else:
        raise Exception(f"URL '{file_dir}' doesn't appear to be a directory nor a file")

    # Iterate through the list of dictionaries
    for file_dict in file_list:
        image = cv2.imread(file_dict["input_image"])
        if image is None:
            continue

        print(f'{file_dict["input_image"]} -> {file_dict["output_image"]}')
        image_shape = image.shape
        image = resize_crop(image)
        batch_image = image.astype(np.float32)/127.5 - 1.0
        batch_image = np.expand_dims(batch_image, axis=0)
        output = sess.run(final_out, feed_dict={input_photo: batch_image})
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.resize(output, (image_shape[1], image_shape[0]))
        cv2.imwrite(file_dict["output_image"], output)

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model_path", help="model path", type=path_is_directory, default="saved_models")
    argParser.add_argument("-i", "--input", help="path to a directory or a single image to process", type=path_exists, default="test_images")
    argParser.add_argument("-o", "--output_folder", help="path to output folder", type=path_is_directory, default="cartoonized_images")
    argParser.add_argument("-r", "--rho", help="image sharpness", type=float, default=1.0)
    args = argParser.parse_args()

    try:
        cartoonize(args.input, args.output_folder, args.model_path, args.rho)
    except Exception as error:
        print(error)
        sys.exit(1)
    else:
        sys.exit(0)

