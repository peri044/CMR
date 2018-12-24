import tensorflow as tf
import numpy as np
import argparse
from dnn_library import *
from skimage import io
from skimage.transform import resize
from preprocessing import preprocessing_factory
from nets import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import cv2
import pdb
from skimage.color import gray2rgb

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def vgg_preprocess(image, base_arch):
    
    """
    Pre-processing for base network.
    """
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(base_arch, is_training=False)
    return tf.expand_dims(image_preprocessing_fn(image, 224, 224), 0)
    
def preprocess_caption(caption):
    """
    Preprocess caption
    """
    new_caption = caption.replace('.', '').lower()
    new_caption = new_caption.replace(',', '').strip()
    
    return new_caption
    
def load_flickr_data(args):
    """
    Load flickr data
    """
    images_path = open(os.path.join(args.data_path, 'train.token')).readlines()
    image_data=[]
    caption_data=[]
    for sample in images_path:
        sample=sample.strip()
        image=sample.split('#')[0]
        caption = sample.split('\t')[1]
        preprocessed_caption = preprocess_caption(caption)
        image_data.append(os.path.join(args.root_path, image))
        caption_data.append(preprocessed_caption)
        
    print "Done loading data"
    return image_data, caption_data
    
def load_coco_data(args):
    """
    Load MSCOCO data
    """
    image_ids=open(args.data_path, 'r').readlines()
    image_data=[]
    for id in image_ids:
        if id.find('train2014')!=-1:
            image_data.append((os.path.join(args.root_path, 'train2014', id.strip())))
        else:
            image_data.append((os.path.join(args.root_path, 'val2014', id.strip())))
            
    caps_path = open(args.caps_path, 'r').readlines()
    caption_data = [preprocess_caption(caption.strip()) for caption in caps_path]
        
    return image_data, caption_data
    
def _bytes_feature(value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
        
def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])
    
def write_tfrecord(feature, caption):
    feature_lists = tf.train.FeatureLists(feature_list={"caption": _bytes_feature_list(caption.split(' '))})
    context = tf.train.Features(feature={"image": _bytes_feature(feature.tostring())})
                                
    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    
    return sequence_example
    
def write_image_tfrecord(image_path, caption):
    image = tf.gfile.FastGFile(image_path, "rb").read()
    feature_lists = tf.train.FeatureLists(feature_list={"caption": _bytes_feature_list(caption.split(' '))})
    context = tf.train.Features(feature={"image": _bytes_feature(image)})
                                
    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    
    return sequence_example
       
        
def feature_extractor(args, image, reuse=None, is_training=False):
    """
    Builds the model architecture
    """			
    # Define the network and pass the input image
    with slim.arg_scope(model[args.base_arch]['scope']):
            logits, end_points = model[args.base_arch]['net'](image, num_classes=1000, is_training=False) #model[args.base_arch]['num_classes']
    

    # features 
    feat_anchor = tf.squeeze(end_points[model[args.base_arch]['end_point']])
        
    return feat_anchor

def augment_image(image):
    """
    Performs 10-cropping on a 256x256 image
    """
    resized_image = resize(image, [256, 256], preserve_range=True)
    augmented_im = np.zeros((10, 224, 224, 3))
    augmented_im[0, :] = resized_image[0:224, 0:224, :]
    augmented_im[1, :] = resized_image[0:224, 32:256, :]
    augmented_im[2, :] = resized_image[32:256, 0:224, :]
    augmented_im[3, :] = resized_image[32:256, 32:256, :]
    augmented_im[4, :] = resized_image[16:240, 16:240, :]
    augmented_im[5, :] = augmented_im[0, :][:, ::-1, :]
    augmented_im[6, :] = augmented_im[1, :][:, ::-1, :]
    augmented_im[7, :] = augmented_im[2, :][:, ::-1, :]
    augmented_im[8, :] = augmented_im[3, :][:, ::-1, :]
    augmented_im[9, :] = augmented_im[4, :][:, ::-1, :]
   
    return augmented_im
        
def main(args):

    # Define the input 
    input_image = tf.placeholder(shape=[None, None, 3], dtype=tf.float32, name='input_image')
    preprocessed_image = vgg_preprocess(input_image, args.base_arch)

    # Extract the features
    # preprocessed_image = tf.placeholder(shape=[10, 224, 224, 3], dtype=tf.float32, name='preprocessed_image')
    features = feature_extractor(args, preprocessed_image, is_training=False)
    
    # Define the saver
    saver = tf.train.Saver()

    # Load the data file
    if args.dataset=='flickr':
        image_data, caption_data = load_flickr_data(args)
        offset = 2000*5
        interval=5
    elif args.dataset=='mscoco':
        image_data, caption_data = load_coco_data(args)
        offset = 0
        interval=1
        
    print "Total number of samples: {}".format(len(image_data))

    # Define the TF record writer
    tfrecord_writer = tf.python_io.TFRecordWriter(args.record_path)
    image_features_np = np.zeros((len(image_data) - offset, 2048))
    # image_features_np = np.zeros((1000, 2048))
    count=0
    image_to_feature={}
    
    with tf.Session() as sess:
        # Restore pre-trained weights
        saver.restore(sess, args.checkpoint)
        for i in range(0, len(image_data), interval):
            if i%1000==0 and i!=0: print "Extracted: {}/{}".format(i, (len(image_data)-offset))
            # sample is of form (image, caption)
            image = io.imread(image_data[i])
            if len(image.shape)!=3: 
                image=gray2rgb(image)
            # pdb.set_trace()
            # augment image and mean subtraction
             
            # images = augment_image(image) 
            # mean_subtracted_image = images - np.array([_R_MEAN, _G_MEAN, _B_MEAN])
            # Run  the session to extract features
            # pdb.set_trace()
            feature_val = sess.run(features, feed_dict={input_image: image})
            # pdb.set_trace()
            # mean_img_features = np.mean(feature_val, axis=0)
            for cap_id in range(i*5, i*5 + 5): # Change to i*5 : i*5 + 5 for MSCOCO
                sequence_example = write_tfrecord(feature_val.astype(np.float32), preprocess_caption(caption_data[cap_id]))
                # sequence_example = write_image_tfrecord(image_data[i], preprocess_caption(caption_data[cap_id]))
                tfrecord_writer.write(sequence_example.SerializeToString())
            image_features_np[count] = np.squeeze(feature_val)
            count+=1
        # np.save(os.path.join(args.save_path, 'flickr_test_r152_precomp.npy'), image_features_np[0:len(image_data) - offset : 5, :])
        np.save(os.path.join(args.save_path, 'coco_train_r152_precomp.npy'), image_features_np)
        # tfrecord_writer.close()
        print "Total number of image features: {}".format(count)
        print "Done extracting Image features !!"
       
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mscoco', help='Data file')
    parser.add_argument('--data_path', type=str, default='/shared/kgcoe-research/mil/cvs_cvpr18/coco/train_filenames.txt', help='Data file')
    parser.add_argument('--caps_path', type=str, default='/shared/kgcoe-research/mil/peri/scan_data/data/coco_precomp/train_caps.txt', help='Data file')
    parser.add_argument('--save_path', type=str, default='/shared/kgcoe-research/mil/cvs_cvpr18/coco', help='Data file')
    parser.add_argument('--root_path', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images', help='Root_path')
    parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_dual_r152_train.tfrecord', help='Root_path')
    parser.add_argument('--base_arch', type=str, default='resnet_v1_152', help='Base architecture of CNN')
    parser.add_argument('--checkpoint', type=str, default='/shared/kgcoe-research/mil/peri/tf_checkpoints/resnet_v1_152.ckpt', help='Path to checkpoint')
    args=parser.parse_args()
    main(args)
