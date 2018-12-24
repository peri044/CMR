import tensorflow as tf
import numpy as np
import os
from preprocessing import preprocessing_factory
import argparse
import pdb

class FlickrDataLoader(object):
    """
    Data loader and writer object for MSCOCO dataset
    """
    def __init__(self, path=None, precompute=False):
        self.data_path=path
        self.precompute=precompute
        
    def _int64_feature(self, value):
        """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        
    def _bytes_feature(self, value):
        """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
        
    def _bytes_feature_list(self, values):
        """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
        return tf.train.FeatureList(feature=[self._bytes_feature(v) for v in values])
        
    def _process_caption(self, caption):
        
        processed_caption = caption.replace(',', '')
        processed_caption = processed_caption.replace('.', '')
        processed_caption = processed_caption.lower()
        return processed_caption
        

    def _make_single_example(self, image_path, caption, precompute=True):
        """
        Make a single example in a TF record
        """
        if not precompute:
            #pdb.set_trace()
            image = tf.gfile.FastGFile(image_path, "rb").read()
            caption=self._process_caption(caption)
            caption_list = caption.split(' ')
            feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
                       
            context = tf.train.Features(feature={
                            "image": self._bytes_feature(image)})
                            
        else:
            caption=self._process_caption(caption)
            caption_list = caption.split(' ')
            feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
            context = tf.train.Features(feature={"image": self._bytes_feature(image_path.tostring())})
                                        
        sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

        return sequence_example
        
    def _make_dataset(self, phase, record_path, num=None):


        path = '/shared/kgcoe-research/mil/Flickr30k/flickr30k_images/train.token'
        file_names = open(path).read().split('\n')
        tfrecord_writer = tf.python_io.TFRecordWriter(record_path)
        img = []
        cap = []
        
        if num is None:
            num=len(file_names) - 10001   
        count=0
        for id in range(num):
            #print(id)
            split_line = file_names[id].split('\t') 
            img.append(split_line[0].split('#')[0])
            cap.append(split_line[1])
        count=0
        resnet_features = np.load('flickr_features.npy')
        print(resnet_features.shape)
        for im_idx in range(29783):
            #print(img[im_idx],cap[im_idx])
            if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num)
            #if os.path.join(args.data_path,'flickr30k_images' + img[im_idx]).strip().find('flickr30k_images') !=-1:
            #    image = os.path.join(args.train_dir, img[im_idx].strip())
            #else:
            #    raise ValueError("Invalid Image")
            for cap_idx in range(im_idx*5, im_idx*5 +5):
                example = self._make_single_example(resnet_features[im_idx,:], cap[cap_idx].strip())
                tfrecord_writer.write(example.SerializeToString())
                count+=1
            
        print "Done generating TF records"
        
    def _precomputed_dataset(self, phase, feature_path, record_path, data_path, num=None):
        """
        Write the whole dataset to a TF record.
        """

        train_ids = open(os.path.join(args.data_path, phase+'.token')).read().split('\n')
        tfrecord_writer = tf.python_io.TFRecordWriter(record_path)
        img = []
        cap = []
        
        if num is None:
            num=len(train_img_features)
        count=0
        for id in range(num - 1):
            #print(id)
            split_line = file_names[id].split('\t') 
            img.append(split_line[0].split('#')[0])
            cap.append(split_line[1])
        print(cap[0])
        print(img[0])
            
        for im_idx in range(num):
            if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num*5)                
            example = self._make_single_example(train_img_features[im_idx], train_ids[im_idx].strip().split('\t')[1], precompute=True)
            tfrecord_writer.write(example.SerializeToString())
            count+=1
            
        print "Done generating TF records"
        
    def _inception_preprocess(self, image):
        
        """
        Pre-processing for inception. Convert the range to [-1, 1]
        """
        return (2.0/255)*image -1.
        
    def _vgg_preprocess(self, image):
        
        """
        Pre-processing for VGG.
        """
        image_preprocessing_fn = preprocessing_factory.get_preprocessing('resnet_v2_152', is_training=True)
        return image_preprocessing_fn(image, 224, 224)
        
    def _parse_single_example(self, example_proto):
        context, sequence = tf.parse_single_sequence_example(
                                        example_proto,
                                        context_features={
                                          "image": tf.FixedLenFeature([], dtype=tf.string),
                                        },
                                        sequence_features={
                                          "caption": tf.FixedLenSequenceFeature([], dtype=tf.string)
                                        })
        if not self.precompute:
            image = tf.image.decode_jpeg(context["image"], channels=3)
            image = self._vgg_preprocess(image)
        else:
            image = tf.decode_raw(context['image'], out_type=tf.float32)
        caption = tf.cast(sequence["caption"], tf.string)[:50] # max_len allowed is 50
        
        return image, caption, tf.size(caption)

    def _read_data(self, record_path, batch_size, phase='train', num_epochs=10):
        dataset = tf.data.TFRecordDataset(record_path)
        if phase !='val':
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(map_func=self._parse_single_example, num_parallel_calls=4)
        if not self.precompute:
            image_shape=[224, 224, 3]
        else:
            image_shape=[2048]
        # Pads the caption to the max caption size in a batch
        dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, 
                                       padded_shapes=(
                                            image_shape,  # img
                                            tf.TensorShape([None]),  # caption
                                            tf.TensorShape([])),# src_len
                                       padding_values=(0.,  # src 
                                                      '</s>',  # caption_pad
                                                       0))) # Seq len
        
        dataset = dataset.prefetch(buffer_size=batch_size)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        image, caption, seq_len = iterator.get_next()

        return image, caption, seq_len

def main(args):

    dataset = FlickrDataLoader(args.data_path)
     # Make the dataset

    dataset._make_dataset(args.phase, args.record_path, num=args.num)

    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/shared/kgcoe-research/mil/Flickr30k')
    parser.add_argument('--train_dir', type=str, default='/shared/kgcoe-research/mil/Flickr30k/flickr30k_images/flickr30k_images')
    #parser.add_argument('--val_dir', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/val2014')
    parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/Flickr30k/flickr30k_resnet_29783.tfrecord')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--precompute', type=str, default='True')
    args = parser.parse_args()

    main(args)

