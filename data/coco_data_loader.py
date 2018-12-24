import tensorflow as tf
import numpy as np
import os
from preprocessing import preprocessing_factory
import argparse
import pdb

class CocoDataLoader(object):
    """
    Data loader and writer object for MSCOCO dataset
    """
    def __init__(self, path=None, precompute=False, use_random_crop=False, max_len=None, model='vse'):
        self.data_path=path
        self.precompute=precompute
        self.use_random_crop=use_random_crop
        self.max_len=max_len
        self.model=model
        
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
        
        processed_caption = caption.replace(',', '').replace('\'','')
        processed_caption = processed_caption.replace('.', '')
        processed_caption = processed_caption.lower().strip()
        return processed_caption
        
    def _make_single_example(self, image_path, caption, precompute=False):
        """
        Make a single example in a TF record
        """
        if not precompute:
            image = tf.gfile.FastGFile(image_path, "rb").read()
            caption=self._process_caption(caption)
            caption_list = caption.split(' ')
            feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
                       
            context = tf.train.Features(feature={
                            "image": self._bytes_feature(image)})
                            
        else:
            caption=self._process_caption(caption).strip()
            caption_list = caption.split(' ')
            feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
            context = tf.train.Features(feature={"image": self._bytes_feature(image_path.tostring())})
                                        
        sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

        return sequence_example
    
    def _make_concept_example(self, image_path, concept, caption, precompute=False):
        """
        Make a single example in a TF record
        """
        if not precompute:
            image = tf.gfile.FastGFile(image_path, "rb").read()
            caption=self._process_caption(caption)
            caption_list = caption.split(' ')
            feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
                       
            context = tf.train.Features(feature={
                            "image": self._bytes_feature(image),
                            "concept": self._bytes_feature(concept.tostring())})
                            
        else:
            caption=self._process_caption(caption)
            caption_list = caption.split(' ')
            feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
            context = tf.train.Features(feature={"image": self._bytes_feature(image_path.tostring()),
                                                 "concept": self._bytes_feature(concept.tostring())})
                                        
        sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

        return sequence_example
    
    def _augment_image(self, image_path):
        """
        Performs 10-cropping on a 256x256 image
        """
        orig_image = io.imread(image_path)
        if len(orig_image.shape)!=3: orig_image = gray2rgb(orig_image) 
        resized_image = resize(orig_image, [256, 256])
        augmented_im = np.zeros((10, 224, 224, 3))
        augmented_im[0, :] = resized_image[0:224, 0:224, :]
        augmented_im[1, :] = resized_image[0:224, 32:256, :]
        augmented_im[2, :] = resized_image[32:256, 0:224, :]
        augmented_im[3, :] = resized_image[32:256, 32:256, :]
        augmented_im[4, :] = resized_image[16:240, 16:240, :]
        augmented_im[5, :] = augmented_im[0, :][::-1, :, :]
        augmented_im[6, :] = augmented_im[1, :][::-1, :, :]
        augmented_im[7, :] = augmented_im[2, :][::-1, :, :]
        augmented_im[8, :] = augmented_im[3, :][::-1, :, :]
        augmented_im[9, :] = augmented_im[4, :][::-1, :, :]
       
        return augmented_im
        
    def _make_dataset(self, phase, record_path, num=None):
        """
        Write the whole dataset to a TF record.
        """
        train_ids = open(os.path.join(args.data_path, phase+'.ids')).readlines()
        train_caps = open(os.path.join(args.data_path, phase+'_caps.txt')).readlines()
        tfrecord_writer = tf.python_io.TFRecordWriter(record_path)

        if num is None:
            num=len(train_ids)
        count=0

        for im_idx in range(num):
            if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num*5)
            if train_ids[im_idx].strip().find('train2014') !=-1:
                image = os.path.join(args.train_dir, train_ids[im_idx].strip())
            elif train_ids[im_idx].strip().find('val2014') !=-1:
                image = os.path.join(args.val_dir, train_ids[im_idx].strip())
            else:
                raise ValueError("Invalid Image")
                
            for cap_idx in range(im_idx*5, im_idx*5 +5):
                example = self._make_single_example(image, train_caps[cap_idx].strip())
                tfrecord_writer.write(example.SerializeToString())
                count+=1
            
        print "Done generating TF records"
        
    def _make_concepts_dataset(self, phase, record_path, concept_path, feature_path, num=None, precompute=False):
        """
        Write the whole dataset to a TF record.
        """
        train_ids = open(os.path.join(args.data_path, phase+'.ids')).readlines()
        train_caps = open(os.path.join(args.data_path, phase+'_caps.txt')).readlines()
        concepts = np.load(concept_path).astype(np.float32)
        tfrecord_writer = tf.python_io.TFRecordWriter(record_path)
            
        if num is None:
            num=len(train_ids)
        count=0
        if not precompute:
            for im_idx in range(num):
                if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num*5)
                if train_ids[im_idx].strip().find('train2014') !=-1:
                    image = os.path.join(args.train_dir, train_ids[im_idx].strip())
                elif train_ids[im_idx].strip().find('val2014') !=-1:
                    image = os.path.join(args.val_dir, train_ids[im_idx].strip())
                else:
                    raise ValueError("Invalid Image")
                    
                for cap_idx in range(im_idx*5, im_idx*5 +5):
                    example = self._make_concept_example(image, concepts[im_idx], train_caps[cap_idx].strip())
                    tfrecord_writer.write(example.SerializeToString())
                    count+=1
        else:
            features = np.load(feature_path).astype(np.float32)
            for im_idx in range(num):
                if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num*5)
                for cap_idx in range(im_idx*5, im_idx*5 +5):
                    example = self._make_concept_example(features[im_idx], concepts[im_idx], train_caps[cap_idx].strip())
                    tfrecord_writer.write(example.SerializeToString())
                    count+=1
            
        print "Done generating TF records"
        
    def _precomputed_dataset(self, phase, record_path, feature_path, captions_path, num=None):
        """
        Write the whole dataset to a TF record.
        """
        train_caps = open(args.captions_path, 'r').readlines()
        train_img_features = np.load(args.feature_path).astype(np.float32) #[0:25000:5]
        tfrecord_writer = tf.python_io.TFRecordWriter(record_path)
        if num is None:
            num=len(train_img_features)
        count=0
        global_feat = np.load('/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_train_r152_precomp.npy').astype(np.float32)
        # for im_idx in range(num):
            # if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num*5)                
            # for cap_idx in range(im_idx*5, im_idx*5 +5):
                # example = self._make_single_example(train_img_features[im_idx], train_caps[cap_idx].strip(), precompute=True)
                # tfrecord_writer.write(example.SerializeToString())
                # count+=1
        # pdb.set_trace()        
        for im_idx in range(num):
            if count%5000==0 and count!=0: print "Generated: {}/{}".format(count, num*5)    
            global_feat_tile = np.tile(np.expand_dims(global_feat[im_idx], 0), [36, 1])
            for cap_idx in range(im_idx, im_idx +5):
                concat_feat = np.concatenate([train_img_features[im_idx], global_feat_tile], axis=1)
                example = self._make_single_example(concat_feat, train_caps[cap_idx].strip(), precompute=True)
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
        image_preprocessing_fn = preprocessing_factory.get_preprocessing('resnet_v1_152', is_training=self.use_random_crop)
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
            if self.model=='vse':
                image_shape=[2048]
            elif self.model=='vse-att':
                image_shape=[2048]
            elif self.model=='vse-vgg':
                image_shape=[4096]
            elif self.model=='bi':
                image_shape=[36, 2048]
            elif self.model=='bi-conv':
                image_shape=[36, 4096]
            image = tf.reshape(image, image_shape)
        max_len=50
        if self.max_len:
            max_len=self.max_len
        caption = tf.cast(sequence["caption"], tf.string)[:max_len] # max_len allowed is 50
        reverse_caption = tf.reverse(caption, axis=[0])
        return image, caption, reverse_caption, tf.size(caption)
        
    def _parse_ism_example(self, example_proto):
        context, sequence = tf.parse_single_sequence_example(
                                        example_proto,
                                        context_features={
                                          "image": tf.FixedLenFeature([], dtype=tf.string),
                                          "concept": tf.FixedLenFeature([], dtype=tf.string),
                                        },
                                        sequence_features={
                                          "caption": tf.FixedLenSequenceFeature([], dtype=tf.string)
                                        })
        if not self.precompute:
            image = tf.image.decode_jpeg(context["image"], channels=3)
            image = self._vgg_preprocess(image)
        else:
            tfrecord_writer = tf.python_io.TFRecordWriter(record_path)
            image = tf.decode_raw(context['image'], out_type=tf.float32)
        concept = tf.decode_raw(context['concept'], out_type=tf.float32)

        max_len=50
        if self.max_len:
            max_len=self.max_len
        caption = tf.cast(sequence["caption"], tf.string)[:max_len] # max_len allowed is 50
        reverse_caption = tf.reverse(caption, axis=[0])
        return image, concept, caption, reverse_caption, tf.size(caption)

    def _read_data(self, record_path, batch_size, phase='train', num_epochs=10):
        dataset = tf.data.TFRecordDataset(record_path)
        if phase !='val':
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(map_func=self._parse_single_example, num_parallel_calls=4)
        # Pads the caption to the max caption size in a batch
        if not self.precompute:
            image_shape=[224, 224, 3]
        else:
            if self.model=='vse':
                image_shape=[2048]
            elif self.model=='vse-att':
                image_shape=[2048]
            elif self.model=='vse-vgg':
                image_shape=[4096]
            elif self.model=='bi':
                image_shape=[36, 2048]
            elif self.model=='bi-conv':
                image_shape=[36, 4096]

        dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, 
                                       padded_shapes=(
                                            image_shape,  # img
                                            tf.TensorShape([self.max_len]),  # caption
                                            tf.TensorShape([self.max_len]),  # caption
                                            tf.TensorShape([])),# src_len
                                       padding_values=(0.,  # src 
                                                      '</s>',  # caption_pad
                                                      '</s>',  # caption_pad
                                                       0))) # Seq len
        
        dataset = dataset.prefetch(buffer_size=batch_size)
        dataset = dataset.repeat(num_epochs)
        
        iterator = dataset.make_one_shot_iterator()
        image, caption, reverse_caption, seq_len = iterator.get_next()
        return image, caption, reverse_caption, seq_len
        
    def _read_ism_data(self, record_path, batch_size, phase='train', num_epochs=10):
        dataset = tf.data.TFRecordDataset(record_path)
        if phase !='val':
            dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.map(map_func=self._parse_ism_example, num_parallel_calls=4)
        # Pads the caption to the max caption size in a batch
        if not self.precompute:
            image_shape=[224, 224, 3]
        else:
            image_shape=[2048] #4096

        dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, 
                                       padded_shapes=(
                                            image_shape,  # img
                                            [256],
                                            tf.TensorShape([self.max_len]),  # caption
                                            tf.TensorShape([self.max_len]),  # caption
                                            tf.TensorShape([])),# src_len
                                       padding_values=(0.,
                                                       0., # src 
                                                      '</s>',  # caption_pad
                                                      '</s>',  # caption_pad
                                                       0))) # Seq len
        
        dataset = dataset.prefetch(buffer_size=batch_size)
        dataset = dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        image, concept, caption, reverse_caption, seq_len = iterator.get_next()
        
        return image, concept, caption, reverse_caption, seq_len

def main(args):

    dataset = CocoDataLoader(args.data_path)
    # Make the dataset
    if not args.precompute and not args.make_concepts:
        dataset._make_dataset(args.phase, args.record_path, num=args.num)
    elif args.precompute:
        dataset._precomputed_dataset(args.phase, args.record_path, args.feature_path, args.captions_path, num=args.num)
    elif args.make_concepts:
        dataset._make_concepts_dataset(args.phase, args.record_path, args.concepts_path, args.feature_path, num=args.num)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/shared/kgcoe-research/mil/peri/scan_data/data/coco_precomp')
    parser.add_argument('--train_dir', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/train2014')
    parser.add_argument('--val_dir', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/val2014')
    parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_train_scan_global_feat.tfrecord')
    parser.add_argument('--concepts_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/feats/rv1_152_tf/logit_vectors_train.npy')
    parser.add_argument('--feature_path', type=str, default='/shared/kgcoe-research/mil/peri/scan_data/data/coco_precomp/train_ims.npy')
    parser.add_argument('--captions_path', type=str, default='/shared/kgcoe-research/mil/peri/scan_data/data/coco_precomp/train_caps.txt')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--precompute', action='store_true', help='Flag to build using precomputed CNN features')
    parser.add_argument('--make_concepts', action='store_true', help='Flag to write concepts')
    parser.add_argument('--num', type=int, default=None)
    args = parser.parse_args()
    main(args)