"""
Data writer, loader for  flowers dataset
"""
import tensorflow as tf
import numpy as np
import os
import argparse
import pdb

class FlowersDataLoader(object):
	"""
	Data loader and writer object for flowers dataset
	"""
	def __init__(self, path=None):
		self.data_path=path
		
	def _int64_feature(self, value):
		"""Wrapper for inserting an int64 Feature into a SequenceExample proto."""
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
		
	def _bytes_feature(self, value):
		"""Wrapper for inserting a bytes Feature into a SequenceExample proto."""
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
		
	def _bytes_feature_list(self, values):
		"""Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
		return tf.train.FeatureList(feature=[self._bytes_feature(v) for v in values])

	def _make_single_example(self, record):
		"""
		Make a single example in a TF record
		"""
		image_path=record.split('__')[0]
		image = tf.gfile.FastGFile(image_path, "rb").read()
		caption=record.split('__')[1]
		caption_list = caption.split(' ')
		label=record.split('__')[2]
		feature_lists = tf.train.FeatureLists(feature_list={"caption": self._bytes_feature_list(caption_list)})
				   
		context = tf.train.Features(feature={
						"label": self._int64_feature(int(label)),
						"image": self._bytes_feature(image)})
										
		sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

		return sequence_example
		
	def _make_dataset(self, record_path, num=None):
		"""
		Write the whole dataset to a TF record.
		"""
		data=open(self.data_path, 'r').readlines()
		tfrecord_writer = tf.python_io.TFRecordWriter(record_path)
		count=0
		if num is not None:
			data = data[:num]
			
		for record in data:
			if count%100==0 and count!=0: print "Generated: {}".format(count)
			example = self._make_single_example(record.strip())
			tfrecord_writer.write(example.SerializeToString())
			count+=1
			
		print "Done generating TF records"
		
	def _inception_preprocess(self, image):
		
		"""
		Pre-processing for inception. Convert the range to [-1, 1]
		"""
		return (2.0/255)*image -1.
		
	def _parse_single_example(self, example_proto):
		context, sequence = tf.parse_single_sequence_example(
										example_proto,
										context_features={
										  "image": tf.FixedLenFeature([], dtype=tf.string),
										  "label": tf.FixedLenFeature([], dtype=tf.int64),
										},
										sequence_features={
										  "caption": tf.FixedLenSequenceFeature([], dtype=tf.string)
										})
		image = tf.image.decode_jpeg(context["image"], channels=3)
		image_rs = tf.image.resize_images(image, size=[224, 224])
		
		# Convert the range to [-1, 1]
		preprocess_image = self._inception_preprocess(image_rs)
		caption = tf.cast(sequence["caption"], tf.string)[:50] # max_len allowed is 50
		label = tf.cast(context["label"], tf.int32)
		
		return preprocess_image, caption, label, tf.size(caption)

	def _read_data(self, record_path, batch_size, num_epochs=10):
		dataset = tf.data.TFRecordDataset(record_path)
		dataset = dataset.shuffle(buffer_size=10000)
		dataset = dataset.map(map_func=self._parse_single_example, num_parallel_calls=4)
		# Pads the caption to the max caption size in a batch
		dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, 
									   padded_shapes=(
											[224, 224, 3],  # img
											tf.TensorShape([None]),  # caption
											tf.TensorShape([]),  # label
											tf.TensorShape([])),# src_len
									   padding_values=(0.,  # src
													  '</s>',  # caption_pad
													   0,  # label
													   0))) # Seq len
		
		dataset = dataset.prefetch(buffer_size=batch_size)
		dataset = dataset.repeat(num_epochs)
		iterator = dataset.make_one_shot_iterator()
		image, caption, label, seq_len = iterator.get_next()

		return image, caption, label, seq_len
		
	
def main(args):
	dataset = FlowersDataLoader(args.data_path)
	dataset._make_dataset(args.record_path, num=args.num)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--record_path', default='/shared/kgcoe-research/mil/peri/flowers_data/flowers_train.tfrecord', help='TF record path')
	parser.add_argument('--data_path', default='/shared/kgcoe-research/mil/peri/flowers_data/record_data/flowers_all_data.txt', help='TF record path')
	parser.add_argument('--num', type=int, default=None, help='Number of samples to write')
	args=parser.parse_args()
	main(args)
				