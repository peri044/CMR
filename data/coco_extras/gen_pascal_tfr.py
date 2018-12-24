import pdb
import tensorflow as tf
import numpy as np
import os 

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
        
def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])
    
def write_tfrecord(image_path, caption, category):
    image = tf.gfile.FastGFile(image_path, "rb").read()
    feature_lists = tf.train.FeatureLists(feature_list={"caption": _bytes_feature_list(caption.split(' '))})
    context = tf.train.Features(feature={"image": _bytes_feature(image), "category":_int64_feature(category)})
    sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    
    return sequence_example

def main():
    root_path='/shared/kgcoe-research/mil/video_project/cvs/pascal/image'
    image_file = open('/shared/kgcoe-research/mil/video_project/cvs/pascal/np/names_test.txt').readlines()
    images=[file.strip() for file in image_file]
    sentences_file = open('pascal_test_sentences.txt', 'r').readlines()
    sentences=[file.strip() for file in sentences_file]
    record_path='/shared/kgcoe-research/mil/cvs_cvpr18/'
    tfrecord_writer=tf.python_io.TFRecordWriter(os.path.join(record_path, 'pascal_test.tfrecord'))
    for i in range(len(images)):
        if i%100==0: print "Processed: {}".format(i)
        for j in range(i*5, i*5 + 5):
            sequence_example= write_tfrecord(images[i], sentences[j], i/10)
            tfrecord_writer.write(sequence_example.SerializeToString())

    tfrecord_writer.close()
    print "Done writing records"
    
if __name__=='__main__':
    main()