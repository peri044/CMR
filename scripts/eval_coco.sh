#/bin/bash

# Define the experiment configuration
batch_size=100
num=5000
num_folds=1
base='resnet_v1_152'
checkpoint=$1
emb_dim=1024
num_units=1024
num_layers=1
dropout=0.0
word_dim=300
model='vse-att'
max_len=20  # Max caption length
stride=4 # Stride of GRU steps to be skipped for HRNE model
val_ids_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/test_filenames.txt'
val_caps_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/testall_caps.txt'
test_sample='COCO_val2014_000000060623.jpg'
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/feats/rv1_152_tf/coco_new_test.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/im_dual_tfrecords/mscoco_val.tfrecord'
record_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_dual_r152_test.tfrecord'
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/f30k_test_dual.tfrecord'
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/pascal/pascal_test.tfrecord'
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_test_scan_global_feat.tfrecord'
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_train_scan_global_feat.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/scan_data/coco_train_scan.tfrecord'
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_scan_test.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/feats/rv1_152_tf/coco_logit_and_image_test.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/feats/rv1_152_tf/coco_test_feat_7x7.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/flickr_data/flickr_image_train.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/flickr_data/flickr_image_test.tfrecord'
# record_path='/shared/kgcoe-research/mil/peri/flickr_data/flickr_test_r152_precomp.tfrecord'
# record_path='/shared/kgcoe-research/mil/Flickr30k/flickr30k_resnet_train_new.tfrecord'
measure='cosine'
root_path='shared/kgcoe-research/mil/peri/scan_data'
mode='val'
dataset='mscoco'
# vocab_file='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc'  # Vocab file for word embeddings
vocab_file='/shared/kgcoe-research/mil/cvs_cvpr18/coco/mscoco_vocab.txt'  # Vocab file for word embeddings
# vocab_file='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/flickr_vocab.txt' # Vocab file for word embeddings
# vocab_size=26735
vocab_size=11355
# vocab_size=8483

python ./eval_flickr.py --batch_size ${batch_size} \
                 --base ${base} \
                 --dataset ${dataset} \
                 --checkpoint ${checkpoint} \
                 --emb_dim ${emb_dim} \
                 --word_dim ${word_dim} \
                 --num_folds ${num_folds} \
                 --num_units ${num_units} \
                 --num_layers ${num_layers} \
                 --dropout ${dropout} \
                 --vocab_file ${vocab_file} \
                 --vocab_size ${vocab_size} \
                 --stride ${stride} \
                 --max_len ${max_len} \
                 --measure ${measure} \
                 --val_ids_path ${val_ids_path} \
                 --test_sample ${test_sample} \
                 --record_path ${record_path} \
                 --root_path ${root_path} \
                 --val_caps_path ${val_caps_path} \
                 --model ${model} \
                 --mode ${mode} \
                 --num ${num} \
                 --precompute \
                 --retrieve_image
                 # --use_abs \
                 
                 # 
                 
                 
                  
                           