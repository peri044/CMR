#/bin/bash

# Define the experiment configuration
batch_size=128
num_epochs=30
base='resnet_v1_152'  # Base architecture of CNN feature extractor
cnn_weights='/shared/kgcoe-research/mil/peri/tf_checkpoints/resnet_v1_152.ckpt'  # CNN pre-trained checkpoint
lstm_weights='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/best_bleu/translate.ckpt-35000' # LSTM pre-trained checkpoint
checkpoint='/shared/kgcoe-research/mil/peri/flickr_data/c_vse_order_e1024_w300_L1_h1024_2018-10-16_19_16/model.ckpt-4000' # CMR model checkpoint for finetuning mode
emb_dim=1024  # CVS dimension
max_len=20  # Max caption length
stride=4 # Stride of GRU steps to be skipped for HRNE model
lr=0.0002
decay_steps=15000
decay_factor=0.1
save_steps=2000 # Step interval for checkpoint saving
num_units=1024 # GRU hidden dimension
num_layers=1 # Number of layers in GRU network
dropout=0.0 #Dropout for GRU network 
margin=0.2 # Margin for pairwise loss
word_dim=300 # Word dimension for GRU encoder
clip_grad_norm=2.0 # Gradient clipping norm value
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/im_dual_tfrecords/mscoco_train.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/feats/rv1_152_tf/coco_logit_and_image_train.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/feats/rv1_152_tf/coco_train_feat_7x7.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/peri/mscoco_data/feats/rv1_152_tf/coco_new_train.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/peri/flickr_data/flickr_train_r152_precomp.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/Flickr30k/flickr30k_resnet_train_new.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/peri/flickr_data/flickr_image_train.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_train_scan_global_feat.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/coco/coco_dual_r152_train.tfrecord' # TFRecord path to read from
# record_path='/shared/kgcoe-research/mil/cvs_cvpr18/pascal/pascal_train.tfrecord' # TFRecord path to read from
record_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/f30k_train_dual.tfrecord' # TFRecord path to read from
model='vse-att'
mode='train'
measure='cosine' # Type of loss
exp_path='/shared/kgcoe-research/mil/cvs_cvpr18/flickr_exp/'
optimizer='adam'
dataset='mscoco'
mine_n_hard=1
lambda_1=9.0
lambda_2=6.0
# vocab_file='/shared/kgcoe-research/mil/cvs_cvpr18/coco/mscoco_vocab.txt'  # Vocab file for word embeddings
vocab_file='/shared/kgcoe-research/mil/cvs_cvpr18/flickr/flickr_vocab.txt'  # Vocab file for word embeddings
# vocab_file='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc'  # Vocab file for word embeddings
vocab_size=8483
# vocab_size=11355
# vocab_size=26375

python ./train_crossmodal.py --batch_size ${batch_size} \
                           --num_epochs ${num_epochs} \
                           --save_steps ${save_steps} \
                           --base ${base} \
                           --cnn_weights ${cnn_weights} \
                           --lstm_weights ${lstm_weights} \
                           --emb_dim ${emb_dim} \
                           --word_dim ${word_dim} \
                           --num_units ${num_units} \
                           --stride ${stride} \
                           --num_layers ${num_layers} \
                           --lambda_1 ${lambda_1} \
                           --lambda_2 ${lambda_2} \
                           --lr ${lr} \
                           --dropout ${dropout} \
                           --vocab_file ${vocab_file} \
                           --vocab_size ${vocab_size} \
                           --margin ${margin} \
                           --measure ${measure} \
                           --decay_steps ${decay_steps} \
                           --decay_factor ${decay_factor} \
                           --clip_grad_norm ${clip_grad_norm} \
                           --record_path ${record_path} \
                           --checkpoint ${checkpoint} \
                           --dataset ${dataset} \
                           --optimizer ${optimizer} \
                           --exp_path ${exp_path} \
                           --model ${model} \
                           --mine_n_hard ${mine_n_hard} \
                           --no_pretrain_lstm \
                           --no_train_cnn \
                           --precompute
                           # --use_abs \
                           # --use_random_crop \
                           # --mode ${mode}
						   # --max_len ${max_len} \
                           # --finetune_with_cnn \
                           # --no_load_cnn
                           # --train_only_emb
                           
                           
                           
                           
                           
                           
                           
                          
                           
