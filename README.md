# Cross-Modal Retrieval Experiments

## Training on MSCOCO
```
sh scripts/train_coco.sh
```
## Training on Flickr
```
sh scripts/train_flickr.sh
```
Checkout the command line options for details of the experiment configuration. Supported loss functions: triplet, order-violation and cosine similarity
dnn_library.py is the interface to use any other base feature extractor.

## Evaluation on MSCOCO
```
sh scripts/eval_coco.sh <path_to_checkpoint>
```
* <path_to_checkpoint> - Specify the path to trained model ckpt.
Checkout the default command line options for evaluation and modify accordingly.
## Evaluation on Flickr
```
sh scripts/eval_flickr.sh <path_to_checkpoint>
```

## Generate Data
In the data folder, you can find scripts for generating TF-records for flowers dataset.
Checkout command line arguments in the scripts for setting paths

* To generate train and text files for flowers
```
python process_flowers_6k.py
```
* To generate TF-records for flowers
```
python flowers_data_loader.py
```
`coco_data_loader.py` is base class to read COCO data. Data-readers and writers are included along with padded batching, pre-processing and data augmentation.
* To generate TF-records for MSCOCO
```
python coco_data_loader.py --num 10000
```
* Generate CNN features for FLICKR
```
python extract_image_features.py --dataset flickr --data_path /shared/kgcoe-research/mil/peri/flickr_data/ --root_path /shared/kgcoe-research/mil/Flickr30k/flickr30k_images/flickr30k_imagebackup2/ --save_path /shared/kgcoe-research/mil/peri/flickr_data/
```
* To generate TF records with precomputed resnet_v1_152 features for flickr captions
```
python extract_image_features.py --record_path /shared/kgcoe-research/mil/Flickr30k/flickr_new_train_feat.tfrecord
```
* To generate TF records with precomputed resnet_v1_152 features
```
python extract_image_features.py --dataset mscoco --root_path /shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/ --data_path /shared/kgcoe-research/mil/peri/mscoco_data/train.ids --save_path /shared/kgcoe-research/mil/peri/mscoco_data/ --caps_path /shared/kgcoe-research/mil/peri/mscoco_data/train_caps.txt
```
Args:
* `--num` : Number of examples to put in TF record. If it is not specified, entire dataset would be taken. Do not specify unless you are trying to overfit on a smaller dataset.
* `--phase` : By default, training phase is set and it picks training + some validation images of MSCOCO
More command line options related to setting the paths to the data can be found in the script `coco_data_loader.py`.
A sample in TF record is of the form (image, caption) 

# Notes
* model.py - Base model class for LSTM encoder, feature extractor, embedding layers and loss function
* dnn_library.py - Dictionary of base feature extractor networks
* Checkpoints and summaries can be found at 

```bash 
/shared/kgcoe-research/mil/peri/mscoco_data/
/shared/kgcoe-research/mil/Flickr30k
```

