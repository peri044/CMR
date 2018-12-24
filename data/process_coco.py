import numpy as np
import os
import pdb
import argparse
import json

def main(args):
      
    instances = json.load(open(args.train_path, 'r'))
    instances_val = json.load(open(args.val_path, 'r'))
    images = instances['images']
    images_val = instances_val['images']
    image_id_to_name={}
    for i, sample in enumerate(images):
        image_id_to_name[sample['id']] = sample['file_name']
        
    for i, sample in enumerate(images_val):
        image_id_to_name[sample['id']] = sample['file_name']
    print "Done writing dict"
        
    file = open('test_filenames.txt', 'w')
    input_file = open('/shared/kgcoe-research/mil/peri/scan_data/data/coco_precomp/testall_ids.txt', 'r').readlines()

    for i, ele in enumerate(input_file):
        ele = int(ele.strip())
        if i%5==0:
            file.write(image_id_to_name[ele]+'\n')
    file.close()
    print "Done writing filenames"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/annotations/instances_train2014.json')
    parser.add_argument('--val_path', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/annotations/instances_val2014.json')
    args=parser.parse_args()
    main(args)