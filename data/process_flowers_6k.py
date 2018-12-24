import numpy as np
import os
import argparse
import shutil
import pdb

def process_caption(caption):
	new_caption= caption.replace('.', '')
	new_caption= new_caption.replace(',', '')
	return new_caption
	
def main(args): 
    image_path=os.path.join(args.root_path, 'jpg')
    text_path=os.path.join(args.root_path, 'text_c10')
    # if os.path.isdir(args.save_path):
        # shutil.rmtree(args.save_path)
    # os.mkdir(args.save_path)
    all_data_file=open(os.path.join(args.save_path, 'flowers_6k_train.txt'), 'w')
    i=0
    train_6k_file = open(args.data_path, 'r').readlines()
    train_6k_filenames=[file.strip() for file in train_6k_file]
    j=0
    for read_name in train_6k_filenames:
        for dir in os.listdir(text_path):
            if i%10==0 and i!=0: print "Processed: {}".format(i)
            label = int(dir.split('_')[1])
            for file in os.listdir(os.path.join(text_path,dir)):
                name=file.strip().split('.')[0]
                image_name=os.path.join(image_path, name+'.jpg')
                if name+'.jpg'==read_name:
                    file_data=open(os.path.join(text_path,dir,file.strip()), 'r').readlines()
                    for caption in file_data:
                        preprocess_caption = process_caption(caption.strip())
                        all_data_file.write(image_name+'__'+preprocess_caption+'__'+ str(label)+'\n')
            i+=1
    all_data_file.close()
				
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--root_path', default='/shared/kgcoe-research/mil/video_project/cvs/flowers', help='Root path')
    parser.add_argument('--data_path', default='/shared/kgcoe-research/mil/video_project/cvs/flowers/np/onlyTrain/names_train.txt', help='Path to 6k train images')
    parser.add_argument('--save_path', default='/shared/kgcoe-research/mil/peri/flowers_data/record_data/new', help='Root path')
    args=parser.parse_args()
    main(args)