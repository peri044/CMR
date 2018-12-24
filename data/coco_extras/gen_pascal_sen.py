import numpy as np
import tensorflow as tf
import pdb
import os

def main():
    root_path='/shared/kgcoe-research/mil/video_project/cvs/pascal/np'
    sen_root_path='/shared/kgcoe-research/mil/video_project/cvs/pascal/sentence'
    filenames = open(os.path.join(root_path, 'names_test.txt'), 'r').readlines()
    output = open('pascal_test_sentences.txt', 'w')
    for file in filenames:
        file=file.strip()
        image_name = file.split('/')[-1].split('.')[0]
        dir_name=file.split('/')[-2]
        text_filename=image_name+'txt'
        abs_text_filename=os.path.join(sen_root_path, dir_name, text_filename)
        sentences = open(abs_text_filename, 'r').readlines()
        for sen in sentences:
            sen=sen.strip().replace(',', '').replace('\'', '').replace('.', '').lower()
            output.write(sen+'\n')
    output.close()
        

if __name__=='__main__':
    main()