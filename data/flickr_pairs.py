import itertools
import numpy as np
import json
import pdb
path = '/shared/kgcoe-research/mil/Flickr30k/flickr30k_images/train.token'
file_names = open(path).read().split('\n')
num = len(file_names) - 1
image_name = []
captions = []
#pdb.set_trace()
for id in range(num):
    #print(id)
    split_line = file_names[id].split('\t') 
    image_name.append(split_line[0].split('#')[0])
    captions.append(split_line[1])

cap_seq = []
img_file = []
num_images = num/5 
n = 0   
for c in range(num_images):
    img_file.append(image_name[n])
    cap_seq.append(captions[c:c+5])
    n = n+5

text_file = open("/shared/kgcoe-research/mil/Flickr30k/encoder_caption.txt", "a")
text_file_1 = open("/shared/kgcoe-research/mil/Flickr30k/decoder_caption.txt", "a")
for seq in cap_seq: 
    sentence_combinations = list(itertools.permutations(cap_seq[0],2))
    for com in sentence_combinations:
        text_file.write('%s\n'%(com[0]))
        text_file_1.write('%s\n'%(com[1]))

    