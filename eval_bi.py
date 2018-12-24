import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import argparse
from model import *
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from preprocessing import preprocessing_factory
from data.coco_data_loader import *
import pdb
import time
from attention import *
import sys

# tf.enable_eager_execution()
    
def order_sim(images, captions):
    """
    Computes the order similarity between images and captions
    """
    clip_diff = np.maximum(captions - images, 0)
    sqr_clip_diff = np.square(clip_diff)
    sim = np.sqrt(np.sum(sqr_clip_diff, axis=-1))
    sim = -np.transpose(sim)
    return sim

def shard_xattn_t2i_gpu(images, captions, seq_len, params, shard_size=100):
    """
    Computer pairwise t2i image-caption distance with sharding  // Very slow in Tensorflow
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1
    
    d = np.zeros((len(images), len(captions)))

    images_tensor = tf.placeholder(shape=[shard_size, images.shape[1], 1024], dtype=tf.float32)
    text_tensor = tf.placeholder(shape=[shard_size, captions.shape[1], 1024], dtype=tf.float32)
    sl_tensor = tf.placeholder(shape=[shard_size], dtype=tf.int32)
    
    # Define the attention
    sim = t2i_attention(images_tensor, text_tensor, sl_tensor, params)
    
    with tf.Session() as sess:
        for i in range(n_im_shard):
            im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
                # print '(%d , %d)\r' % (i, j),
                cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
                im = images[im_start:im_end]
                s = captions[cap_start:cap_end]
                s_l = seq_len[cap_start:cap_end]
                
                sim_values = sess.run(sim, feed_dict={images_tensor: im,
                                                    text_tensor: s,
                                                    sl_tensor: s_l})
                                                    
                d[im_start:im_end, cap_start:cap_end] = sim_values

        return d
        
def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum

    return p
    
def cosine_sim(query, ref, axis=2):
    
    numerator = np.sum(np.multiply(query, ref), axis=axis)
    query_norm = np.linalg.norm(query, axis=axis)
    ref_norm = np.linalg.norm(ref, axis=axis)
    
    return np.divide(numerator, np.maximum(query_norm*ref_norm, 1e-8))
    
def compute_attention_cpu(query, context, params):
    """
    query: (B x n_query x d)
    context: (B x n_context x d)
    """
    batch_size_q, num_words_q = query.shape[0], query.shape[1]
    batch_size_c, num_regions_c = context.shape[0], context.shape[1]
    
    queryT = np.transpose(query, [0, 2, 1])
    attn = np.matmul(context, queryT) # B x n_context x n_query

    # leaky l2 norm
    clip_attn = tf.nn.leaky_relu(attn, alpha=0.1) # B x n_context x n_query
    clip_attn = np.maximum(0.1*attn, attn) # Leaky relu
    l2_norm = np.expand_dims(np.linalg.norm(clip_attn, axis=2), 2)
    norm_attn = np.divide(clip_attn, l2_norm) # B x n_context x n_query

    attn_transpose = np.transpose(norm_attn, [0, 2, 1]) # B x n_query x n_context
    soft_attn = softmax(attn_transpose*params.lambda_1, axis=2) # B x n_query x n_context
    soft_attn_transpose = np.transpose(soft_attn, [0, 2, 1]) # B x n_context x n_query
    
    context_transpose = np.transpose(context, [0, 2, 1]) # B x d x n_context
    weighted_attn = np.matmul(context_transpose, soft_attn_transpose) # B x d x n_query
    weighted_attn_context = np.transpose(weighted_attn, [0, 2, 1]) # B x n_query x d
    
    return weighted_attn_context, soft_attn_transpose
    
        
def t2i_attention_cpu(image_embeddings, text_embeddings, seq_len, params):
    """
    Text-to-Image Attention
    """
    n_image = image_embeddings.shape[0]
    n_caption = text_embeddings.shape[0]

    similarities=[]
    for i in range(n_caption):
        n_word = seq_len[i]
        cap_i = np.expand_dims(text_embeddings[i, :n_word, :], 0)
        tiled_cap_i = np.tile(cap_i, [n_image, 1, 1])
        
        weighted_attn_context, attn = compute_attention_cpu(tiled_cap_i, image_embeddings, params)
        # row_sim --> B x n_word
        row_sim = cosine_sim(tiled_cap_i, weighted_attn_context) # B x n_word x d , B x n_word x d
        row_sim = np.mean(row_sim, axis=1)
        similarities.append(row_sim)

    sim_matrix = np.stack(similarities, axis=1)
    
    return sim_matrix

def shard_xattn_t2i_cpu(images, captions, seq_len, params, shard_size=100):
    """
    Computer pairwise t2i image-caption distance with sharding  // Very slow in Tensorflow
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1
    
    d = np.zeros((len(images), len(captions)))

    sim_row_vals = []
    sim_col_vals = []
    with tf.Session() as sess:
        for i in range(n_im_shard):
            if i%500==0 and i!=0: print "Computed: {} images".format(i)
            im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
            for j in range(n_cap_shard):
                cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
                im = images[im_start:im_end]
                s = captions[cap_start:cap_end]
                s_l = seq_len[cap_start:cap_end]
                sim = t2i_attention_cpu(im, s, s_l, params)
                d[im_start:im_end, cap_start:cap_end] = sim

        return d    
    
def t2i(sim):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    

    npts = sim.shape[0] # sim --> n_images x n_captions
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sim = sim.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sim[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)
    
def i2t(sim):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    npts = sim.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sim[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)
	
def eval(args):
    # pdb.set_trace()
    if args.dataset=='mscoco':
        dataset = CocoDataLoader(precompute=args.precompute, max_len=args.max_len, model=args.model)
        image, caption, reverse_caption, seq_len = dataset._read_data(args.record_path, args.batch_size, phase=args.mode, num_epochs=args.num_epochs)
    elif args.dataset=='coco-ism':
        dataset = CocoDataLoader(precompute=args.precompute, max_len=args.max_len)
        image, logit_feat, caption, reverse_caption, seq_len = dataset._read_ism_data(args.record_path, args.batch_size, phase=args.mode, num_epochs=args.num_epochs)
    else:
        raise ValueError("Invalid dataset !!")

    # Call the CMR model
    model=CMR(base=args.base, embedding_dim=args.emb_dim, word_dim=args.word_dim, vocab_file=args.vocab_file, vocab_size=args.vocab_size)
    if args.model=='vse':
        image_embeddings_t, text_embeddings_t = model.build_vse_model(image, reverse_caption, seq_len, args, is_training=False)
    elif args.model=='hrne':
        image_embeddings_t, text_embeddings_t = model.build_hrne_model(image, reverse_caption, seq_len, args, is_training=False)
    elif args.model=='ism':
        image_embeddings_t, text_embeddings_t = model.build_ism_model(image, logit_feat, caption, seq_len, args, is_training=False)
    elif args.model=='feat':
        image_embeddings_t, text_embeddings_t = model.build_featmap_model(image, caption, seq_len, args, is_training=False)
    elif args.model=='bi':
        image_embeddings_t, text_embeddings_t, _ = model.build_bidirectional_model(image, caption, seq_len, args, is_training=False)
    elif args.model=='bi-conv':
        image_embeddings_t, text_embeddings_t, _ = model.build_bidirectional_model(image, caption, seq_len, args, is_training=False)
    else:
        raise ValueError("Invalid Model !!")

    max_words = 30
    # Define the arrays for embedding vectors
    image_embeddings_val=np.zeros((args.num, 36, args.emb_dim))
    text_embeddings_val=np.zeros((args.num, max_words, args.emb_dim))
    sequence_lengths=np.zeros((args.num), dtype=np.int32)

    print "Total number of validation samples: {}".format(args.num)

    # Define a saver
    saver=tf.train.Saver()       
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.tables_initializer())
        saver.restore(sess, args.checkpoint)
        start_time = time.time()
        for i in range(0, args.num, args.batch_size):
            if i%5000==0: print "Processed: {}".format(i)
            try:
                ie, te, s_l = sess.run([image_embeddings_t, text_embeddings_t, seq_len])
                # pdb.set_trace()
                image_embeddings_val[i:i+args.batch_size, :, :] = np.squeeze(ie)
                n_words = te.shape[1]
                text_embeddings_val[i:i+args.batch_size, :n_words, :] = np.squeeze(te)
                sequence_lengths[i:i+args.batch_size] = np.squeeze(s_l)
            except tf.errors.OutOfRangeError:
                break

    r1, r5, r10 = 0., 0., 0.
    # Average over 5 folds
    results=[]
    for fold in range(args.num_folds):
        # Get the similarity matrix
        sim = shard_xattn_t2i_gpu(image_embeddings_val[5000*fold: 5000*fold + 5000:5], text_embeddings_val[5000*fold: 5000*fold + 5000], sequence_lengths[5000*fold: 5000*fold + 5000], args)
        pdb.set_trace()
        print 'Fold: {}'.format(fold)
        ri, ri0 = i2t(sim)
        print "Image to Text: "
        print "R@1: {} R@5: {} R@10 : {} Med: {} Mean: {}".format(ri[0], ri[1], ri[2], ri[3], ri[4])
        
        rt, rt0 = t2i(sim)
        print "Text to Image: "
        print "R@1: {} R@5: {} R@10 : {} Med: {} Mean: {}".format(rt[0], rt[1], rt[2], rt[3], rt[4])
        print '--------------------------------------------'
        results += [list(ri) + list(rt)]

    print("Mean metrics: ")
    mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
          mean_metrics[:5])
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
          mean_metrics[5:10])
        
	

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
	parser.add_argument('--dataset', type=str, default='mscoco', help="Type of dataset")
	parser.add_argument('--num', type=int, default=None, help="Number of examples to be evaluated")
	parser.add_argument('--stride', type=int, default=4, help="Value of stride in HRNE")
	parser.add_argument('--max_len', type=int, default=None, help="Value of maximum caption length")
	parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs to be evaluated")
	parser.add_argument('--emb_dim', type=int, default=1024, help="Batch size")
	parser.add_argument('--word_dim', type=int, default=300, help="Word Embedding dimension")
	parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
	parser.add_argument('--lambda_1', type=float, default=9., help="dropout")
	parser.add_argument('--num_folds', type=int, default=5, help="Number of folds for Cross validation")
	parser.add_argument('--margin', type=float, default=0.05, help="Margin for sim loss")
	parser.add_argument('--precompute', action='store_true', help="Flag to use precomputed CNN features")
	parser.add_argument('--num_units', type=int, default=1024, help="Number of hidden RNN units")
	parser.add_argument('--vocab_size', type=int, default=26375, help="Number of hidden RNN units")
	parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in RNN network")
	parser.add_argument('--vocab_file', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc', help="Val file")
	parser.add_argument('--val_ids_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/test.ids', help="Test IDs path")
	parser.add_argument('--val_caps_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/test_caps.txt', help="Test captions path")
	parser.add_argument('--measure', type=str, default='cosine', help="Type of measure")
	parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/coco_val_precompute.tfrecord', help="Path to val tfrecord")
	parser.add_argument('--root_path', type=str, default='/shared/kgcoe-research/mil/video_project/mscoco_skipthoughts/images/val2014', help="Experiment dir")
	parser.add_argument('--checkpoint', type=str, default='/shared/kgcoe-research/mil/peri/flowers_data/checkpoints_CMR_finetune_2018-08-11_16_45/model.ckpt-28000', help="LSTM checkpoint")
	parser.add_argument('--model', type=str, default='vse', help="Name of the model")
	parser.add_argument('--mode', type=str, default='val', help="Training or validation")
	parser.add_argument('--base', type=str, default='resnet_v2_152', help="Base architecture")
	parser.add_argument('--use_abs', action='store_true', help="use_absolute values for embeddings")
	parser.add_argument('--finetune_with_cnn', action='store_true', help="use_absolute values for embeddings")
	args = parser.parse_args()
	print '--------------------------------'
	for key, value in vars(args).items():
		print key, ' : ', value
	print '--------------------------------'
	eval(args)