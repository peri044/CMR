import tensorflow as tf
import numpy as np
import pdb

def cosine_sim(query, ref, axis=2):
    
    numerator = tf.reduce_sum(tf.multiply(query, ref), axis=axis)
    query_norm = tf.norm(query, axis=axis)
    ref_norm = tf.norm(ref, axis=axis)
    
    return numerator/tf.maximum(query_norm*ref_norm, 1e-8)

def compute_attention(query, context, params):
    """
    query: (B x n_query x d)
    context: (B x n_context x d)
    """
    batch_size_q, num_words_q = query.shape.as_list()[0], query.shape.as_list()[1]
    batch_size_c, num_regions_c = context.shape.as_list()[0], context.shape.as_list()[1]

    attn = tf.matmul(context, query, transpose_b=True) # B x n_context x n_query
    
    # clipped leaky l2 norm
    clip_attn = tf.nn.leaky_relu(attn, alpha=0.1) # B x n_context x n_query
    norm_attn = tf.nn.l2_normalize(clip_attn, axis=2) # B x n_context x n_query
    attn_transpose = tf.transpose(norm_attn, [0, 2, 1]) # B x n_query x n_context
    soft_attn = tf.nn.softmax(attn_transpose*params.lambda_1) 
    soft_attn_transpose = tf.transpose(soft_attn, [0, 2, 1]) # B x n_context x n_query
    
    context_transpose = tf.transpose(context, [0, 2, 1]) # B x d x n_context
    weighted_attn = tf.matmul(context_transpose, soft_attn_transpose) # B x d x n_query
    weighted_attn_context = tf.transpose(weighted_attn, [0, 2, 1]) # B x n_query x d
    
    return weighted_attn_context, soft_attn_transpose
    
def t2i_attention(image_embeddings, text_embeddings, seq_len, params):
    """
    Text-to-Image Attention
    """
    n_image = image_embeddings.shape.as_list()[0]
    n_caption = text_embeddings.shape.as_list()[0]

    similarities=[]
    for i in range(n_caption):
        n_word = seq_len[i]
        cap_i = tf.expand_dims(text_embeddings[i, :n_word, :], 0)
        tiled_cap_i = tf.tile(cap_i, [n_image, 1, 1])
        
        weighted_attn_context, sim_matrix = compute_attention(tiled_cap_i, image_embeddings, params) # Weighted image vector
        # row_sim --> B x n_word
        row_sim = cosine_sim(tiled_cap_i, weighted_attn_context) # B x n_word x d , B x n_word x d
        row_sim = tf.reduce_mean(row_sim, axis=1)
        similarities.append(row_sim)

    sim_matrix = tf.stack(similarities, axis=1)

    return sim_matrix
        
def i2t_attention(image_embeddings, text_embeddings, seq_len, params):
    """
    Image-to-Text Attention
    """
    n_image = image_embeddings.shape.as_list()[0]
    n_caption = text_embeddings.shape.as_list()[0]

    similarities=[]
    for i in range(n_caption):
        n_word = seq_len[i]
        cap_i = tf.expand_dims(text_embeddings[i, :n_word, :], 0)
        tiled_cap_i = tf.tile(cap_i, [n_image, 1, 1])
        
        weighted_attn_context, sim_matrix = compute_attention(image_embeddings, tiled_cap_i, params) # Weighted sentence vector
        # row_sim --> B x n_word
        row_sim = cosine_sim(image_embeddings, weighted_attn_context, axis=2) # B x n_word x d , B x n_word x d
        row_sim = tf.reduce_mean(row_sim, axis=1)
        similarities.append(row_sim)

    sim_matrix = tf.stack(similarities, axis=1)

    return sim_matrix     

def aligned_attention(X1, X2, emb_dim, is_training=True, reuse=False, skip=True):
    with tf.variable_scope('image_embedding_align_att', reuse=False):
        image_W = tf.get_variable('att_img', [emb_dim, emb_dim], trainable=is_training)
        sent_W = tf.get_variable('att_sent', [emb_dim, emb_dim], trainable=is_training)
        att_W = tf.get_variable('att_enc_W', [emb_dim, emb_dim], trainable=is_training)
        att_b = tf.get_variable('att_enc_b', [emb_dim], trainable=is_training)
        
        e_it = tf.matmul(tf.tanh(tf.nn.bias_add(tf.matmul(X1,image_W)+tf.matmul(X2,sent_W)+tf.multiply(X1,X2),att_b)),att_W)
        alpha_it = tf.nn.softmax(e_it)
        
        if skip:
            X1_att = X1 + tf.multiply(alpha_it, X1)
            X2_att = X2 + tf.multiply(alpha_it, X2)
        else:
            X1_att = tf.multiply(alpha_it, X1)
            X2_att = tf.multiply(alpha_it, X2)            

        # X1_att = tf.nn.l2_normalize(X1_att, 1, epsilon=1e-10)
        # X2_att = tf.nn.l2_normalize(X2_att, 1, epsilon=1e-10)
        return X1_att, X2_att, alpha_it
        
        
    