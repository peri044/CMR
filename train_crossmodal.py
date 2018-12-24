import tensorflow as tf
import numpy as np
from data.flowers_data_loader import *
from data.coco_data_loader import *
from data.flickr_data_loader import *
from data.pascal_data_loader import *
from model import *
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
import pdb
import datetime
import time
from attention import *
# tf.enable_eager_execution()
tf.set_random_seed(1234)
from loss import *
from attention import *

def get_vars(all_vars, scope_name, index):
	"""
	Helper function used to return specific variables of a subgraph
	Args: 
		all_vars: All trainable variables in the graph
		scope_name: Scope name of the variables to retrieve
		index: Clip the variables in the graph at this index
	Returns:
		Dictionary of pre-trained checkpoint variables: new variables
	"""
	ckpt_vars = [var for var in all_vars if var.op.name.startswith(scope_name)]
	ckpt_var_dict = {}
	for var in ckpt_vars:
		actual_var_name  = var.op.name
		if actual_var_name.find('Logits') ==-1:
			clip_var_name = actual_var_name[index:]
			ckpt_var_dict[clip_var_name] = var
		
	return ckpt_var_dict

def get_training_op(loss, args):
    """
    Defines the optimizers and returns the training op
    """
    # Gather all the variables in the graph
    all_vars = tf.trainable_variables()
    # Global step for the graph
    global_step = tf.train.get_or_create_global_step(graph=tf.get_default_graph())

    INITIAL_LEARNING_RATE=args.lr
    DECAY_STEPS = args.decay_steps
    LEARNING_RATE_DECAY_FACTOR = args.decay_factor
    # Decay the learning rate exponentially based on the number of steps.
    lr_non_emb = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  DECAY_STEPS,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
                                  
    lr_emb = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  DECAY_STEPS,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning rate', lr_non_emb)
    # Define the optimizers. Here, feature extractor and metric embedding layers have different learning rates during training.
    if args.optimizer=='adam':
        optimizer_non_emb = tf.train.AdamOptimizer(learning_rate=lr_non_emb)
        optimizer_emb = tf.train.AdamOptimizer(learning_rate=lr_non_emb)
    elif args.optimizer=='momentum':
        optimizer_non_emb = tf.train.MomentumOptimizer(learning_rate=lr_non_emb, momentum=0.9)
        optimizer_emb = tf.train.MomentumOptimizer(learning_rate=lr_emb, momentum=0.9)
    
    # Get variables of specific sub networks using scope names
    vars_fe = get_vars(all_vars, scope_name='Feature_extractor', index=18)
    vars_ie = get_vars(all_vars, scope_name='image_embedding', index=0)
    vars_te = get_vars(all_vars, scope_name='text_embedding', index=0)

    vars_emb_matrix = get_vars(all_vars, scope_name='embeddings/embedding', index=0)
    vars_seq2seq = get_vars(all_vars, scope_name='dynamic_seq2seq', index=0)

    vars_shared = get_vars(all_vars, scope_name='shared_embedding', index=0)
    fe_len, ie_len, te_len, emb_matrix_len, seq2seq_len, shared_len = len(vars_fe.values()), len(vars_ie.values()), len(vars_te.values()), len(vars_emb_matrix.values()), len(vars_seq2seq.values()), len(vars_shared)

    # Calculate gradients for respective layers
    if args.train_only_emb:
        grad = tf.gradients(loss, vars_ie.values() + vars_te.values()+ vars_shared.values())
        grad_ie = grad[:ie_len]
        grad_te = grad[ie_len:ie_len+te_len]
        grad_shared = grad[ie_len+te_len:]
    elif args.no_train_cnn:
        grad = tf.gradients(loss, vars_ie.values() + vars_te.values() + vars_seq2seq.values()+ vars_emb_matrix.values()+vars_shared.values())
        if args.clip_grad_norm:
            grad = [tf.clip_by_norm(tensor, args.clip_grad_norm, name=tensor.op.name+'_norm') if tensor is not None else None for tensor in grad]
        grad_ie = grad[:ie_len]
        grad_te = grad[ie_len: ie_len+te_len]
        grad_seq2seq = grad[ie_len+te_len: ie_len+te_len+seq2seq_len]
        grad_emb = grad[ie_len+te_len+seq2seq_len:ie_len+te_len+seq2seq_len+emb_matrix_len]
        grad_shared = grad[ie_len+te_len+seq2seq_len+emb_matrix_len:]
    else:
        grad = tf.gradients(loss, vars_fe.values() + vars_ie.values() + vars_te.values() + vars_seq2seq.values()+ vars_emb_matrix.values()+ vars_shared.values())
        if args.clip_grad_norm:
            grad = [tf.clip_by_norm(tensor, args.clip_grad_norm, name=tensor.op.name+'_norm') if tensor is not None else None for tensor in grad]

        grad_fe = grad[: fe_len]
        grad_ie = grad[fe_len: fe_len+ ie_len]
        grad_te = grad[fe_len+ie_len: fe_len+ie_len+te_len]
        grad_seq2seq = grad[fe_len+ ie_len+te_len: fe_len+ ie_len+te_len+seq2seq_len]
        grad_emb = grad[fe_len+ ie_len+te_len+seq2seq_len: fe_len+ ie_len+te_len+seq2seq_len+emb_matrix_len]
        grad_shared = grad[fe_len+ ie_len+te_len+seq2seq_len+emb_matrix_len:]

    # Define pre-trained savers
    image_pretrain_saver=None
    if not args.precompute:
        image_pretrain_saver = tf.train.Saver(var_list=vars_fe)
    lstm_pretrain_saver = tf.train.Saver(var_list= dict(vars_seq2seq.items() + vars_emb_matrix.items()))

    # Apply the gradients, update ops for batchnorm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        if args.train_only_emb:
            train_op = optimizer_emb.apply_gradients(zip(grad_ie+grad_te+grad_shared, vars_ie.values() + vars_te.values() + vars_shared.values()), global_step=global_step)
        elif args.no_train_cnn:
            train_op_non_emb = optimizer_non_emb.apply_gradients(zip(grad_seq2seq, vars_seq2seq.values()), global_step=global_step)
            train_op_emb = optimizer_emb.apply_gradients(zip(grad_ie+grad_te+grad_emb+grad_shared, vars_ie.values() + vars_te.values()+ vars_emb_matrix.values()+ vars_shared.values()))
            # Group individual training ops
            train_op = tf.group(train_op_non_emb, train_op_emb)
        else:
            train_op_non_emb = optimizer_non_emb.apply_gradients(zip(grad_fe+grad_seq2seq+grad_emb, vars_fe.values()+vars_seq2seq.values()+vars_emb_matrix.values()), global_step=global_step)
            train_op_emb = optimizer_emb.apply_gradients(zip(grad_ie+grad_te+grad_shared, vars_ie.values() + vars_te.values()+ vars_shared.values()))

            # Group individual training ops
            train_op = tf.group(train_op_non_emb, train_op_emb)

    return train_op, image_pretrain_saver, lstm_pretrain_saver,  global_step

def train(args):

    # Read the data
    if args.dataset=='flowers':
        dataset = FlowersDataLoader()
        image, caption, label, seq_len = dataset._read_data(args.record_path, args.batch_size, num_epochs=args.num_epochs)
    elif args.dataset=='mscoco':
        dataset = CocoDataLoader(precompute=args.precompute, use_random_crop=args.use_random_crop, max_len=args.max_len, model=args.model)
        image, caption, reverse_caption, seq_len = dataset._read_data(args.record_path, args.batch_size, phase=args.mode, num_epochs=args.num_epochs)
    elif args.dataset == 'flickr30k':
        dataset = FlickrDataLoader(precompute=args.precompute)
        image, caption, seq_len = dataset._read_data(args.record_path, args.batch_size, num_epochs=args.num_epochs)
    elif args.dataset == 'coco-ism':
        dataset = CocoDataLoader(precompute=args.precompute, use_random_crop=args.use_random_crop, max_len=args.max_len)
        image, logit_feat, caption, reverse_caption, seq_len = dataset._read_ism_data(args.record_path, args.batch_size, num_epochs=args.num_epochs)
    elif args.dataset == 'pascal':
        dataset = PascalDataLoader(precompute=args.precompute, use_random_crop=args.use_random_crop, max_len=args.max_len)
        image, caption, category, seq_len = dataset._read_data(args.record_path, args.batch_size, num_epochs=args.num_epochs)

    # Call the CMR model
    model=CMR(base=args.base, margin=args.margin, embedding_dim=args.emb_dim, word_dim=args.word_dim, vocab_file=args.vocab_file, vocab_size=args.vocab_size)
    if args.model=='vse' or args.model=='vse-vgg':
        image_embeddings, text_embeddings = model.build_vse_model(image, reverse_caption, seq_len, args, is_training=True)
        loss, loss_s, loss_im = model.sim_loss(image_embeddings, text_embeddings, args)
    elif args.model=='vse-att':
        image_embeddings, text_embeddings = model.build_vse_model(image, reverse_caption, seq_len, args, is_training=True)
        image_embeddings, text_embeddings, _ = aligned_attention(image_embeddings, text_embeddings, args.emb_dim, is_training=True)
        loss, loss_s, loss_im = model.sim_loss(image_embeddings, text_embeddings, args)
    elif args.model=="ism":
        image_embeddings, text_embeddings = model.build_ism_model(image, logit_feat, caption, seq_len, args, is_training=True)
        loss, loss_s, loss_im = model.sim_loss(image_embeddings, text_embeddings, args)
    elif args.model=='hrne':
        image_embeddings, text_embeddings = model.build_hrne_model(image, reverse_caption, seq_len, args, is_training=True)
        loss, loss_s, loss_im = model.sim_loss(image_embeddings, text_embeddings, args)
    elif args.model=='feat':
        image_embeddings, text_embeddings = model.build_featmap_model(image, reverse_caption, seq_len, args, is_training=True)
        loss, loss_s, loss_im = model.sim_loss(image_embeddings, text_embeddings, args)
    elif args.model=='bi':
        image_embeddings, text_embeddings, sim_matrix = model.build_bidirectional_model(image, caption, seq_len, args, is_training=True)
        loss, loss_s, loss_im = model.sim_loss(image_embeddings, text_embeddings, args, sim_scores=sim_matrix)
    elif args.model=='bi-conv':
        image_embeddings, text_embeddings, sim_matrix = model.build_bidirectional_model(image, caption, seq_len, args, is_training=True)
        loss, loss_s, loss_im = model.sim_loss(image_embeddings, text_embeddings, args, sim_scores=sim_matrix)
    elif args.model=='pascal':
        image_features, text_features = model.build_vse_model(image, caption, seq_len, args, is_training=True)
        with tf.variable_scope('image_embedding_global') as scope:
            image_embeddings = visual_feature_embed(image_features, 512)
            text_embeddings = sent_feature_embed(text_features, 512)
            image_embeddings, text_embeddings, _ = aligned_attention(image_embeddings, text_embeddings, 512)
        class_loss = category_loss(image_embeddings, text_embeddings, category, 20)
        metric_loss = triplet_loss(image_embeddings, text_embeddings, category, 1.)
        loss = metric_loss + class_loss
        
    total_loss = loss
    # Get the training op
    train_op, image_pretrain_saver, lstm_pretrain_saver, global_step = get_training_op(total_loss, args)

    # Add summaries
    if not args.precompute:
        tf.summary.image('Image', image)
    # Add summaries for image and text losses
    # tf.summary.scalar('Sentence Loss', loss_s)
    # tf.summary.scalar('Image Loss', loss_im)
    tf.summary.scalar('Total Loss', total_loss)

    #Merge summaries
    summary_tensor = tf.summary.merge_all()

    now = datetime.datetime.now()
    summary_dir_name = args.exp_path+'/s_'+args.model+'_'+args.measure+'_e'+str(args.emb_dim)+'_w'+str(args.word_dim)+'_L'+str(args.num_layers)+'_h'+str(args.num_units)+'_'+now.strftime("%Y-%m-%d_%H_%M")
    checkpoint_dir_name = args.exp_path+'/c_'+args.model+'_'+args.measure+'_e'+str(args.emb_dim)+'_w'+str(args.word_dim)+'_L'+str(args.num_layers)+'_h'+str(args.num_units)+'_'+now.strftime("%Y-%m-%d_%H_%M")
    summary_filewriter = tf.summary.FileWriter(summary_dir_name, tf.get_default_graph())

    # Checkpoint saver to save the variables of the entire graph. Training monitored session handles queue runners internally.
    num_hours=1.0
    if args.precompute: num_hours=0.25
    all_vars = tf.global_variables()
    if args.finetune_with_cnn:
        cnn_vars = [var for var in all_vars if var.op.name.startswith('image_embedding_2')]
        non_cnn_vars = list(set(all_vars) - set(cnn_vars))
        tmp_saver = tf.train.Saver(var_list=non_cnn_vars)
    else:
        tmp_saver = tf.train.Saver(var_list=all_vars)
        
    checkpoint_saver = tf.train.Saver(keep_checkpoint_every_n_hours=num_hours, max_to_keep=0)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(saver=checkpoint_saver, checkpoint_dir=checkpoint_dir_name, save_steps=args.save_steps)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.train.MonitoredTrainingSession(hooks=[checkpoint_saver_hook], config=config) as sess:
        # Write the parameters of the experiment in checkpoint_dir
        param_file = open(os.path.join(checkpoint_dir_name, 'exp_params.txt'), 'w')
        for key, value in vars(args).items():
            param_file.write(str(key)+' : '+ str(value)+'\n')
        param_file.close()
        # Load the pre-trained models
        if args.mode=='finetune':
            print "Restored : {}".format(args.checkpoint)
            tmp_saver.restore(sess, args.checkpoint)
            if args.finetune_with_cnn and not args.no_load_cnn: 
                print "Restored : {}".format(args.cnn_weights)
                image_pretrain_saver.restore(sess, args.cnn_weights)
        else:
            if not args.precompute:
                print "Restored : {}".format(args.cnn_weights)
                image_pretrain_saver.restore(sess, args.cnn_weights)
            if not args.no_pretrain_lstm:
                print "Restored : {}".format(args.lstm_weights)
                lstm_pretrain_saver.restore(sess, args.lstm_weights)
        start_time=time.time()
        while not sess.should_stop():
            try:
                summary_val, seq_len_val, _, total_loss_val, loss_s_val, loss_im_val, global_step_value, ie, te = sess.run([summary_tensor, seq_len, train_op, \
                                                                                                                total_loss, loss_s, loss_im, global_step,
                                                                                                                image_embeddings, text_embeddings])
                                                                                                                
                # summary_val, seq_len_val, _, total_loss_val, class_loss_val, metric_loss_val, global_step_value= sess.run([summary_tensor, seq_len, train_op, \
                                                                                                                # total_loss, class_loss, metric_loss, global_step])                                                                                            
                if (global_step_value+1)%100==0: 
                    end_time=time.time()
                    print "Iteration: {} Total: {} Sentence : {} Image : {} Step time: {}".format(global_step_value+1, total_loss_val, loss_s_val, loss_im_val, (end_time-start_time)/100)
                    # print "Iteration: {} Total: {} Metric : {} Class : {} Step time: {}".format(global_step_value+1, total_loss_val, metric_loss_val, class_loss_val, (end_time-start_time)/100)
                    summary_filewriter.add_summary(summary_val, global_step_value)
                    start_time=time.time()
                    
            except tf.errors.OutOfRangeError:
                break
                
        print "Training completed"

                
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--save_steps', type=int, default=2000, help="Checkpoint saving step interval")
    parser.add_argument('--decay_steps', type=int, default=10000, help="Checkpoint saving step interval")
    parser.add_argument('--decay_factor', type=float, default=0.9, help="Checkpoint saving step interval")
    parser.add_argument('--emb_dim', type=int, default=2048, help="CVS dimension")
    parser.add_argument('--word_dim', type=int, default=300, help="Word Embedding dimension")
    parser.add_argument('--margin', type=float, default=0.05, help="Margin component")
    parser.add_argument('--lambda_1', type=float, default=9.0, help="SCAN softmax temperature")
    parser.add_argument('--lambda_2', type=float, default=9.0, help="LSE similarity temperature")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--vocab_size', type=int, default=26375, help="Number of hidden RNN units")
    parser.add_argument('--num_units', type=int, default=1024, help="Number of hidden RNN units")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in RNN network")
    parser.add_argument('--clip_grad_norm', type=float, default=None, help="Value of gradient clipping")
    parser.add_argument('--max_len', type=int, default=None, help="Value of maximum caption length")
    parser.add_argument('--stride', type=int, default=4, help="Value of stride in HRNE")
    parser.add_argument('--record_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_train.tfrecord', help="Batch size")
    parser.add_argument('--optimizer', type=str, default='adam', help="Optimizer")
    parser.add_argument('--base', type=str, default='inception_v1', help="Base architecture")
    parser.add_argument('--dataset', type=str, default='mscoco', help="Dataset mscoco|flickr30k|flowers")
    parser.add_argument('--measure', type=str, default='cosine', help="Type of loss")
    parser.add_argument('--exp_path', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data', help="Experiment dir")
    parser.add_argument('--vocab_file', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/vocab_mscoco.enc', help="Validation TF record")
    parser.add_argument('--train_only_emb', action='store_true', help="train only embedding layer")
    parser.add_argument('--no_train_cnn', action='store_true', help="Flag to not train CNN")
    parser.add_argument('--no_pretrain_lstm', action='store_true', help="Flag to not use pre-trained LSTM weights")
    parser.add_argument('--mine_n_hard', type=int, default=0, help="Flag to enable hard negative mining")
    parser.add_argument('--precompute', action='store_true', help="Flag to use precomputed CNN features")
    parser.add_argument('--use_random_crop', action='store_true', help="Use random cropping")
    parser.add_argument('--finetune_with_cnn', action='store_true', help="Flag to finetune the model with CNN and embedding already trained together before")
    parser.add_argument('--no_load_cnn', action='store_true', help="Flag to not load cnn weights while finetuning")
    parser.add_argument('--use_abs', action='store_true', help="use_absolute values for embeddings")
    parser.add_argument('--cnn_weights', type=str, default='/shared/kgcoe-research/mil/peri/tf_checkpoints/inception_v1.ckpt', help="CNN checkpoint")
    parser.add_argument('--lstm_weights', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/mscoco_1024d_2gru/best_bleu/translate.ckpt-35000', help="LSTM checkpoint")
    parser.add_argument('--checkpoint', type=str, default='/shared/kgcoe-research/mil/peri/mscoco_data/check_vse_order_e2048_w300_2018-08-22_19_31/model.ckpt-11700', help="CMR checkpoint")
    parser.add_argument('--model', type=str, default='vse', help="Name of the model")
    parser.add_argument('--mode', type=str, default='train', help="Fine-tuning the model")
    args=parser.parse_args()
    print '--------------------------------'
    for key, value in vars(args).items():
        print key, ' : ', value
    print '--------------------------------'
    train(args)
