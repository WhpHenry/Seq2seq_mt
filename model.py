import time
import tensorflow as tf 


from utils import _GO_i, _EOS_i
_epochs = 10
_batch_size = 128
_num_utils = 128
_layers_count = 1
_max_tar_seq_len = 20
_encode_embed_dim = 100
_decode_embed_dim = 100
_lr = 0.001
_display_step = 50
_save_step = 50
_model_path = 'saved_networks/'
_model_name = "model-mt"

class Seq2seq(object):
    
    def __init__(self, source_vocab_size, target_vocab_size,  
                 num_units = 128, layers_count = 1, max_tar_seq_len = 20, 
                 encode_embed_dim = 100, decode_embed_dim = 100, 
                 go_idx = _GO_i, eos_idx = _EOS_i, batch_size = _batch_size):

        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')     # [batch_size,doc_len]
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')   # [batch_size,doc_len]
        self.source_seq_len = tf.placeholder(tf.int32, (None,), name='source_seq_len')# [batch_size] 
        self.target_seq_len = tf.placeholder(tf.int32, (None,), name='target_seq_len')# [batch_size]
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')   

        # build graph
        self.graph = tf.Graph()
        # with self.graph.as_default():
        train_logits, infer_logits = self.build_model(tf.reverse(self.inputs, [-1]),
                                                      self.targets,
                                                      num_units,
                                                      layers_count,
                                                      self.source_seq_len,
                                                      source_vocab_size,
                                                      encode_embed_dim,
                                                      self.target_seq_len,
                                                      target_vocab_size,
                                                      decode_embed_dim,
                                                      max_tar_seq_len,
                                                      go_idx,
                                                      eos_idx,
                                                      batch_size)

        training_logits = tf.identity(train_logits.rnn_output, name='logits')
        inference_logits= tf.identity(infer_logits.sample_id, name='predictions')
        # bool mask with [True] * target_seq_len and shape: [max_tar_seq_len]
        masks = tf.sequence_mask(self.target_seq_len, 
                                 max_tar_seq_len, 
                                 dtype=tf.float32, name='masks')

        with tf.name_scope('optimization'):
            # Weighted cross-entropy loss for a sequence of logits
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, masks)
            # Construct Adam optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # Compute gradients of loss. the first part of minimize()
            gradients = optimizer.compute_gradients(cost) 
            clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) 
                                 for grad, var in gradients if grad is not None]
            # Apply gradients to variables. the second part of minimize()
            train_op = optimizer.apply_gradients(clipped_gradients)

        self.batch_size = batch_size
        self.inference_logits = inference_logits
        self.cost = cost
        self.train_op = train_op
        self.saver = tf.train.Saver()
        print('\n* Seq2seq model initial finished *\n')

    # return LSTM or GRU cell
    def _cell(self, num_units, seed=123, is_gru=False):
        init = tf.random_uniform_initializer(-1, 1, seed=seed)
        if is_gru:
            return tf.contrib.rnn.GRUCell(num_units, initializer=init)
        return tf.contrib.rnn.LSTMCell(num_units, initializer=init)

    def encoder(self, ipt, num_units, layers_count, vocab_size, embed_dim, seq_len):
        '''
            ipt: input data of encoder, shape=[batch_size, doc_len]
            num_units: rnn cell num
            layers_count: rnn layers num
            vocab_size: integer number of source symbols in vocabulary. 
            embed_dim: after sequence embed, ipts should be [batch_size, doc_len, embed_dim]
            seq_len: dynamic rnn need all(batch_size) sequences' length
        '''
        embed = tf.contrib.layers.embed_sequence(ipt, vocab_size, embed_dim)
        rnns = tf.contrib.rnn.MultiRNNCell([self._cell(num_units) for _ in range(layers_count)])
        opts, stats = tf.nn.dynamic_rnn(cell=rnns, 
                                        inputs=embed, 
                                        sequence_length=seq_len,
                                        dtype=tf.float32)
        # Line 96 - ValueError: Tensor("rnn/Const:0", shape=(1,), dtype=int32) 
        # must be from the same graph as Tensor("Equal:0", shape=(1,), dtype=bool).
        return opts, stats

    def decoder_inputs(self, tar_seq, batch_size, go_idx):
        # remove <EOS> in sequence of each batch
        sub_tar_seq =  tf.strided_slice(tar_seq, [0, 0], [batch_size, -1], [1, 1])
        # add index of <GO> in sequence head of each batch
        decode_ipt = tf.concat([tf.fill([batch_size, 1], go_idx), sub_tar_seq], axis=1)
        return decode_ipt

    def _decode_training(self, embed, tar_seq_len, decode_cell, 
                         encode_stats, opt_layer, max_iter):
        '''
            embed: decode input after embeded and embedding_lookup
            max_iter: here will be max_target_seq_len
            opt_layer: apply to the RNN output prior to storing the result or sampling
        '''

        helper = tf.contrib.seq2seq.TrainingHelper(inputs=embed,
                                                   sequence_length=tar_seq_len,
                                                   time_major=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decode_cell,
                                                  helper=helper,
                                                  initial_state=encode_stats,
                                                  output_layer=opt_layer)
        # return opts, stats, final_seq_lens 
        # impute finished: if True, states for batch entries which are marked as 
        # finished get copied through and the corresponding outputs get zeroed out 
        # maximum_iterations: maximum allowed number of decoding 
        opts, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                       impute_finished=True,
                                                       maximum_iterations=max_iter)
        return opts

    def _decode_inference(self, go_idx, eos_idx, embed, batch_size,
                         decode_cell, encode_stats, opt_layer, max_iter):
                         
        '''
            go_idx: will be a token constanted as a [batch_size] vector
            embed: decode input after embeded before embedding_lookup
            
        '''
        go_token = tf.tile(tf.constant([go_idx], dtype=tf.int32), [batch_size], name='start_token')
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embed,
                                                          start_tokens=go_token,
                                                          end_token=eos_idx)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decode_cell,
                                                  helper=helper,
                                                  initial_state=encode_stats,
                                                  output_layer=opt_layer)    
        opts, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                       impute_finished=True,
                                                       maximum_iterations=max_iter)
        return opts

    def decoder(self, decode_ipts, vocab_size, embed_dim, num_units, layers_count, 
                tar_seq_len, encode_stats, max_tar_seq_len, go_id, eos_id, batch_size): 
        '''
            vocab_size: integer number of target symbols in vocabulary. 
        '''
        infer_embed = tf.Variable(tf.random_uniform([vocab_size, embed_dim]))
        # embedding lookup: catch vec in infer_embed by id in decode_ipts
        train_embed = tf.nn.embedding_lookup(infer_embed, decode_ipts)

        rnns = tf.contrib.rnn.MultiRNNCell([self._cell(num_units, seed=456) 
                                            for _ in range(layers_count)])

        # full connect dense layer of output
        opt_layer = tf.layers.Dense(vocab_size)

        with tf.variable_scope('decoder'):
            tr_logits = self._decode_training(train_embed, 
                                              tar_seq_len, 
                                              rnns, 
                                              encode_stats,
                                              opt_layer,
                                              max_tar_seq_len) 
        with tf.variable_scope('decoder', reuse=True):
            if_logits = self._decode_inference(go_id, eos_id, 
                                               infer_embed,
                                               batch_size, 
                                               rnns, 
                                               encode_stats,
                                               opt_layer,
                                               max_tar_seq_len) 
        return tr_logits, if_logits

    def build_model(self, inputs, targets, num_units, layers_count, 
                    source_seq_len, source_vocab_size, encode_embed_dim, 
                    target_seq_len, target_vocab_size, decode_embed_dim, 
                    max_tar_seq_len, go_idx, eos_idx, batch_size):
        '''
            targets: target_data for decode training, e.g.
                     give the same input as encoder, like [<GO> inputs] 
        '''
        _, stats = self.encoder(inputs, 
                                num_units, 
                                layers_count, 
                                source_vocab_size, 
                                encode_embed_dim, 
                                source_seq_len)

        decode_ipts = self.decoder_inputs(targets, batch_size, go_idx)

        tr_logits, inf_logits = self.decoder(decode_ipts, 
                                             target_vocab_size, 
                                             decode_embed_dim, 
                                             num_units, 
                                             layers_count,
                                             target_seq_len, 
                                             stats, 
                                             max_tar_seq_len, 
                                             go_idx, 
                                             eos_idx, 
                                             batch_size)
        return tr_logits, inf_logits
    
    def _get_batch(self, source, target):
        '''
            yield batches
        '''
        for i in range(len(source)//self.batch_size):
            start = i * self.batch_size

            s_batch = source[start : start + self.batch_size]
            t_batch = target[start : start + self.batch_size]

            s_seq_lens = [len(s) for s in s_batch]  # source seq len
            t_seq_lens = [len(t) for t in t_batch]  # target seq len

            yield s_batch, t_batch, s_seq_lens, t_seq_lens


    def training(self, source, target, source_data_count, epochs=_epochs, rate=_lr):
        
        print('Training get start !')

        with tf.Session() as sess:
            # load checkpoint including all structure
            checkpoint = tf.train.get_checkpoint_state(_model_path)     
            if checkpoint and checkpoint.model_checkpoint_path:                 
                # load trained network
                self.saver.restore(sess, checkpoint.model_checkpoint_path)
                print ("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print ("no old network weights can be loaded")
            
            sess.run(tf.global_variables_initializer())
            steps = source_data_count//self.batch_size
            for epoch_i in range(epochs):
                for batch_i, (s_batch, t_batch, s_seq_lens, t_seq_lens) in enumerate(
                    self._get_batch(source, target)
                ):
                    start = time.time()
                    _, loss = sess.run(
                        [self.train_op, self.cost], 
                        feed_dict={
                            self.inputs: s_batch,
                            self.targets: t_batch,
                            self.source_seq_len: s_seq_lens,
                            self.target_seq_len: t_seq_lens,
                            self.learning_rate: rate
                        }
                    )

                    # run decode inference_logits and display all states
                    if batch_i % _display_step == 0:
                        sess.run(                   
                            self.inference_logits,
                            feed_dict={
                                self.inputs: s_batch,
                                self.source_seq_len: s_seq_lens,
                                self.target_seq_len: t_seq_lens
                            }
                        )
                        period = time.time() - start
                        start += period
                        print('Epoch {} Batch {}/{} - Loss: {} - Period: {}'
                        .format(epoch_i, batch_i, steps, loss, period))
                # save epoch i model
                print('save sub_epoch_{} model'.format(epoch_i+1))
                self.saver.save(sess, _model_path + _model_name, global_step=((epoch_i+1) * steps))
            # Save final Model
            print('Model Trained and Saved')
            self.saver.save(sess, _model_path + _model_name, global_step=(epochs * steps))
            print('Training {} epoachs finished'.format(epochs))
        
    
    def inference(self, sentence_int_seq):
        # get sentence_int_seq after txt to int transform
        print('Inferencing get start !')
        my_graph = tf.Graph()
        start = time.time()
        with tf.Session(graph=my_graph) as sess:
            checkpoint = tf.train.get_checkpoint_state(_model_path)
            if checkpoint and checkpoint.model_checkpoint_path:                 
                # load trained network
                model_file = (checkpoint.model_checkpoint_path + '.meta')
                loader = tf.train.import_meta_graph(model_file) # import .meta file
                loader.restore(sess, tf.train.latest_checkpoint(_model_path))
                print ("Successfully loaded: ", checkpoint.model_checkpoint_path)
            else:
                raise Exception("ERROR: No trained model can be loaded for predict")            

            input_data = my_graph.get_tensor_by_name('inputs:0')
            inf_logits = my_graph.get_tensor_by_name('predictions:0')
            target_seq_len = my_graph.get_tensor_by_name('target_seq_len:0')
            source_seq_len = my_graph.get_tensor_by_name('source_seq_len:0')
                    
            translate_logits = sess.run(
                inf_logits,
                feed_dict={
                    input_data: [sentence_int_seq] * self.batch_size,
                    target_seq_len: [len(sentence_int_seq) * 2] * self.batch_size, 
                    source_seq_len: [len(sentence_int_seq)] * self.batch_size 
                }
            )[0]
        print('Inferencing finished, cost time {}'.format(time.time()-start))
        return translate_logits 
    