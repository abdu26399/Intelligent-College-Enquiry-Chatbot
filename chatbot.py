import numpy as np
import tensorflow as tf
import re
import time


# MODULE 1 - DATA PREPROCESSING 

# we get the dataset first

singletalk = open("singleconvos.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')
setoftalk = open("setofconvos.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')

#Link up every convo and its ID by using a dictionary
    
linkidline = {}
for line in singletalk:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        linkidline[_line[0]] = _line[4]
 
# Creating a list of all of the conversations
linkidconvoset = []
for conversation in setoftalk[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    linkidconvoset.append(_conversation.split(','))
 
#separate the questions and the answers from the dataset
ask = []
get = []
for conversation in linkidconvoset:
    for i in range(len(conversation) - 1):
        ask.append(linkidline[conversation[i]])
        get.append(linkidline[conversation[i+1]])
     
#doing a first cleaning of the text like making text lowercase, removing apostrophes
        
def cleandataset(data):
    data= data.lower()
    data = re.sub(r"i'm", "i am", data)
    data = re.sub(r"he's", "he is", data)
    data = re.sub(r"she's", "she is", data)
    data = re.sub(r"that's", "that is", data)
    data = re.sub(r"where's", "where is", data)
    data = re.sub(r"how's", "how is", data)
    data = re.sub(r"\'ll", " will", data)
    data = re.sub(r"\'ve", " have", data)
    data = re.sub(r"\'re", " are", data)
    data = re.sub(r"\'d", " would", data)
    data = re.sub(r"won't", " will not", data)
    data = re.sub(r"can't", " cannot", data)
    data = re.sub(r"it's", " it is", data)
    data = re.sub(r"don't", " do you not", data)
    data = re.sub(r"haven't", " have not", data)
    data = re.sub(r"let's", " let us", data)
    data = re.sub(r"wouldn't", " would not", data)
    data = re.sub(r"you're", " you are", data)
    data = re.sub(r"who'll", " who will", data) 
    data = re.sub(r"[-()\"#/&@;:<>{}+=~|.,*$]", "", data)
    return data

#removing unnecessary symbols from questions
cleanedque = []
for que in ask:
    cleanedque.append(cleandataset(que))

#removing unnecessary symbols from answers  
cleanedans = []
for ans in get:
    cleanedans.append(cleandataset(ans))
    
#filtering out very small questions
short_questions = []
short_answers = []
i = 0
for question in cleanedque:
    if 1 <= len(question.split()) <= 25:
        short_questions.append(question)
        short_answers.append(cleanedans[i])
    i += 1
cleanedque = []
cleanedans = []
i = 0
for answer in short_answers:
    if 2 <= len(answer.split()) <= 25:
        cleanedans.append(answer)
        cleanedque.append(short_questions[i])
    i += 1

#create a dictionary to map every word to the number of times it occurs  
word2count = {}
for askque in cleanedque:
    for word in askque.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in cleanedans:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

#Create 2 dictionaries to map the words of questions and asnwers to an integer
limit_que = 0
ques2int = {}
wordnumber = 0
for word, number in word2count.items():
    if number>= limit_que:
        ques2int[word] = wordnumber
        wordnumber +=   1
limit_ans = 0
ans2int = {}
wordnumber = 0
for word, number in word2count.items():
    if number>=limit_ans:
        ans2int[word] = wordnumber
        wordnumber += 1

#Now add the Final tokens  like EOS, SOS to the dictionaries we created before
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    ques2int[token] = len(ques2int) + 1
for token in tokens:
    ans2int[token] = len(ans2int) + 1

#creating the inverse mapping of the dictionary
int2ans = {w_i : w for w, w_i in ans2int.items()} 

#Add EOS to end of every answer as it will be used in decoding later
for i in range(len(cleanedans)):
    cleanedans[i] += ' <EOS>'
    
#Convert all the questions and answers into integers and replace the words filtered by <OUT> that were used less than the threshold limit given above
quesintoint = []
for question in cleanedque:
    ints = []
    for word in question.split():
        if word not in ques2int:
            ints.append(ques2int['<OUT>'])
        else:
            ints.append(ques2int[word])
    quesintoint.append(ints)
ansintoint = []
for answer in cleanedans:
    ints = []
    for word in answer.split():
        if word not in ans2int:
            ints.append(ans2int['<OUT>'])
        else:
            ints.append(ans2int[word])
    ansintoint.append(ints)    
    
#sort questions and answers by length of questions to speed up and optimize the training
sortedcleanquestions = []
sortedcleananswers = []
for length in range(1, 25   +1):
    for i in enumerate(quesintoint):
        if len(i[1]) == length:
            sortedcleanquestions.append(quesintoint[i[0]])
            sortedcleananswers.append(ansintoint[i[0]])
            
###### MODULE 2 - BUILDING THE MODEL USING DEEP LEARNING

# Creating a placeholder for the answers to be predicted
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keepprob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, lr, keepprob

# Preprocessing the answers which will be predicted
def processpredictions(targets, word2int, grouplength):
    beginning = tf.fill([grouplength, 1], word2int['<SOS>'])
    ending = tf.strided_slice(targets, [0,0], [grouplength, -1], [1,1]) # batch_size being the number of lines
    preprocessed_targets = tf.concat([beginning,ending], 1) # 1 = horizontal; 2 = vertical
    return preprocessed_targets

# Creating the encode layer which is the first main step of the architecture
def layer1_e(inprnn, no_of_input_tensors,nooflayers, keepprob, lngthofseqnce):
    lstm = tf.contrib.rnn.BasicLSTMCell(no_of_input_tensors)
    lstm_droprate = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keepprob) # Deactivates neurons during training
    e_cell = tf.contrib.rnn.MultiRNNCell([lstm_droprate] * nooflayers)
    # All their outputs - encoder_output, encoder_state; we only want the encoder_state so we use _
    _,e_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=e_cell, cell_bw=e_cell,sequence_length=lngthofseqnce,inputs=inprnn,dtype=tf.float32)
    return e_state

# Decoding the training set
def d_train(e_state, d_cell, d_embed_inp, lngthofseqnce, decoding_scope, op, keepprob, grouplength):
    attention_states = tf.zeros([grouplength, 1, d_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option="bahdanau", num_units =d_cell.output_size)
    d_train_function = tf.contrib.seq2seq.attention_decoder_fn_train(e_state[0],attention_keys,attention_values,attention_score_function,attention_construct_function,name='attn_dtrain')   
    # Outputs are d_output, d_final state, decoder final_context_state; we only want first one
    d_out,_,_ = tf.contrib.seq2seq.dynamic_rnn_decoder(d_cell,d_train_function,d_embed_inp,lngthofseqnce,scope=decoding_scope)
    d_op_droprate = tf.nn.dropout(d_out, keepprob)
    return op(d_op_droprate)

# Decoding the test/validation set
def d_test(e_state, d_cell, d_embed_matrix,startofline,endofline, maxlengthofword, noofwords, decoding_scope, op, keepprob, grouplength):
    attention_states = tf.zeros([grouplength, 1, d_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option="bahdanau", num_units =d_cell.output_size)
    d_test_func = tf.contrib.seq2seq.attention_decoder_fn_inference(op,e_state[0],attention_keys,attention_values,attention_score_function,attention_construct_function,d_embed_matrix,startofline,endofline,maxlengthofword,noofwords,name='attn_dinference')
    # Outputs are test predictions, d_final_state, d_final_context_state; we only want first one
    predicttestanswers,_,_ = tf.contrib.seq2seq.dynamic_rnn_decoder(d_cell,d_test_func,scope=decoding_scope)
    return predicttestanswers

#creating the final layer ie the decode layer of the main deep learning NLP architecture
def layer2_d(d_embed_inp, d_embed_matrix, e_state, noofwords, lngthofseqnce, no_of_input_tensors, nooflayers, word2int, keepprob, grouplength):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(no_of_input_tensors)
        lstm_droprate = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keepprob)
        d_cell = tf.contrib.rnn.MultiRNNCell([lstm_droprate] * nooflayers)
        wghts = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        op_func = lambda x: tf.contrib.layers.fully_connected(x,noofwords,None, scope = decoding_scope,weights_initializer = wghts,biases_initializer = biases)
        train_predictions = d_train(e_state,d_cell,d_embed_inp,lngthofseqnce,decoding_scope,op_func,keepprob,grouplength)
        decoding_scope.reuse_variables()
        predicttestanswers = d_test(e_state,d_cell,d_embed_matrix, word2int['<SOS>'], word2int['<EOS>'],lngthofseqnce - 1 ,noofwords,decoding_scope,op_func,keepprob,grouplength)
    return train_predictions, predicttestanswers   

#Build the seq2seq model using above functions
def seq2seq_model(inputs, targets, keepprob, grouplength, lngthofseqnce, answers_num_words, questions_num_words, encoder_embed_size, decoder_embed_size, no_of_input_tensors, nooflayers, ques2int):
    encoder_embed_inp = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embed_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    e_state = layer1_e(encoder_embed_inp, no_of_input_tensors, nooflayers, keepprob, lngthofseqnce)
    preprocessed_targets = processpredictions(targets, ques2int, grouplength)
    d_embed_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embed_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(d_embed_matrix, preprocessed_targets)
    train_predictions, predicttestanswers = layer2_d(decoder_embedded_input,
                                                         d_embed_matrix,
                                                         e_state,
                                                         questions_num_words,
                                                         lngthofseqnce,
                                                         no_of_input_tensors,
                                                         nooflayers,
                                                         ques2int,
                                                         keepprob,
                                                         grouplength)
    return train_predictions, predicttestanswers

### MODULE - 3 TRAINING THE CHATBOT

#Set the Hyperparamters
epochs = 100
grouplength = 10
no_of_input_tensors = 256
nooflayers = 3
encoder_embed_size = 256
decoder_embed_size = 256
learn_rate = 0.01
learn_rate_decay = 0.9
min_learn_rate = 0.0001
keep_probability = 0.5

#Define a session
tf.reset_default_graph()
session = tf.InteractiveSession()

#Load the model inputs
inputs, targets, lr, keepprob = model_inputs()

#Set the sequence length
lngthofseqnce = tf.placeholder_with_default(25, None,name = 'sequence_length')

#set input shape of input tensor
inp_shape = tf.shape(inputs)

# get the training and test predictions                Shape the input by using reverse #keep prob is real one keep_probability = keepprob connection will be done later for API     
train_predictions, predicttestanswers = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keepprob,
                                                       grouplength,
                                                       lngthofseqnce,
                                                       len(ans2int),
                                                       len(ques2int),
                                                       encoder_embed_size,
                                                       decoder_embed_size,
                                                       no_of_input_tensors,
                                                       nooflayers,
                                                       ques2int)

#Set up 2 scopes adam optimiser and loss error and apply gradient clipping to them.
with tf.name_scope("optimization_scope"):
    lss_err = tf.contrib.seq2seq.sequence_loss(train_predictions, targets, tf.ones([inp_shape[0],lngthofseqnce]))
    optimizer = tf.train.AdamOptimizer(learn_rate)
    gradients = optimizer.compute_gradients(lss_err)
    clip_gradients = [(tf.clip_by_value(gradient_tensor,-5.,5.), gradient_variable) for gradient_tensor, gradient_variable in gradients if gradient_tensor is not None]
    optimizer_clipped = optimizer.apply_gradients(clip_gradients)

#Apply padding to the sequences with PAD token to match the length of quentions and answers
def apply_pad(seqnces_batch, word2int):
    max_seq_length = max([len(sequence) for sequence in seqnces_batch])
    return [sequence + [word2int['<PAD>']] * (max_seq_length - len(sequence))for sequence in seqnces_batch]

#Split the data into batches of ques and ans
def split_que_ans_to_batches(questions, answers, grouplength):
    for index_of_batch in range(0,len(questions) // grouplength):
        start_index = index_of_batch * grouplength
        ques_in_batch = questions[start_index : start_index + grouplength]
        ans_in_batch = answers[start_index : start_index + grouplength]
        padded_ques_in_batch = np.array(apply_pad(ques_in_batch,ques2int))
        padded_ans_in_batch = np.array(apply_pad(ans_in_batch,ans2int))
        yield padded_ques_in_batch, padded_ans_in_batch

##Split Ques And Ans into train/validation sets
train_validation_split = int(len(sortedcleanquestions) * 0.15)
train_ques = sortedcleanquestions[train_validation_split:]
train_ans = sortedcleananswers[train_validation_split:]
validation_ques = sortedcleanquestions[:train_validation_split]
validation_ans = sortedcleananswers[:train_validation_split]

#Training
batch_index_check_train_loss = 100
batch_index_check_validation_loss = ((len(train_ques)) // grouplength // 2) - 1
total_train_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for index_of_batch, (padded_ques_in_batch, padded_ans_in_batch) in enumerate (split_que_ans_to_batches(train_ques, train_ans, grouplength)):
        start_time = time.time()
        _, batch_train_loss_error = session.run([optimizer_clipped, lss_err], {inputs: padded_ques_in_batch,
                                                                                               targets: padded_ans_in_batch,
                                                                                               lr: learn_rate,
                                                                                               lngthofseqnce: padded_ans_in_batch.shape[1],
                                                                                               keepprob: keep_probability})
        total_train_loss_error += batch_train_loss_error
        end_time = time.time()
        batch_time = end_time - start_time
        if index_of_batch % batch_index_check_train_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,epochs,index_of_batch, len(train_ques) // grouplength, total_train_loss_error / batch_index_check_train_loss, int(batch_time * batch_index_check_train_loss)))
    #3 figures over total no ; 4 figures over total batches   trainlosserror= 6figures,3decimals, train time on 10 BATCH
            total_train_loss_error = 0
        if index_of_batch % batch_index_check_validation_loss == 0 and index_of_batch > 0:
            total_validation_loss_error = 0
            start_time = time.time()
            for batch_index_validation, (padded_ques_in_batch, padded_ans_in_batch) in enumerate(split_que_ans_to_batches(validation_ques, validation_ans, grouplength)):
                batch_validation_loss_error = session.run(lss_err, {inputs: padded_ques_in_batch,
                                                                       targets: padded_ans_in_batch,
                                                                       lr: learn_rate,
                                                                       lngthofseqnce: padded_ans_in_batch.shape[1],
                                                                       keepprob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - start_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_ques) / grouplength)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learn_rate *= learn_rate_decay
            if learn_rate < min_learn_rate:
                learn_rate = min_learn_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I can speak better now')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I cannot speak better presently, I need to practice more to answer your questions more accurately")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("I am sorry, I cannot speak better than this. This is the best I can do.")
        break
print("Game Over")

#loading the checkpoints
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)
 
# Converting the questions from strings to lists of encoding integers
def convert_string2int(ques, word2int):
    ques = cleandataset(question)
    return [word2int.get(word, word2int['<OUT>']) for word in ques.split()]
 
# Setting up the chat
while(True):
    ques = input("User: ")
    if ques == 'Goodbye':
        break
    ques = convert_string2int(ques, ques2int)
    ques = ques + [ques2int['<PAD>']] * (25 - len(ques))
    fake_batch = np.zeros((grouplength, 25))
    fake_batch[0] = ques
    predicted_answer = session.run(predicttestanswers, {inputs: fake_batch, keepprob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if int2ans[i] == 'i':
            token = 'I'
        elif int2ans[i] == '<EOS>':
            token = '.'
        elif int2ans[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + int2ans[i]
        answer += token
        if token == '.':
            break
    print('Bot: ' + answer)