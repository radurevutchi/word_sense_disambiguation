import numpy as np
import pandas as pd
import os
import sys
import random
import keras
import tensorflow as tf
import json_lines
import keras.backend as K
tf.gfile = tf.io.gfile

os.environ['CUDA_VISIBLE_DEVICES'] = '1'





# Loss function which dynamically selects which word to finetune on
def custom_loss(y_true, y_pred):

	y_true1 = []
	y_pred1 = []
	length = y_true.shape[0]
	print(y_true.numpy())
	for i in range(length):
		y_true1.append(y_true[i][y_true[i][0][0]])
		y_pred1.append(y_pred[i][y_pred[i][0][0]])

	assert(y_pred1[0].shape[0] == 768)
	assert(y_true1[0].shape[0] == 768)

	loss = K.square(y_pred - y_true)
	print(loss)
	print('dd')
	loss = K.sum(loss, axis=1)
	print(loss)
	return loss







# CREATES LABELS for CLS TOKENS
# SAVES THEM in DICTIONARY

sense_labels = open('sense_definition.txt','r').readlines()
sense_labels = list(map(lambda x: x[0:x.index(',')],sense_labels))
sense_cls = {}
cls_labels = []
with open('output_definition.jsonl') as f:
	for item in json_lines.reader(f):
		cls_labels.append(item)

for si in range(len(sense_labels)):

	sense_cls[sense_labels[si]] = cls_labels[si]['features'][0]['layers'][0]['values']
assert(len(sense_cls) == 65)





# BERT tokenizer
from google_research_code import tokenization
tokenizer = tokenization.FullTokenizer(
	vocab_file='google_research_code/vocab.txt', do_lower_case=True)



# Sense labels
# Maps labels to word in each sentence
sentences = []
word_indices = []
sense_mappings = []
words_of_interest = ['serve','back','left','right','open']
# NEED A SENTENCE TO SENSE MAPPING
for word_of_interest in words_of_interest:

	print(word_of_interest.upper())
	print('\n\n')
	directory = 'testing_' + word_of_interest + '/'

	# sense mapping index and label files
	sense_index = np.load(directory + 'sense_mappings_index_test.npy', allow_pickle=True)
	sense_label = np.load(directory + 'sense_labels_test.npy', allow_pickle=True)
	sents_tokenized = open(directory + 'sentences_tokenized_test.txt').readlines()



	######
	# Iterates through each sentence of data
	# OBTAINS EMBEDDING LAYERS FROM JSON FILE
	######
	sense_embedding = {}
	for si in range(len(sense_index)):

		# Index, label of current sentence
		index0 = sense_index[si]
		label0 = sense_label[si]
		temp_sent =  sents_tokenized[si][:-1]
		xml_tokens_string_bert = ['[CLS]'] + tokenizer.tokenize(temp_sent) + ['[SEP]']

		# Iterates through each token of current sentence
		for i in range(len(xml_tokens_string_bert)):

			#print(i, sent0[i]['token'])
			if xml_tokens_string_bert[i] == word_of_interest and i in index0:
				assert(xml_tokens_string_bert[i] == word_of_interest)

				# Identifies the label and index
				label_idx = index0.index(i)
				label = label0[label_idx]
				sentence = sents_tokenized[si][:-1]

				if label in sense_cls:

					sentences.append(sentence)
					word_indices.append([np.array([i]*768)])
					sense_mappings.append(label)


word_indices = np.array(word_indices)




Y = []
for si in range(len(sentences)):
	Y.append(sense_cls[sense_mappings[si]])

Y = np.array(Y)
print(Y.shape)
assert(len(Y) == len(sentences))


# transformers library
# Tokenizes and converts to input embeddings
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# ALL INPUT TRAINING DATA
input_ids = []
token_type_ids = []
attention_mask = []
for sentence in sentences:
	tokenized_X = tokenizer([sentence], max_length=50, return_tensors='tf', padding='max_length', truncation=True)
	input_ids.append(tokenized_X['input_ids'][0].numpy())
	token_type_ids.append(tokenized_X['token_type_ids'][0].numpy())
	attention_mask.append(tokenized_X['attention_mask'][0].numpy())



X = [input_ids, token_type_ids, attention_mask, word_indices]



# Loads bert model using keras
BERT_PRETRAINED_DIR = '/home/rrevutch/bert/google_research_code_bert_base_uncased'#'../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))
from keras_bert.bert import get_model
from keras_bert.loader import load_trained_model_from_checkpoint
from keras.optimizers import Adam
from keras_bert import extract_embeddings
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda, Concatenate
from keras.models import Model
import re
import codecs



# Setting up logistics
adam = Adam(lr=2e-5,decay=0.01)
maxlen = 50
print('begin_build')
config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, trainable=True, seq_len=maxlen)
model.summary()




#
# Custom tensorflow layers
#
def lambda1(x):

	indices = tf.dtypes.cast(x[:,50,0], tf.int32)
	row_indices = tf.range(tf.shape(indices)[0])
	full_indices = tf.stack([row_indices, indices], axis=1)
	return tf.gather_nd(x, full_indices)




inp = Input(shape=(1,768,))
shrink_layer = Lambda(lambda x: x)(model.get_layer('Encoder-12-FeedForward-Norm').output)
concat_layer = Concatenate(axis=1)([shrink_layer, inp])
cls_layer = Lambda(lambda1, output_shape=(768,))(concat_layer)

model3 = Model(inputs=model.input + [inp],
						outputs=[cls_layer])
model3.compile(loss='mse',
				optimizer=adam)
model3.summary()



'''
# Sample Testing Input
# Tests if custom lambda layer works
input_ids = []
token_type_ids = []
attention_mask = []
s = "Hello there, I fly airplanes."
tokenized_X = tokenizer([s], max_length=50, return_tensors='tf', padding='max_length', truncation=True)
input_ids.append(tokenized_X['input_ids'][0].numpy())
token_type_ids.append(tokenized_X['token_type_ids'][0].numpy())
attention_mask.append(tokenized_X['attention_mask'][0].numpy())
s = "Hello there, I fly airplanes."
tokenized_X = tokenizer([s], max_length=50, return_tensors='tf', padding='max_length', truncation=True)
input_ids.append(tokenized_X['input_ids'][0].numpy())
token_type_ids.append(tokenized_X['token_type_ids'][0].numpy())
attention_mask.append(tokenized_X['attention_mask'][0].numpy())




X = [input_ids, token_type_ids, attention_mask, [[np.array([0]*768)],[np.array([1]*768)]] ]
print(X)
embedding_output = model3.predict(X)
print(embedding_output.shape)
print(embedding_output)
'''






model3.fit(X,
			Y,
			batch_size=8,
			epochs=4)





'''
# TEST BUILT-IN EXTRACT EMBEDDINGS FUNCTION
s = "Hello there, I fly airplanes."
e = extract_embeddings(BERT_PRETRAINED_DIR, [s])
e = e[0]
print(len(e))
print(e[0])
print()
print(e[1])
'''


'''
for i in range(len(sentences)):
	print(i)
	print(sentences[i])
	print(word_indices[i])
	print(sense_mappings[i])
'''



# create separate input which connects to end of layer (it gives output label index to calc loss on)
