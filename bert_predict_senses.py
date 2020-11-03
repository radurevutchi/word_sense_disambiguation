'''
Extracts features for each word. Splits in balanced fashion for train/test.
Runs experiments on Clustering and Neural Network.
'''




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
from keras_bert.bert import get_model
from keras_bert.loader import load_trained_model_from_checkpoint
from keras.optimizers import Adam
from keras_bert import extract_embeddings
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda, Concatenate, Activation
from keras.models import Model
import re
import codecs
from transformers import BertTokenizer
import numpy as np
from sklearn.metrics import pairwise_distances as paird
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from google_research_code_bert_base_uncased import tokenization

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# LOADS PRE-TRAINED BERT MODEL USING KERAS-BERT
BERT_PRETRAINED_DIR = '/home/rrevutch/bert/google_research_code_bert_base_uncased'
# Setting up logistics
print('Import Bert Model')
adam = Adam(lr=2e-5,decay=0.01)
MAXLEN_BERT_MODEL = 200
config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, trainable=True, seq_len=MAXLEN_BERT_MODEL)

print('Adding Custom Layers for dynamic target embedding output')
def lambda1(x):
	indices = tf.dtypes.cast(x[:,MAXLEN_BERT_MODEL,0], tf.int32)
	row_indices = tf.range(tf.shape(indices)[0])
	full_indices = tf.stack([row_indices, indices], axis=1)
	return tf.gather_nd(x, full_indices)

def tokenize_sents(sents, word_indices):
	# transformers library
	# Tokenizes and converts to input embeddings
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


	# ALL INPUT TRAINING DATA X data
	tokenized_X = tokenizer(sents, max_length=MAXLEN_BERT_MODEL, return_tensors='np', padding='max_length', truncation=True)
	input_ids = tokenized_X['input_ids']
	token_type_ids = tokenized_X['token_type_ids']
	attention_mask = tokenized_X['attention_mask']
	word_indices = np.array(list(map(lambda x: [x], word_indices)))
	X = [input_ids, token_type_ids, attention_mask, word_indices]


	return X




inp1 = Input(shape=(1,))#(1,768))
inp2 = Lambda(lambda x: tf.map_fn(lambda y: tf.fill((1,768), y[0]), x), output_shape=(1,768))(inp1)
concat_layer = Lambda(lambda x: tf.keras.backend.concatenate([x[0], x[1]],axis=1))([model.layers[-9].output, inp2])
cls_layer = Lambda(lambda1, output_shape=(768,))(concat_layer)
model3 = Model(inputs=model.input + [inp1],
						outputs=[cls_layer])
model3.compile(loss='mse',
				optimizer=adam)

model3.load_weights('finetuning/ws_bert_base_200len_epoch4_layerall_batch16.h5')
model3.summary()



'''
# Sample Testing Input
#1 = -0.0910
#2 = 0.3142
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
s = ["Hello there, I fly airplanes."]*2
tokenized_X = tokenizer(s, max_length=MAXLEN_BERT_MODEL, return_tensors='np', padding='max_length', truncation=True)
input_ids = tokenized_X['input_ids']
token_type_ids = tokenized_X['token_type_ids']
attention_mask = tokenized_X['attention_mask']
target_indices = np.array([[0],[1]])
print(target_indices.shape)

X = [input_ids, token_type_ids, attention_mask, target_indices]
print(X)

embedding_output = model3.predict(X)
print(embedding_output)
'''






# Modify to change directory of train/test data

words_of_interest = ['serve']#,'back','left','right','open']
C_ACC = []
NN_ACC = []
aggregate_senses = []




tokenizer = tokenization.FullTokenizer(
	vocab_file='google_research_code_bert_base_uncased/vocab.txt', do_lower_case=True)
####
# ITERATES THROUGH EACH WORD
####
for word_of_interest in words_of_interest:

	print(word_of_interest.upper())
	print('\n\n')





	####
	# Embeddings/features file
	# IMPORTS DATA FROM ALL DATA FILES
	####

	# Max number of sense to consider for token (most frequent)
	max_senses = 10000
	directory = 'testing_' + word_of_interest + '/'

	# sense mapping index and label files
	sense_index = np.load(directory + 'sense_mappings_index_test.npy', allow_pickle=True)
	sense_label = np.load(directory + 'sense_labels_test.npy', allow_pickle=True)
	sents_tokenized = open(directory + 'sentences_tokenized_test.txt').readlines()





	######
	# Iterates through each sentence of data
	# OBTAINS EMBEDDING LAYERS FROM JSON FILE
	######
	sense_embeddings = {}
	sense_embeddings_target_index = {}
	for si in range(len(sents_tokenized)):

		# Index, label of current sentence
		index0 = sense_index[si]
		label0 = sense_label[si]
		temp_sent =  sents_tokenized[si][:-1]
		xml_tokens_string_bert = ['[CLS]'] + tokenizer.tokenize(temp_sent) + ['[SEP]']

		# Iterates through each token of current sentence
		for i in range(len(xml_tokens_string_bert)):
			if xml_tokens_string_bert[i] == word_of_interest and i in index0:

				# Identifies the label and index
				label_idx = index0.index(i)
				label = label0[label_idx]

				# Ensures token has label
				if label != 'NA' and '_' not in label:

					# MAP EACH EMBEDDING TO DICTIONARY WITH SENSES AS KEYS
					if label in sense_embeddings:
						sense_embeddings[label].append(temp_sent)
					else:
						sense_embeddings[label] = [temp_sent]

					if label in sense_embeddings_target_index:
						sense_embeddings_target_index[label].append(i)
					else:
						sense_embeddings_target_index[label] = [i]






	######
	# Counts frequency per sense and reduces to max_senses variable
	# REMOVES SENSES WITH FREQUENCY OF 1
	#####
	key_count = []
	delete_low_freq = []
	for keyi in sense_embeddings:

	    # by total number of word count in data
	    #key_count.append((keyi, sum(word_senses[keyi].values())))

	    # by number of senses per word
		if len(sense_embeddings[keyi]) > 1:
			key_count.append((keyi, len(sense_embeddings[keyi])))
		else:
			delete_low_freq.append(keyi)

	#Removes senses with frequency 1
	for k in delete_low_freq:
		del sense_embeddings[k]

	# sorts senses by frequency count
	key_count.sort(key=lambda x: x[1], reverse=True)
	if len(key_count) > max_senses:
		key_count_remove = key_count[max_senses:]
		for kk,i in key_count_remove:
			del sense_embeddings[kk]
	print("Number of senses: " + str(len(sense_embeddings)))
	aggregate_senses += list(sense_embeddings.keys()) #All senses across all words





	data = {}
	#all_data = []
	count_instances = 0 # Count frequency over all senses for current word/token

	# ITERATES THROUGH EACH SENSE OF THE GIVEN WORD
	# CONSTRUCTS THE EMBEDDINGS FOR TRAINING DATA
	for k in sense_embeddings.keys():
		data[k] = [] # adds new key to train/test data dictionary
		sense_embeddings[k] = model3.predict(tokenize_sents(sense_embeddings[k], sense_embeddings_target_index[k]))
		# counts occurences for current sense and aggregates them
		print("Count for " + k + ': ' + str(len(sense_embeddings[k])))
		count_instances += len(sense_embeddings[k])

		# Iterates through each embeddings of current sense
		for embed in sense_embeddings[k]:
			vector = embed
			data[k].append(vector)
		embedding_dimension = len(vector)
		data[k] = np.array(data[k])


	print('Total Count: ', count_instances)




	inner_sense_avg = {}
	train_centers = {}
	# CALCULATE DISTANCES for statistics
	for k in data.keys():
		train_centers[k] = np.mean(data[k],axis=0)
		# CALCULATE DISTANCES BETWEEN NEIGHBORS INSIDE SENSE
		distances = paird(data[k], metric='euclidean')
		distances = np.ndarray.flatten(distances)
		distances = np.array(list(filter(lambda x: x != 0,distances)))
		inner_sense_avg[k] = np.mean(distances)
	print(inner_sense_avg)


	sc = []
	for k in train_centers.keys():
		sc.append(train_centers[k])
	sc = np.array(sc)
	print(paird(sc, metric='euclidean'))

	sc = []
	train_centers = {}










	####
	# Process and ONE-HOT encode training data
	# Multi-class, one for each possible sense
	####
	# Use for multi-class problem (each sense gets a class)

	X = []
	Y = []
	for k in data.keys():
		X.append(data[k])
		y = [k]*len(data[k])
		Y.append(y)
	X = np.concatenate((X))
	Yclass = np.concatenate((Y))

	assert(len(X) == len(Yclass))
	enc = LabelBinarizer()
	Y = enc.fit_transform(Yclass)















	# Number of times to replicate experiments (with different random splits)
	fold_acc = []
	num_folds = 3
	for fold in range(num_folds):



		# Creates balanced train test splits
		sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=3)
		# Neural Network and Clustering training data
		for train_index, test_index in sss.split(X,Y):
			x_train, x_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			y_train_class, y_test_class = Yclass[train_index], Yclass[test_index]
			#print("TRAIN:",train_index,"TEST:",test_index)


		print(y_train, y_test)









		####
		# CLUSTERING
		# Cannot be used for binary classification problem
		####
		# Set up data for Clustering algorithm
		num_senses = len(data.keys())
		train_centers = {}
		train_dict = {}
		sc = []

		for i in range(len(y_train_class)):
			if y_train_class[i] in train_dict:
				train_dict[y_train_class[i]].append(x_train[i])
			else:
				train_dict[y_train_class[i]] = [x_train[i]]


		# Finds the average center of each sense group in training data
		sense_to_class = {}
		si = 0
		for k in train_dict.keys():
			train_centers[k] = np.mean(train_dict[k],axis=0)
			sense_to_class[k] = si
			si += 1
		y_test_class = np.array(list(map(lambda x: sense_to_class[x], y_test_class)))

		# Runs clustering algorithm by initializing clusters to cluster centers
		for k in train_centers.keys():
			sc.append(train_centers[k])
		sc = np.array(sc)
		kmeans = KMeans(n_clusters=num_senses,random_state=51, init=sc,n_init=20,max_iter=6000).fit(x_train)
		cluster_acc = 0
		prediction = kmeans.predict(x_test)
		cluster_acc = accuracy_score(prediction, y_test_class)









		####
		# Neural Network for WSD Prediction
		####


		print('##### Initializing NN')
		print('Input Size = ' + str(embedding_dimension))
		print()


		model = Sequential()

		model.add(Dense(units=100,input_dim=embedding_dimension))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		model.add(Dense(units=num_senses)) #num_senses if multi-class method
		model.add(Activation('softmax')) #softmax if multi class method

		opt = optimizers.SGD(lr=0.005)
		model.compile(loss='categorical_crossentropy', #categorical crossent if multi class method
				optimizer=opt,
				metrics=['accuracy'])


		model.fit(x_train,
			y_train,
			epochs=60,
			batch_size=10,
			validation_data=(x_test,y_test))



		#from sklearn.metrics import classification_report
		#Y_test_class = np.argmax(y_test,axis=1)
		y_pred = model.predict_classes(x_test)

		fold_acc.append(model.evaluate(x_test, y_test)[-1])



	C_ACC.append(cluster_acc)
	NN_ACC.append(sum(fold_acc)/num_folds)


print('WORDS:', words_of_interest)
print('CLUSTER accuracies', C_ACC)
print('NEURAL-NET accuracies', NN_ACC)














































'''
# CENTER POINT OF EACH SENSE
sense_center[k] = np.mean(data[k],axis=0)

# CALCULATE DISTANCES BETWEEN NEIGHBORS INSIDE SENSE
distances = paird(data[k], metric='euclidean')
distances = np.ndarray.flatten(distances)
distances = np.array(list(filter(lambda x: x != 0,distances)))
inner_sense_avg[k] = np.mean(distances)
'''


#print(aggregate_senses)



'''
####
# IMPORT WORDNET - SAVE SENSE DEFINITIONS
####
import nltk
from nltk.corpus import wordnet as wn



definition_file = open('definition.txt','w')
sense_definition_file = open('sense_definition.txt','w')

for s in aggregate_senses:
	sense_definition_file.write(s + ',' + wn.lemma_from_key(s).synset().definition()+'\n')
	definition_file.write(wn.lemma_from_key(s).synset().definition() + '\n')

sense_definition_file.close()
definition_file.close()
'''











# ORIGINAL CLUSTERING ALGORITHM TRAINED ON ALL DATA AND TESTED ON ALL DATA

'''
# Creates clustering algorithm
all_data = np.concatenate(all_data)
num_senses = len(sense_center.keys())
sc = []
for k in sense_center.keys():
	sc.append(sense_center[k])
sc = np.array(sc)


# init=sc means we don't generate center points from random, they are mean of each sense
kmeans = KMeans(n_clusters=num_senses,random_state=50, init=sc,n_init=20,max_iter=6000).fit(all_data)



si = 0
cluster_acc = 0
t = 0


print()
print('Cluster assignments:')
# Iterates through each sense
# Accuracy only correct if ini=sc in clustering
for k in data.keys():

	# Accuracy is weighed based on frequency of each sense
	# Predicts cluster for each sense
	prediction = kmeans.predict(data[k])
	print(prediction,list(prediction).count(si)/len(prediction))
	cluster_acc += list(prediction).count(si)/len(prediction)*len(data[k])
	si += 1
	t += len(data[k])
cluster_acc /= t
print('Total acc: ' + str(cluster_acc))
print()

print('Inner sense average distance')
# Calculates distances between cluster midpoints
print(inner_sense_avg)
print()

print('Inter sense distances')
sc = []
for k in sense_center.keys():
	sc.append(sense_center[k])
sc = np.array(sc)
print(paird(sc, metric='euclidean'))
print('\n\n')

'''
