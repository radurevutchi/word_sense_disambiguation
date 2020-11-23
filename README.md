# Word Sense Disambiguation Research Project 2020

This summer (May - August), I had the opportunity of working with Professor Kemal Oflazer on a new approach to the WSD problem. We attempted to use Google's BERT base and large models to fine-tune for word sense labels. Through this, we took an approach of fine-tuning the word embeddings themselves. This allowed us to closely match CLS tokens to target word (desired sense) embeddings. By doing this, we did not have to fine-tune a separate model for each stem word. It allows us to fine-tune BERT on all senses all at the same time.
<br>
<br>

Concepts Learned:
* Importance and low-cost of fine-tuning
* Tensor calculations vs. Array Calculations
* Wordnet synset and lemma mappings

Skills Learned:
* Fine-Tuning strategies (epochs, task type)
* Creating custom loss function
* Creating custom tensorflow layers (dynamically picks different index to fine tune each sample)
* Using NLTK wordnet functionality


Files
* fine_tune_embeddings.py - Used to fine-tune input BERT model based on a list of senses and mapping sentences
* bert_predict_senses.py - Imports BERT model (original / fine-tuned), extracts sentence embeddings, uses target word embedding on separate MLP to predict sense class. Creates MLP for each sense.
* *.pdf - Reports along the way of the project
