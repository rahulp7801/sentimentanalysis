# -*- coding: utf-8 -*-
""" 
BERT stands for Bidirectional Encoder Representations from Transformers. It is an advanced, pre-trained NLP model that understands language by looking at the context of each word. Because BERT has already been created and pre-trained for us (by Google), it can be fine-tuned for our finance task by adding just one additional output layer—just like that, we can create a state-of-the-art model.

## Goals
In this notebook we will be:

*   Learning and understanding the BERT ML technique and its signficance in NLP
*   Preprocessing data for BERT and training our model
*   Running and finetuning our BERT model with the *PyTorch huggingface* transformer library

# **Milestone 1: Learning and understanding the BERT ML technique and its significance in NLP.**
"""

import os
import torch
import numpy as np
import transformers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#(TODO) dynamically access this with Twitter API and convert to similar csv
# Reading the hard-coded csv with tweets 
def get_finance_train():
  df_train = pd.read_csv("assets/finance_train.csv")
  return df_train

def get_finance_test():
  df_test = pd.read_csv("assets/finance_test.csv")
  return df_test

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

print ("Train and Test Files Loaded as train.csv and test.csv")

LABEL_MAP = {0 : "negative", 1 : "neutral", 2 : "positive"} # Sentiment Analysis
NONE = 4 * [None]
RND_SEED=2020

"""

BERT is a pretrained NLP model that is more contextually aware than anything we've seen before. Specifically, it is an NLP model designed 
to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on what precedes and succeeds a word. Therefore, 
it is better able to capture a word within the context of a sentence. Some other important highlights about BERT are signified below:

1. **Quick Training:**  BERT comes pre-trained, so we can train it on our data very quickly.

2. **Less Data:** Because BERT is pre-trained, we don't need as much training data.

3. **Great Results:** BERT has proven to be an excellent NLP model.

4. **Multi-Lingual Capabilities:** If we want to apply our network to foreign markets, BERT works on more than just English!

**Milestone 2: Preprocessing data for BERT and training our model.**

**Reading in the Datasets**

"""

df_train = get_finance_train()
df_test = get_finance_test()

sentences = df_train["Sentence"].values
labels = df_train["Label"].values

"""

**Tokenization & Input Formatting**

Just as before, we need to break down our input sentences to smaller tokens. BERT expects input sentences to be broken down into individual tokens and the input data needs to follow BERT's pre-defined format.

### BERT Tokenizer ###

The BERT Tokenizer is a two-step process in processing our data. First, we will initialize the tokenizer using our dataset. Thereafter, we will convert the original sentences to tokenized sentences.

#### Step 1 : Create the Tokenizer ####

When using BERT, our text needs to be split into tokens and those tokens need to be mapped to an index using BERT's vocabulary. We will use the `BertTokenizer` module from the transformer library to tokenize our data. 
To do so, call the function `BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)` and save the returned value in a variable named `tokenizer`.

"""

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True) # The actual tokenizer model

'''

tokenizer.vocab_size

^^ finds the number of words in BERT's vocabulary

'''

# Apply the tokenizer to the first element in your training dataset: `sentences[0]`. 

tokenized = tokenizer.tokenize(sentences[0])
print(tokenized)

"""
**Mapping the tokens to our index**

Each token has a corresponding index in the vocabulary. To keep track of the vector representations more concisely, we will convert each of the tokenized lists into a list of indices.

Use the `convert_tokens_to_ids()` function of the `tokenizer` to encode our word tokens into numerical values. Call the command `tokenizer.convert_tokens_to_ids(TOKENIZED_SENTENCE)` passing in the tokenized version of `sentences[0]`
"""

indx = tokenizer.convert_tokens_to_ids(tokenized)
print(tokenized, indx)

"""

When using BERT, we need to modify our input to match the BERT Format. To do so, we will apply these three steps on the input sentences:

1. Add special tokens to the start (`[CLS]`) and end (`[SEP]`) of each sentence.
2. Pad & truncate all sentences to a single constant length.
3. Explicitly differentiate real tokens from padding tokens with the "attention mask".

#### **Special Tokens** ####

These are two tokens we add to the start and end respectively:


*   `[CLS]` - stands for "classification," and is used to identify new sentences.
*   `[SEP]` - stands for "separator," and identifies if a pair of sentences are consecutive in a corpus or not (used for next sentence prediction).

"""

sentences_with_special_tokens = []

for sentence in sentences:
  sentences_with_special_tokens.append(f"[CLS] {sentence} [SEP]")

"""
#### **Tokenize your sentences**

Now we will tokenize our new list of input sentences using our BERT `tokenizer`. Do so by looping through your new `sentences_with_special_tokens` list and apply `tokenizer.tokenize(SENTENCE)` to each sentence. Add each tokenized sentence to a list named `tokenized_texts`.
"""

tokenized_texts = []

for sentence in sentences_with_special_tokens:
  tokenized_texts.append(tokenizer.tokenize(sentence))

"""
#### **Sentence Length** ####

Since our input has sentences of varying length, we need to pad them to be the same length. BERT has the follow constraints on sentence length:
1. All sentences must be padded or truncated to a single, fixed length.
2. The maximum sentence length is 512 tokens.

In order to handle this, every input sequence is padded to pre-defined fixed length with a special `[PAD]` token, which pads the sequence with zeros.

#### **Coding Exercise: Encode tokenized sentences with indices**

Before we properly pad the sentences, we will first map each token to its corresponding index for each tokenized sentence. To do so, we will loop through your `tokenized_texts` variable call the `tokenizer.convert_tokens_to_ids(TOKENIZED_SENTENCE)` 
while passing in your tokenized sentence to convert it to a list of corresponding indices.
"""

input_ids = []

for token in tokenized_texts:
  input_ids.append(tokenizer.convert_tokens_to_ids(token))

"""
#### **Padding input**

Now we will pad our input to ensure that every sequence has the same length. To do so, we will utilize the `keras` function named `pad_sequences()`. Call the function below to properly pad your sequences. We want to pad every sequence to a length of `128`.
"""

input_ids = pad_sequences(input_ids, 
                          maxlen=128, ### Maximum indices we are setting, can be changed
                          dtype="long",
                          truncating="post", 
                          padding="post")

"""
####**Attention Masks**####

An Attention Mask is an array of 1s and 0s indicating which tokens are padding and which are not. The idea behind attention masks is that we do not want the extra padded sequence tokens to contribute to the input features of the machine learning model. 
So, we essentially zero them out to highlight their insignificance before passing them through the model.

Each sequence has a corresponding attention mask. Consider the input sequence below and its corresponding attention mask. We want to create an attention mask for each 
sequence before passing our data through the BERT model.

#### **Create attention masks**

The attention mask for a particular tokenized sequence will have the same exact length and will have a `1.0` at every index that there is a token and a `0.0` at every index that there is a padding. 

Loop through your `input_ids` and for each sequence create a corresponding attention mask and add it to the `attention_masks` list.

"""

attention_masks = []

for input in input_ids:
  attention_masks.append([float(i>0) for i in input]) # List comprehension, if a value exists then say it is 1.0, if it is 0 (added by padding) say it is 0.0 (irrelevant)


"""
**Setting up your Data for BERT**

Finally, we need to split our data and convert it to objects that will work most efficiently with the BERT model training procedure. Specifically, we will split our training data into training and validation sets. 

Additionally, we will convert our data objects into `tensor` s such that we can easily feed the input into the model.

#### **Train/Test Split**

To run machine learning models, we first need to split our dataset into training and validation sections. We can do so by using the `train_test_split()` function. One call to this function returns four separate items:
  1. X_train
  2. X_val
  3. y_train
  4. y_val

"""
X_train, X_val, y_train, y_val = train_test_split(input_ids, labels, test_size = 0.15, random_state= RND_SEED) # RND_SEED = 2020

"""However, we only need to save the first two returned items in variables named `train_masks` and `validation_masks`."""

train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, test_size = 0.15, random_state = RND_SEED) # RND_SEED = 2020

"""In addition to converting our data to `tensor`, `tensorflow` array objects, we will also create a `DataLoader` for our input. A `DataLoader` is simply an object that simplifies and streamlines feeding in data to our model."""

#Convert data to tensors and create DataLoaders
#(TODO) more research on tensors and DataLoaders
train_inputs = torch.tensor(np.array(X_train));
validation_inputs = torch.tensor(np.array(X_val));
train_masks = torch.tensor(np.array(train_masks));
validation_masks = torch.tensor(np.array(validation_masks));
train_labels = torch.tensor(np.array(y_train));
validation_labels = torch.tensor(np.array(y_val));

batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels);
train_sampler = RandomSampler(train_data); # Samples data randonly for training
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size);
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels);
validation_sampler = SequentialSampler(validation_data); # Samples data sequentially
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size);

"""
**Running and finetuning our BERT model with the PyTorch huggingface transformer library.**

To use BERT in our sentiment analysis, we first want to modify BERT to output either a positive, neutral, or negative classification.
 Next, we want to train the model on our dataset so that the entire model, end-to-end, is well-suited to our task. 

The huggingface PyTorch implementation includes a set of interfaces designed for a variety of NLP tasks. These interfaces are all built on 
top of a pre-trained BERT model, and each has different top layers and output types designed to accomodate their specific NLP task. 

(An analogy: imagine BERT is a garden hose; then the huggingface PyTorch interfaces are like different nozzles for the garden hose, so we can use it for "jet," "mist," etc.) 

For our sentiment classification task we will use:

* **BertForSequenceClassification** - This is the normal BERT model with an added single linear layer on top for classification that we will use as a sentence classifier. 
As we feed input data, the entire pre-trained BERT model and the additional untrained classification layer is trained on our specific task. 


The documentation for this can be found [here](https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#bertforsequenceclassification).

We will create a `BertForSequenceClassification` model and save it in a variable named `model`.
"""

# Initialize BertForSequenceClassification model
# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT small model, with an uncased vocab.
    num_labels = 3,    
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states. (unnecessary) (TODO) more research
);

# Given that this a huge neural network, we need to explicity specify 
# in pytorch to run this model on the GPU.
print(f'TORCH AVAULABE {torch.cuda.is_available()}')
model.cuda();

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

"""
**Initialize hyperparameters**

There are two hyperparamters we will consider: **Learning rate** and **Epochs**. Improperly setting these hyperparamters could lead to overfitting or underfitting. 

We will set these values as follows:

1. **Learning rate**: 2e-5
2. **Epochs**: 4

The learning rate parameter comes as part of an optimizer object that we use in our model. 

"""

optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
epochs = 4

"""
**Training Model Steps**

This is what each epoch is doing

Training Steps:
*   Unpack our data inputs and labels from the DataLoader objects
*   Clear out the gradients calculated in the previous pass
*   Forward pass (feed data through network)
*   Backward pass (backpropagation)
*   Update parameters
"""

# TRAINING THE MODEL

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# We'll store training and validation loss, 
# validation accuracy, and timings.
training_loss = []
validation_loss = []
training_stats = []
for epoch_i in range(0, epochs):
    print('aaaahhhh')
    # Training
    print('Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training the model')
    # Reset the total loss for  epoch.
    total_train_loss = 0
    # Put the model into training mode. 
    model.train()
    # For each batch of training data
    for step, batch in enumerate(train_dataloader):
        print('alright')
        # Progress update every 40 batches.
        if step % 20 == 0 and not step == 0:
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

        # STEP 1 & 2: Unpack this training batch from our dataloader. 
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)  
        print('kendrick')
        # STEP 3
        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()    

        # STEP 4
        # Perform a forward pass (evaluate the model on this training batch).
        # It returns the loss (because we provided labels) and 
        # the "logits"--the model outputs prior to activation.
        outputs = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # STEP 5
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # STEP 6
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    # Validation
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("Evaluating on Validation Set")
    # Put the model in evaluation mode
    model.eval()
    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        #Step 1 and Step 2
        # Unpack this validation batch from our dataloader. 
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # The "logits" are the output
            # values prior to applying an activation function like the softmax.
            outputs = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("Validation Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    
    
    print("Validation Loss: {0:.2f}".format(avg_val_loss))
    

    training_loss.append(avg_train_loss)
    validation_loss.append(avg_val_loss)
    # Record all statistics from this epoch.
    
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy
            
        }
    )
    
print("Training complete!")


fig = plt.figure(figsize=(12,6))
plt.title('Loss over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.plot(training_loss, label = "t-loss") # Training Loss
plt.plot(validation_loss, label = "v-loss") # Validation (Testing) Loss

plt.legend()
plt.show()

"""

To be able to assess the accuracy on the test set, we need to carry out the data preprocessing, tokenization, padding, and formatting on the test set saved in `df_test` from earlier.

First, to make this process easier, save the sentences and labels from `df_test` in variables named `test_sentences` and `test_labels`.

"""

test_sentences = df_test["Sentence"].values
test_labels = df_test["Label"].values


"""Now we will format our test input data similarly to how we formatted our training input data. """

# Process and prepare our test data
test_input_ids, test_attention_masks = [], []

# Add Special Tokens
test_sentences = ["[CLS] " + sentence + " [SEP]" for sentence in test_sentences]

# Tokenize sentences
tokenized_test_sentences = [tokenizer.tokenize(sent) for sent in test_sentences]

# Encode Tokens to Word IDs
test_input_ids = [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_test_sentences]

# Pad the inputs
test_input_ids = pad_sequences(test_input_ids, 
                               maxlen=128, 
                               dtype="long",
                               truncating="post", 
                               padding="post")

# Create Attention Masks
for sequence in test_input_ids:
  mask = [float(i>0) for i in sequence]
  test_attention_masks.append(mask)

"""Just as before, we will convert our data to `tensor` and create a `DataLoader` for our inputs."""

# Convert data to tensors and create DataLoaders
batch_size = 32  
test_input_ids = torch.tensor(test_input_ids)
test_attention_masks = torch.tensor(test_attention_masks)
test_labels = torch.tensor(test_labels)
prediction_data = TensorDataset(test_input_ids, test_attention_masks, test_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

""" Evaluate your accuracy on test dataset!"""

# Prediction on test set
print('Predicting labels for {:,} test sentences...'.format(len(test_input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

  
print ('Test Accuracy: {:.2%}'.format(flat_accuracy(logits, label_ids)))
