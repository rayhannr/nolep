import numpy as np
import matplotlib.pyplot as plt

import string
import random
import re
import requests
import os
import textwrap

##Createing substitution cipher

#one will act as the key, other as the value
letters1 = list(string.ascii_lowercase)
letters2 = list(string.ascii_lowercase)

true_mapping = {}

#shuffle second set of letters, this is used for the value
random.shuffle(letters2)

#populate the map
#The purpose of zip() is to map the similar index of multiple containers so that they can be used just using as single entity.
for key, value in zip(letters1, letters2):
    true_mapping[key] = value
    
##The language model
    
#initialize markov matrix
M = np.ones((26, 26))

#initial state distribution
pi = np.zeros(26) #the number that i appears as the first letter of a word

#function to update the Markov matrix
def update_transition(ch1, ch2):
    #for example we take bigram 'ae', then i will be 'a' and j will be 'e'
    i = ord(ch1) - 97
    j = ord(ch2) - 97
    M[i, j] += 1 #probability i is followed by j
    
#function to update the initial state distribution
def update_pi(ch):
    i = ord(ch) - 97
    pi[i] += 1 
    
#get the log-probs of a word/token
def get_word_prob(word):
    i = ord(word[0]) - 97 #take the first char in the word
    logp = np.log(pi[i])
    
    for ch in word[1:]:
        j = ord(ch) - 97
        logp += np.log(M[i, j])
        i = j #we need this line so that i represent the previous j in the next iteration
        
    return logp

#get the probability of a sequence of words
def get_sequence_prob(words):
    #if input is a string, split into an array of tokens
    if type(words) == str:
        words = words.split()
        
    logp = 0
    for word in words:
        logp += get_word_prob(word)
        
    return logp

#download the file
if not os.path.exists('moby_dick.txt'):
    print('Downloading moby dick...')
    r = requests.get('https://lazyprogrammer.me/course_files/moby_dick.txt')
    with open('moby_dick.txt', 'w') as f:
        f.write(r.content.decode())
        
#for replacing non-alphabet chars
regex = re.compile('[^a-zA-Z]')

#load in words
for line in open('moby_dick.txt', encoding='utf-8'):
    line = line.rstrip() #strip out any whitespace
    
    #there are blank lines in the file
    if line:
        line = regex.sub(' ', line) #replace all non-alpha chars with space
        
        #split the tokens in the line and lowercase
        tokens = line.lower().split()
        
        for token in tokens:
            #update the model
            
            #first letter
            ch0 = token[0]
            update_pi(ch0)
            
            #other letters
            for ch1 in token[1:]:
                update_transition(ch0, ch1)
                ch0 = ch1

#normalize the probabilities
pi /= pi.sum()
M /= M.sum(axis=1, keepdims=True)

### encode a message

# this is a random excerpt from Project Gutenberg's
# The Adventures of Sherlock Holmes, by Arthur Conan Doyle
# https://www.gutenberg.org/ebooks/1661

original_message = 'I then lounged down the street and found, as I expected, that there was a mews in a lane which runs down by one wall of the garden. I lent the ostlers a hand in rubbing down their horses, and received in exchange twopence, a glass of half-and-half, two fills of shag tobacco, and as much information as I could desire about Miss Adler, to say nothing of half a dozen other people in the neighbourhood in whom I was not in the least interested, but whose biographies I was compelled to listen to.'

#function to encode a message
def encode_message(msg):
    msg = msg.lower()
    msg = regex.sub(' ', msg)
    
    #make the encoded message
    coded_msg = []
    for ch in msg:
        coded_ch = ch #could just be a space
        if ch in true_mapping:
            coded_ch = true_mapping[ch]
        coded_msg.append(coded_ch)
        
    return ''.join(coded_msg)

encoded_message = encode_message(original_message)

#function to decode a message
def decode_message(msg, word_map):
    decoded_msg = []
    for ch in msg:
        decoded_ch = ch #could just be a space
        if ch in word_map:
            decoded_ch = word_map[ch]
        decoded_msg.append(decoded_ch)
    
    return ''.join(decoded_msg)

##run a genetic algorithm to decode the message
    
#this is our initialization point
dna_pool = []
for _ in range(20):
    dna = list(string.ascii_lowercase)
    random.shuffle(dna)
    dna_pool.append(dna)
    
#mutating the dna by swapping two chromosomes
def evolve_offspring(dna_pool, n_children):
    #make n_children per offspring
    offspring = []
    
    for dna in dna_pool:
        for _ in range(n_children):
            copy = dna.copy()
            j = np.random.randint(len(copy))
            k = np.random.randint(len(copy))
            
            #swap
            tmp = copy[j]
            copy[j] = copy[k]
            copy[k] = tmp
            offspring.append(copy)
            
    return offspring + dna_pool

num_iters = 1200
scores = np.zeros(num_iters)
best_dna = None
best_map = None
best_score = float('-inf')

for i in range(num_iters):
    if i > 0:
        #get offspring from the current dna pool
        dna_pool = evolve_offspring(dna_pool, 3)
        
    #calculate score for each dna
    dna2score = {}
    for dna in dna_pool:
        #populate map
        current_map = {}
        for key, value in zip(letters1, dna):
            current_map[key] = value
        
        decoded_message = decode_message(encoded_message, current_map)
        score = get_sequence_prob(decoded_message)
        
        #store it
        #needs to be a string to be a dict key
        dna2score[''.join(dna)] = score
        
        #record the best so far
        if score > best_score:
            best_dna = dna
            best_map = current_map
            best_score = score
    
    #average score for this generation
    scores[i] = np.mean(list(dna2score.values()))
    
    #keep the best 5 dna
    #also turn them back into list of single chars
    sorted_dna = sorted(dna2score.items(), key=lambda x: x[1], reverse=True)
    dna_pool = [list(k) for k, v in sorted_dna[: 5]]
    
    if i % 200 == 0:
        print("iter:", i, "score:", scores[i], "best so far:", best_score)
        
#use best score
decoded_message = decode_message(encoded_message, best_map)

print('Log likelihood of decoded message:', get_sequence_prob(decoded_message))
print('Log likelihood of true message:', get_sequence_prob(regex.sub(' ', original_message.lower())))

#which letters are wrong?
for true, v in true_mapping.items():
    pred = best_map[v]
    if true != pred:
        print('true: %s, pred: %s' % (true, pred))

print('decoded message: \n', textwrap.fill(decoded_message))
print('\nTrue message:\n', original_message)

plt.plot(scores)
plt.show()