import numpy as np
import os

import typing
from typing import Any, Tuple

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import tensorflow_text as tf_text

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys

MODEL=sys.argv[1]
INFILE=sys.argv[2]
OUTFILE=sys.argv[3]
NUMMAX=500

def tf_lower_and_split_punct(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    #text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    #text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)
    
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

#@title Labeled attention plots
def plot_attention(attention, sentence, predicted_sentence,FIGNAME='attn.png'):
  #plt.style.use('fivethirtyeight')
  import seaborn as sns
  sns.set()

  sentence = tf_lower_and_split_punct(sentence).numpy().decode().split()
  predicted_sentence = predicted_sentence.numpy().decode().split() + ['[END]']
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)

  attention = attention[:len(predicted_sentence), :len(sentence)]

  fontdict = {'fontsize': 12}
  ax.matshow(attention, cmap='jet', vmin=0.0)
  ax.set_xticklabels([''] + sentence,fontdict=fontdict,   rotation=0)
  ax.set_yticklabels([''] + predicted_sentence, rotation=90,fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  ax.set_title('attention weights',fontsize=20,y=1.5,color='.5')
  #plt.setp(ax.get_xticklabels(), fontsize=16,y=-.02)
  #plt.setp(ax.get_yticklabels(), fontsize=16)
  ax.set_facecolor((1.0, 0.47, 0.42,0))
  fig.set_facecolor((1,0,1,0))
  ax.spines['bottom'].set_color('.75')
  ax.spines['top'].set_color('.75') 
  ax.spines['right'].set_color('.75')
  ax.spines['left'].set_color('.75')
  ax.grid(color='#dddddd', linestyle='--', linewidth=.2,alpha=.2)
  plt.setp(ax.spines.values(), linewidth=1)
  ax.tick_params(axis='x', colors='.5')    #setting up X-axis tick color to red
  ax.tick_params(axis='y', colors='.5',pad=15)  #setting up Y-axis tick color to black

  ax.set_xlabel('Input',fontsize=20,color='.5',labelpad=20)
  ax.set_ylabel('Output',fontsize=20,color='.5')
  #plt.suptitle('Attention weights')
  plt.savefig(FIGNAME,dpi=300)


indata=np.loadtxt(INFILE).astype(int).astype(str)
s=[]
for i in indata:
    s= np.append(s,' '.join(i[:-1])) # Yi made a change here since the last column of the test data is actually the ground truth values

input_text = tf.constant(s[:NUMMAX])

reloaded = tf.saved_model.load(MODEL)
result = reloaded.tf_translate(input_text)

for i in range(NUMMAX):
#i=0
    plot_attention(result['attention'][i], input_text[i], 
               result['text'][i],FIGNAME=MODEL+'attn'+str(i)+'.png')
    break

if os.path.exists(OUTFILE):
    os.remove(OUTFILE)
f=open(OUTFILE,'a')
for tr in result['text']:
    a=np.array(tr.numpy().decode().split()).astype(int)
    np.savetxt(f,a,fmt='%d' , newline=" ")
    f.write("\n")
f.close()
print()




