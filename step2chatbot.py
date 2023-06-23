import json,os
import wget
import nltk   # nltk.download('punkt')
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from download import download
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

%matplotlib inline
unknown=["I am confused","please try again","use another way","None","Try another way!"]

Lemmatizer=WordNetLemmatizer()
intents=json.load(open("intents6.json",))     # change here only 
words=pickle.load(open("words.pkl","rb"))     # used in line 
classes=pickle.load(open("classes.pkl","rb")) # used in line 41
model=load_model("chatbot.h5")

def clean_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[Lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence):
    sentence_words=clean_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)
def predict_class(sentence):
    bow=bag_of_words(sentence)
    #res=model.predict(np.array([bow]))[0]
    res=model.predict(np.array([bow]),verbose=0)[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    return_list=[]
    for r in results:
        return_list.append({"intent":classes[r[0]],"probability":str(r[1])})  # using classes here
    return return_list

def get_response(intents_list,intents_json):
    tag=intents_list[0]["intent"]
    list_of_intents=intents_json["intents"]
    for i in list_of_intents:
        if i["tag"]==tag:
            result=random.choice(i["responses"])
            break
    #return result # UnboundLocalError: local variable 'result' referenced before assignment
    try:
        return result
    except NameError:
        random.choice(unknown)
        
    
def showVideo(url):
    """
    loc="http://datayyy.com/5h.mp4"
    showVideo(loc)
    """
    outfile="abc@xx99ffkk.mp4"
    wget.download(url,outfile)
    os.system(outfile)
def showImage(url):
    from matplotlib.pyplot import imshow
    """
    infile="http://datayyy.com/images/fun.png"
    showImage(infile)
    """
    outfile="abc_xx99ffkk.png"
    #wget.download(url,outfile)
    download(url,outfile,replace=True,verbose=False)
    #plt.imshow(outfile)
    #aa=Image.open(outfile)
    #aa.show()
    pil_im = Image.open(outfile, 'r')
    imshow(np.asarray(pil_im))
# Figures now render in the Plots pane by default. To make them also appear inline in the Console, uncheck "Mute Inline Plotting" under the Plots pane options menu.       
   
message="1"
print("\n\n")
print("* -----------------------------------------------------*")
print("*  This Chatbot tutor explains Time Value of Money     *")
print("*     1) type 'sample questions' to see some questions *")
print("*     2) type menu to see 5 formulae                   *")
print("*     3) type quit to exit                             *")
print("* -----------------------------------------------------*\n\n")
print(" ------------------------------------------------")
while message!='quit':
    message=input("    You: ")
    if message=="quit":
        print("\n\n* ------------------------------------------------------------*")
        print("*    Bot: Thank you for using Chatbot Tutor, see you next time!!  *")
        print("* ------------------------------------------------------------*\n\n")
    else:
        ints=predict_class(message)
        response=get_response(ints,intents)    # intents used here @!!!!!!!
        print("    Bot:",response)
        print(" ------------------------------------------------")
 


"""
WARNING:tensorflow:5 out of the last 18 calls to <function Model.make_predict_function.<locals>.predict_function 
at 0x000002044C302290> triggered tf.function retracing. Tracing is expensive and the excessive number of 
tracings could be due to 

(1) creating @tf.function repeatedly in a loop,
(2) passing tensors with different shapes, 
(3) passing Python objects instead of tensors. 

For (1), please define your @tf.function outside of the loop. 
For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. 
For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and 
https://www.tensorflow.org/api_docs/python/tf/function for  more details.


keyword="video"

a=intents["intents"]
a[1]['tag']
Out[136]: 'age'

a[1]['tag']=='age'
Out[137]: True

a[1]['tag']=='Age'
Out[138]: False
a[1]['tag'].lower()=='Age'.lower()
Out[139]: True


"""






