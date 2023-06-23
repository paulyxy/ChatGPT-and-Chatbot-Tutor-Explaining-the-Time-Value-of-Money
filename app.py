import json
import nltk
import math
import pickle
import random
import numpy as np
import streamlit as st
import numpy_financial as npf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide")

st.title("Chatbot Tutor to explain the concept of Time Value of Money") 
st.write("By Dr. Yan, v1.0, 6/18/2023")
a1=" type 'sample questions' to see some questions; "
a2=" type menu to see 5 formulae;"
a3=" type quit to exit"
with st.expander("Click here to see/hide a simple explanation."):
    st.write(a1+a2+a3)
        
col1,col2,col3=st.columns([1.1,1,1.2])

Lemmatizer=WordNetLemmatizer()
intents=json.load(open("C:/yan/teaching/chatbot/code/one_set/intents5.json",))      # change here only 
words=pickle.load(open("C:/yan/teaching/chatbot/code/one_set/words.pkl","rb"))
classes=pickle.load(open("C:/yan/teaching/chatbot/code/one_set/classes.pkl","rb"))
model=load_model("C:/yan/teaching/chatbot/code/one_set/chatbot.h5")
#
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
#
def predict_class(sentence):
    bow=bag_of_words(sentence)
    #res=model.predict(np.array([bow]))[0]
    res=model.predict(np.array([bow]),verbose=0)[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    return_list=[]
    for r in results:
        return_list.append({"intent":classes[r[0]],"probability":str(r[1])})
    return return_list
#
def get_response(intents_list,intents_json):
    tag=intents_list[0]["intent"]
    list_of_intents=intents_json["intents"]
    for i in list_of_intents:
        if i["tag"]==tag:
            result=random.choice(i["responses"])
            break
    return result
col1.write("Part I: Chatbot tutor for your questions")
with col1.expander("Click here to see/hide five formulae"):    
    st.latex(r'''FV = PV(1+R)^n''')
    st.latex(r'''PV= \frac{FV}{(1+R)^n}''')
    st.latex(r'''PV(perpetuity)= \frac{C}{R}''')
    st.latex(r'''PV(annuity)=\frac{C}{R}[1-\frac{1}{(1+R)^n}]''')
    st.latex(r'''FV(annuity)=\frac{C}{R}[(1+R)^n-1]''')
#message=st.text_input("You:")
done=0
with col1.expander("Results: click here to show the conversation:"):
     #if message is not None:
    #st.write(message)
    message="-9"
    message=st.text_input("You:")
    ints=predict_class(message)
    response=get_response(ints,intents)
    if len(message)==0:
        response="Enter your question above, please."
    #st.write("    You:",message)
    st.write("    Bot:",response)
    done=1
    if done!=0:
        st.balloons()

col2.write("Part II: Financial Calculator for five functions:")
#with col2.expand("Click here to choose:"):   
choice=col2.selectbox("Choose one rom the list:",("None","pv(one fv)","fv()","pv(perpetuity)","pv(annuity)","fv(annuity)"))
if choice=="pv(one fv)":
    col2.latex(r'''PV= \frac{FV}{(1+R)^n}''')
    fv=col2.number_input("Input fv (future value):")
    r=col2.number_input("Input R (discount rate):")
    n=col2.number_input("Input n (number of periods):")
    pv=fv/(1+r)**n
    with col2.expander("Click here to see/hide your result"):    
         st.write("PV=", pv)
elif choice=="fv()":
    col2.latex(r'''FV = PV(1+R)^n''')
    pv=col2.number_input("Input pv (present value):")
    r=col2.number_input("Input R (discount rate):")
    n=col2.number_input("Input n (number of periods):")
    fv=pv*(1+r)**n
    with col2.expander("Click here to see/hide your result"):    
         st.write("FV=", fv)
    pv=col2.number_input("Input pv:")
elif choice=="pv(perpetuity)":
    col2.latex(r'''PV(perpetuity)= \frac{C}{R}''')
    c=col2.number_input("Input C (cash flow):")
    r=col2.number_input("Input R (discount rate):",0.001)
    pv=c/r
    with col2.expander("Click here to see/hide your result"):    
         st.write("PV(perpetuity)=",pv)
elif choice=="pv(annuity)":
    col2.latex(r'''PV(annuity)=\frac{C}{R}[1-\frac{1}{(1+R)^n}]''')
    c=col2.number_input("Input C (cash flow):")
    r=col2.number_input("Input R (discount rate):",0.001)
    n=col2.number_input("Input n (number of periods):")
    pv=c/r*(1-1/(1+r)**n)
    with col2.expander("Click here to see/hide your result"):    
         st.write("PV(annuity)=", pv)
elif choice=="fv(annuity)":
    col2.latex(r'''FV(annuity)=\frac{C}{R}[(1+R)^n-1]''')
    c=col2.number_input("Input C (cash flow):")
    r=col2.number_input("Input R (discount rate):",0.001)
    n=col2.number_input("Input n (number of periods):")
    fv=c/r*((1+r)**n-1)
    with col2.expander("Click here to see/hide your result"):    
        st.write("FV(annuity)=", fv)    
else:
    col2.write("Click above entry to choose one function.")
#    
col3.write("Part III: Mimic Excel five functions:")
with col3.expander("Click here to see/hide a simple explanation"):    
    st.write("For the following general formula, we have five variables: PV,FV, R, C and n.")
    st.latex(r'''PV=PV(one FV)+PV(annuity)=\frac{FV}{(1+R)^n}+\frac{C}{R}[1-\frac{1}{(1+R)^n}]''')
    st.write(" Note that for a given set of 4 values, we can estimate number 5 (see below).")
with col3.expander("Click here to see/hide the inputs of 5 Excel functions"):    
    st.write("Mimic 5 Excel functions:pv(),fv(),pmt(), rate(), and nper()")
    st.latex(r'''=pv(rate,nper,pmt,[fv],[type])''')
    st.latex(r'''=fv(rate,nper,pmt,[pv],[type])''')
    st.latex(r'''=pmt(nper,pmt,pv,[fv],[type])''')
    st.latex(r'''=rate(nper,pmt,pv,[fv],[type])''')
    st.latex(r'''=nper(rate,pmt,pv,[fv],[type])''')
#
choice=col3.selectbox("Choose one rom the list:",("None","pv()","fv()","pmt()","rate()","nper()"))
if choice=="pv()":
    col3.latex(r'''=pv(rate,nper,pmt,[fv],[type])''')
    rate=col3.number_input("Input rate (discount rate):")
    nper=col3.number_input("Input n (number of periods):")
    pmt=col3.number_input("Input pmt (cash flow):")
    fv=col3.number_input("Input fv (future value, optional):",0)
    myType=col3.number_input("Input type (optional):",0)
    pv=npf.pv(rate,nper,pmt,fv,myType)
    with col3.expander("Click here to see/hide your result"):    
        st.write("PV=", pv)
elif choice=="fv()":
    col3.latex(r'''=fv(rate,nper,pmt,[pv],[type])''')
    rate=col3.number_input("Input R (discount rate):")
    nper=col3.number_input("Input n (number of periods):")
    pmt=col3.number_input("Input pmt (cash flow):")
    fv=col3.number_input("Input pv (present value):",0.0)
    myType=col3.number_input("Input type (optional):",0)
    fv=npf.pv(rate,nper,pmt,pv,myType)
    with col3.expander("Click here to see/hide your result"):    
        st.write("FV=", fv)
elif choice=="pmt()":
    st.latex(r'''=pmt(rate,nper,pv,[fv=0],[type=0])''')
    rate=col3.number_input("Input rate (discount rate):")
    nper=col3.number_input("Input nper (number of periods):")
    pv=col3.number_input("Input pv (present value):")
    fv=col3.number_input("Input fv (present value):",0.0)
    myType=col3.selectbox("Input type (optional):",(0,1))
    pmt=npf.pv(rate,nper,pv,fv,myType)
    if myType==1:
        pmt=pmt*(1+rate)
    with col3.expander("Click here to see/hide your result"):    
        st.write("pmt=", pmt)
elif choice=="rate()":
    col3.latex(r'''=rate(nper,pmt,pv,[fv],[type])''')
    nper=col3.number_input("Input nper (number of periods):")
    pmt=col3.number_input("Input pmt (cash flow):")
    pv=col3.number_input("Input pv (present value):")
    fv=col3.number_input("Input fv (present value):")
    myType=col3.selectbox("Input type (optional):",(0,1))
    rate=npf.rate(nper,pmt,pv,fv,myType)
    if math.isnan(rate):
        rate="#NUM!   (check the signs of your pv, fv, and pmt)"
    with col3.expander("Click here to see/hide your result"):    
        st.write("rate=", rate)
elif choice=="nper()":
    col3.latex(r'''=nper(rate,pmt,pv,[fv],[type])''')
    rate=col3.number_input("Input rate (discount rate):")
    pmt=col3.number_input("Input pmt (cash flow):")
    pv=col3.number_input("Input pv (present value):")
    fv=col3.number_input("Input fv (present value):")
    myType=col3.selectbox("Input type (optional):",(0,1))
    if pmt==0 and np.sign(pv*fv)==1:
        nper="Check the sign of your inputs: pmt,pv, and fv"
        break
    if pmt==0:
        pmt=1e-7
    nper=npf.nper(rate,pmt,pv,fv,myType)
    with col3.expander("Click here to see/hide your result"):    
        st.write("nper=",nper)
else:
     col3.write("Click above entry to choose one function.")
     #       
        
  
    

    
  
