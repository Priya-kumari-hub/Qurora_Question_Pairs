
import streamlit as st
import helper 
import pickle

model = pickle.load(open('model/quora_xgb_model.pkl','rb'))

st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1, q2)
    proba = model.predict_proba(query)[0][1]  # Get probability of being duplicate
    
    st.write(f"Similarity confidence: {proba:.1%}")
    if proba > 0.45:  # Adjustable threshold
        st.header('Duplicate')
        st.success("These questions are duplicates")
    else:
        st.header('Not Duplicate')
        st.error("These questions are not duplicates")


