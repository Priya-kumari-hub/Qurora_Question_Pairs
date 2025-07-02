
import streamlit as st
import helper 
import pickle

model = pickle.load(open('model/quora_xgb_model.pkl','rb'))

st.header('Duplicate Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    try:
        query = helper.query_point_creator(q1, q2)
        proba = 1-model.predict_proba(query)[0][1]
        
        st.write(f"Raw similarity scores: {helper.string_similarity(q1, q2)}")
        st.write(f"Duplicate probability: {proba:.1%}")
        
        if proba > 0.3:  
            st.header('Duplicate')
            st.success("These questions are duplicates")
        else:
            st.header('Not Duplicate')
            st.error("These questions are not duplicates")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Debug info:", helper.preprocess(q1), helper.preprocess(q2))

