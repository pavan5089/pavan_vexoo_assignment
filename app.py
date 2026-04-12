import streamlit as st
from part1_ingestion.ingestion import sliding_window, build_pyramid, retrieve

st.title("AI Document QA System")

text = st.text_area("Enter Document")

query = st.text_input("Ask Question")

if st.button("Get Answer"):
    chunks = sliding_window(text)
    pyramid = build_pyramid(chunks)
    answer = retrieve(query, pyramid)

    st.write("Answer:", answer)