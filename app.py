import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sbert_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

def precompute_embeddings(df, file_path):
    if not os.path.exists(file_path):
        st.write("Computing and saving embeddings...")
        embeddings = get_sbert_embeddings(df['combined_text'].tolist())
        torch.save(embeddings, file_path)
    else:
        st.write("Loading precomputed embeddings...")
    return torch.load(file_path)

def search_courses(query, df, course_embeddings, top_k=5):
    query_embedding = get_sbert_embeddings([query])[0]

    cosine_similarities = util.pytorch_cos_sim(query_embedding, course_embeddings)[0]

    results = [
        (df.iloc[i]['S.NO'], df.iloc[i]['TITLE'], df.iloc[i]['DESCRIPTION'], score.item()) 
        for i, score in enumerate(cosine_similarities)
    ]

    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)

    unique_results = []
    seen_ids = set()
    for course in sorted_results:
        course_id = course[0]
        if course_id not in seen_ids:
            unique_results.append(course)
            seen_ids.add(course_id)
        if len(unique_results) == top_k:
            break
    
    return unique_results

st.title("Course Search with Sentence-BERT")

df = pd.read_csv("cleaned_courses.csv")
df['combined_text'] = (df['TITLE'] + " " + df['DESCRIPTION']).str.lower().str.strip()

embeddings_file = "course_embeddings.pt"
course_embeddings = precompute_embeddings(df, embeddings_file)

query = st.text_input("Enter a topic or keyword to search for courses:")

if query:
    results = search_courses(query, df, course_embeddings)
    
    if results:
        st.write(f"Results for query: '{query}'")
        for course_id, title, description, score in results:
            st.markdown(f"**Course ID**: {course_id}")
            st.markdown(f"**Title**: {title}")
            st.markdown(f"**Description**: {description}")
            st.markdown(f"**Similarity Score**: {score:.4f}")
            st.markdown("---")
    else:
        st.write("No relevant courses found.")