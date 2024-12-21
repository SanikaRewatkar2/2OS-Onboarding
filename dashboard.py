# Specific file for my streamlit dashboard
# Ideally, I would have written the final metrics and RMSEs to another file
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Comparison Dashboard", layout="wide", initial_sidebar_state="expanded")

# First row: columns corresponding to models
col4o, colturbo, colembeddings = st.columns(3)

# Column for gpt-4o-mini 
with col4o:
    st.header("gpt-4o-mini") # change to a metric?
    st.metric("Semantic Similarity RMSE", "0.7221")
    st.metric("BLEU RMSE (relative to score of 1)", "0.5108")

# Column for gpt-3.5-turbo-0125 
with colturbo:
    st.header("gpt-3.5-turbo-0125") # change to a metric?
    st.metric("Semantic Similarity RMSE", "1.2753")
    st.metric("BLEU RMSE (relative to score of 1)", "0.4887")

# Column for text-embeddings-3-small
with colembeddings:
    st.header("text-embeddings-3-small") # change to a metric?
    st.metric("Cosine Similarity RMSE", "1.0384")

# Second row: some nice little graphs!
stsgraph, bleugraph = st.columns(2)
stsgraph.subheader("Semantic Similarity RMSE")
sts_d = {"gpt-4o-mini": [0.7221495689952324], "gpt-3.5-turbo-0125": [1.27534309109353], "text-embeddings-3-small": [1.0383671609301046]}
sts_data = pd.DataFrame(data = sts_d, index=["RMSE"])
stsgraph.bar_chart(data = sts_data, color=["#5C0029", "#73BDBC", "#ACF7C1"], stack=False)

bleugraph.subheader("Average BLEU Score")
bleu_d = {"gpt-4o-mini": [0.686314], "gpt-3.5-turbo-0125": [0.623784]}
bleu_data = pd.DataFrame(data = bleu_d, index=["BLEU"])
bleugraph.bar_chart(data = bleu_data, color=["#5C0029", "#73BDBC"], stack=False)

