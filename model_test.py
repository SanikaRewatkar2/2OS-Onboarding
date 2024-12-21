from datasets import load_dataset
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # my API key is saved as the following environment variable
import numpy as np

# Prepare semantic similarity dataset
ds = load_dataset("mteb/stsbenchmark-sts")
ds_sentences = ds.select_columns(["sentence1", "sentence2"])
ds_scores = ds.select_columns(["score"])
ds_sentences = ds_sentences['train'][:10]
ds_scores = ds_scores['train'][:10]
base_scores = []

# Prepare translation dataset
engspan_sentences = load_dataset("kirchik47/english-spanish-translator")
engspan_sentences = engspan_sentences['train'][:10] # just running it on the first 10
#base_translations = []

# Prepare BLEU metric
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


# function to generate semantic similarity score from a given model
def get_semantic_similarity(name_of_model):
    gpt_scores = []
    for i in range(10):
        sen1 = ds_sentences["sentence1"][i]
        sen2 = ds_sentences["sentence2"][i]
        if len(base_scores) < 10:
            base_scores.append(ds_scores["score"][i])
        completion = client.chat.completions.create(
            model=name_of_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who only gives responses consisting of one number."},
                {"role": "user", "content": "Your job is to provide a semantic similarity score between the following sentences. Sentence 1: " + sen1 + " Sentence 2: " + sen2 + " Scores must be decimal values between 0.0 and 5.0, inclusive. Only output a single number."}
            ]
        )
        gpt_scores.append(float(completion.choices[0].message.content))
        print("Base Score: " + str(base_scores[i]) + "; " + name_of_model + " Score: " + str(gpt_scores[i]))
    return gpt_scores

# function to calculate rmse
def get_rmse(base, gpt):
    sts_true = np.array(base)
    sts_pred = np.array(gpt)
    rmse = np.sqrt(np.mean((sts_pred - sts_true)**2))
    return rmse

# function to get cosine similarity scores
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)

# function to get css from embeddings
def get_css_from_embeddings():
    css_scores = []
    for i in range(10):
        sen1 = ds_sentences["sentence1"][i]
        sen2 = ds_sentences["sentence2"][i]
        embedding1 = client.embeddings.create(input=sen1, model="text-embedding-3-small").data[0].embedding
        embedding2 = client.embeddings.create(input=sen2, model="text-embedding-3-small").data[0].embedding
        css = cosine_similarity(embedding1, embedding2) * 5 # scale up
        css_scores.append(float(css))
        print("Base Score: " + str(base_scores[i]) + "; CSS Score: " + str(css_scores[i]))
    return css_scores 

# function to get translations and bleu
def get_translation_and_bleu(name_of_model):
    #gpt_translations = []
    bleu_scores = []
    sf = SmoothingFunction()
    for i in range(10):
        span_sen = engspan_sentences["sentences_es"][i]
        b_t = str(engspan_sentences["sentences_en"][i])
        #base_translations.append(b_t.split())
        b_t_array = b_t.split()
        b_t_input = [b_t_array]
        completion = client.chat.completions.create(
            model=name_of_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Translate the following Spanish text to English, and output only the translation: " + span_sen}
            ]
        )
        g_t = str(completion.choices[0].message.content)
        g_t_array = g_t.split()
        print("Base Translation: " + b_t + "; " + name_of_model + " Translation: " + g_t)
        score = sentence_bleu(b_t_input, g_t_array, smoothing_function=sf.method4)
        print("BLEU Score: " + str(score))
        bleu_scores.append(score)
    return bleu_scores
    

# Stage 1: run gpt-4o-mini on the first ten sentences of semantic similarity dataset
print("Model Tested: gpt-4o-mini")
gpt_scores_4o = get_semantic_similarity("gpt-4o-mini")

# Stage 2: calculate RMSE for base scores vs gpt scores
rmse_sts_4o = get_rmse(base_scores, gpt_scores_4o)
print("RMSE: " + str(rmse_sts_4o))

# Repeat Stages 1 and 2 for embeddings
print("Model Tested: text-embeddings-3-small")
css_scores_embedding = get_css_from_embeddings()
rmse_css_embedding = get_rmse(base_scores, css_scores_embedding)
print("RMSE: " + str(rmse_css_embedding))

# Stage 3: evaluate BLEU metrics on a simple Spanish-to-English translation task
print("Translation Task: gpt-4o-mini")
bleu_scores_4o = get_translation_and_bleu("gpt-4o-mini")

# Base BLEU score is 1, so make an array of all 1s and calculate RMSE
one_arrays = []
for i in range(10):
    one_arrays.append(1.0)
rmse_bleu_4o = get_rmse(one_arrays, bleu_scores_4o)
print("RMSE: " + str(rmse_bleu_4o))


# Repeat it all for another model! I pick... gpt-3.5-turbo-0125

# Stage 1: run gpt-3.5-turbo-0125 on the first ten sentences of semantic similarity dataset
print("Model Tested: gpt-3.5-turbo-0125")
gpt_scores_turbo = get_semantic_similarity("gpt-3.5-turbo-0125")

# Stage 2: calculate RMSE for base scores vs gpt scores
rmse_sts_turbo = get_rmse(base_scores, gpt_scores_turbo)
print("RMSE: " + str(rmse_sts_turbo))

# Stage 3: evaluate BLEU metrics on a simple translation task
print("Translation Task: gpt-3.5-turbo-0125")
bleu_scores_turbo = get_translation_and_bleu("gpt-3.5-turbo-0125")

# Base BLEU score is 1, so make an array of all 1s and calculate RMSE
rmse_bleu_turbo = get_rmse(one_arrays, bleu_scores_turbo)
print("RMSE: " + str(rmse_bleu_turbo))




