# Onboarding Assignment
## Time Breakdown
* Tutorials for Using OpenAI API, HuggingFace Datasets, BLEU Metric, and Streamlit: 1 hour
* Installations and Setup: 0.5 hours
* Running Models and Calculating Metrics: 1.5 hours
* Building Dashboard: 1 hour

## Libraries Used
* HuggingFace Datasets
* Numpy
* nltk
* pandas
* Streamlit

## Process
* The given semantic similarity dataset was loaded from HuggingFace. For the first 10 data points of stsbenchmark-sts, semantic similarity scores were calculated using gpt-4o-mini and gpt-3.5-turbo-0125.
* The RMSE relative to the dataset's base scores was then calculated for each model. 
* For further comparison, cosine similarity scores were calculated using text-embeddings-3-small, and RMSE was calculated. 
* A Spanish-to-English translation dataset, english-spanish-translator, was then loaded from HuggingFace. For the first 10 Spanish sentences in the dataset, gpt-4o-mini and gpt-3.5-turbo-0125 were queried to translate the Spanish sentence into English. 
* BLEU scores for the model output relative to the dataset translation were calculated, and an RMSE relative to a maximum score of 1 was determined for each model.
* Metrics calculated from runs were not persisted, so ideally, some form of data persistence would have been implemented. As such, metrics from one run (RMSE for STS, RMSE for BLEU, average BLEU score) were hardcoded into dashboard.py. 
* The final dashboard displayed metrics for each model, along with a bar graph comparing STS RMSE and average BLEU score for each model.

## Setup
Configure python virtual environment. Run with python model_test.py to calculate semantic similarity RMSE and BLEU scores on a simple translation task for gpt-4o-mini and gpt-3.5-turbo-0125 (API key must be set up as an environment variable). Run with streamlit run dashboard.py to display a dashboard showing model metrics and comparative graphs. (All dashboard metrics were calculated from a previous run and thus might differ from current model_test.py run.)