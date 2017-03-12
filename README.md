# Similarity_Task_Glove
For a given input word we find out the most similar word among the four options given using the Glove word vector for each word.

We use the following metrics for comparing the word vectors : 

1. Cosine similarity
2. Euclidean distance
3. Manhattan distance

Amongst the 40 entries given in the word similarity dataset (i.e. file **word-similarity-dataset**), we report the number of entries which gave the highest score to the correct answer. Also, we report the Mean Reciprocal Rank (MRR) for each distance measure.

We output two files to analyse the results : 

1. simSummary.csv : Each row in the file is in the format `Distance Metric, Number of questions which are correct, Total questions evalauted, MRR`.
2. simOutput.csv : Each row in the file is in the format `File line number, Query word, Option word, Distance metric(C/E/M), Similarity score`.



