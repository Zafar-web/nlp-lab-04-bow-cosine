NLP Lab 04 — Bag of Words & Cosine Similarity

Course Information

Course Title: Natural Language Processing
Topic: Vector Space Modeling — Bag of Words (BoW) & Cosine Similarity
Department: Artificial Intelligence — University of Sindh

Student Information

Student Name: Zafarullah Laghari
Roll Number: 2k24/AI/100

⸻

1. Objective

The objective of this laboratory exercise is to provide hands-on experience in representing textual data as numerical vectors and computing pairwise document similarity.

The main learning outcomes are:

* Understand tokenization, vocabulary extraction, and term-frequency representation using the Bag of Words (BoW) model.
* Implement term feature extraction using Python’s scikit-learn library and CountVectorizer.
* Compute cosine similarity between document vectors.
* Construct a basic information retrieval/search query matching mechanism.

⸻

2. Software and Libraries

The lab was performed using Python 3.8+.

Required libraries:

pip install numpy pandas scikit-learn

Python modules used:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

⸻

3. Task 1 — Bag of Words Matrix Construction

Dataset

Three customer reviews were provided:

1. “The product performance is amazing and fast”
2. “The service was fast and performance was great”
3. “Terrible customer service and bad performance”

CountVectorizer(stop_words="english") was used to remove English stop words and create the Bag of Words representation.

Code

The complete executable code for Task 1 is available in solution.py.

Output

Vocabulary

The extracted vocabulary is:

['amazing', 'bad', 'customer', 'fast', 'great', 'performance',
 'product', 'service', 'terrible']

Bag of Words Matrix

	amazing	bad	customer	fast	great	performance	product	service	terrible
Review 1	1	0	0	1	0	1	1	0	0
Review 2	0	0	0	1	1	1	0	1	0
Review 3	0	1	1	0	0	1	0	1	1

## TASK 1 ScreenShot
![Task 1 output](task1_output.png)
⸻

4. Task 2 — Document Search Engine & Relevance Ranking

Documents

The following four documents were used:

1. Machine learning algorithms analyze structured data effectively
2. Deep learning and neural networks excel at processing unstructured data
3. Natural language processing helps computers understand human language
4. Python is widely used for machine learning and data science

Search Query

machine learning algorithms for data

Method

CountVectorizer was fitted on the documents. The query was then transformed using the same vectorizer.

Cosine similarity was calculated using:

cosine_similarity(query_vector, doc_vectors)

The documents were then sorted from highest similarity score to lowest similarity score.

Output

The first document receives the highest score because it contains several terms from the query, including machine, learning, algorithms, and data.

Document 3 receives a score of 0.0000 because it has no overlapping vocabulary terms with the query.

## Task 2 Screenshot

![Task 2 Output](task2_output.png)
⸻

5. Viva & Reflection Questions

Question 1 — Word Order Invariance

Question

Why does the sentence “Dog bites man” have the exact same Bag of Words representation as “Man bites dog”? How does this impact sentiment analysis?

Answer

Bag of Words only considers the frequency or count of words and ignores their order. Both sentences contain the words dog, bites, and man once. Therefore, they produce the same Bag of Words representation.

This limitation can affect sentiment analysis because word order and context can change the meaning of a sentence. BoW may fail to correctly understand relationships between words, especially when sentences contain negation or different word arrangements.

⸻

Question 2 — Sparsity Issue

Question

What happens to the memory size and density of the BoW matrix when the corpus contains 100,000 unique vocabulary words?

Answer

If the corpus contains 100,000 unique vocabulary words, the Bag of Words matrix will have 100,000 columns. Since an individual document normally contains only a small number of these words, most of the matrix values will be zero.

This creates a sparse matrix. A dense matrix representation can consume a very large amount of memory. Sparse matrix representations store mainly the non-zero values and therefore reduce memory usage and improve efficiency.

⸻

Question 3 — Zero Similarity

Question

Explain why Document 3 in Task 2 receives a Cosine Similarity score of 0.0000 when queried against “machine learning algorithms for data”.

Answer

Document 3 is:

Natural language processing helps computers understand human language

The query is:

machine learning algorithms for data

There are no common vocabulary terms between Document 3 and the query. Therefore, their Bag of Words vectors have no overlapping non-zero terms.

As a result, their dot product is zero, and the cosine similarity is:

0.0000

Therefore, according to the Bag of Words representation, Document 3 has no term-based similarity with the given query.

⸻

6. Conclusion

In this laboratory exercise, textual data was converted into numerical vectors using the Bag of Words model and CountVectorizer.

The vocabulary and term-frequency matrix were successfully constructed for the customer reviews. A basic document search engine was also implemented using cosine similarity to rank documents according to their relevance to a search query.

The experiment also demonstrated important limitations of Bag of Words, including its inability to preserve word order and the sparsity problem in large vocabularies.
