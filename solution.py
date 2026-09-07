#==================================================#
######TASK1 BAG OF WORDS MATRIX CONSTRUCTION########
#==================================================#
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
corpus=[
    "The product performance is amazing and fast",
    "The service was fast and performance was great",
    "Terrible customer service and bad performance"
]
vectorizer=CountVectorizer(stop_words='english')
X=vectorizer.fit_transform(corpus)
vocabulary=vectorizer.get_feature_names_out()
print("Vocabulary:")
print(vocabulary)
df=pd.DataFrame(
    X.toarray(),
    columns=(vocabulary)
)
print("\nBag of words Matrix:")
print(df)



#======================================================#
####TASK2  DOCUMENT SEARCH ENGINE & RELEVANCE RANKING###
#=======================================================#

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Documents
documents = [
    "Machine learning algorithms analyze structured data effectively",
    "Deep learning and neural networks excel at processing unstructured data",
    "Natural language processing helps computers understand human language",
    "Python is widely used for machine learning and data science"
]

# Search query
query = ["machine learning algorithms for data"]

# Create vectorizer
vectorizer = CountVectorizer()

# Fit on documents
doc_vectors = vectorizer.fit_transform(documents)

# Transform query using the same vectorizer
query_vector = vectorizer.transform(query)

# Calculate cosine similarity
similarity_scores = cosine_similarity(query_vector, doc_vectors)[0]

# Create results DataFrame
results = pd.DataFrame({
    "Document": documents,
    "Cosine Similarity": similarity_scores
})

# Sort from highest to lowest
results = results.sort_values(
    by="Cosine Similarity",
    ascending=False
)

# Reset index
results = results.reset_index(drop=True)

print("Ranked Search Results:")
print(results)
