import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
# Replace 'books.csv' with your dataset file
books = pd.read_csv('books.csv')

# Preprocessing the data
books['Summary'] = books['Summary'].str.lower()  # Convert to lowercase

# User input
user_input = input("Enter a plot summary of a book you enjoyed: ").lower()

# Vectorize the summaries and user input using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['Summary'])

# Add the user's input to the TF-IDF matrix
user_tfidf = tfidf.transform([user_input])

# Calculate cosine similarity
similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)

# Get the top 5 similar books
top_matches = similarity_scores[0].argsort()[-5:][::-1]  # Indices of top scores

# Display recommendations
print("\nRecommended Books:")
for index in top_matches:
    print(books.iloc[index]['Title'])
