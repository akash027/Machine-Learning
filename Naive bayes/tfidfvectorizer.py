from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform(["I like machine learning and clustering algorithms",
                                  "Apples, oranges and any kind of fruits are healthy",
                                  "Is it feasible with machine learning algorithm",
                                  "My family is happy because of the healthy fruits"])

print(tfidf.A)

#to understand ho much one sentence is similar to another
print((tfidf*tfidf.T).A)
