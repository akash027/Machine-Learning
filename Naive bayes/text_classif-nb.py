from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics','sci.med']

training_Data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)

print("\n".join(training_Data.data[0].split("\n")[:10]))

print("target is :", training_Data.target_names[training_Data.target[1]])


#we just count the word occurences
countVectorizer = CountVectorizer()
xTrainCounts = countVectorizer.fit_transform(training_Data.data)

#countVectorizer.vocabulary_.get('software')

#we transform the word occurences into tfidf
# Tfidfvecorizer = CountVectorizer() + TfidfTransformer

tfidfTransformer = TfidfTransformer()
xTrainTfidf = tfidfTransformer.fit_transform(xTrainCounts)


model = MultinomialNB().fit(xTrainTfidf, training_Data.target)


new = ['This has nothing to do with church or religion', 'software engineering is getting hotter nowadays']

xNewCounts = countVectorizer.transform(new)
xNewTfidf = tfidfTransformer.transform(xNewCounts)

predicted = model.predict(xNewTfidf)

for doc, category in zip(new, predicted):
    print('%r ----------> %s'%(doc, training_Data.target_names[category]))