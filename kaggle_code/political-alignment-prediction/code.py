import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import os
from nltk import TweetTokenizer

os.mkdir('../output')

class CustomTokenizer:  # collection class of different tokenizers
    def tweet_identity(arg):
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        return tokenizer.tokenize(arg)


df = pd.read_csv('../input/train.csv', sep=',', encoding="utf-8-sig")
df = df.dropna()

X = df.drop(['party', 'Id'], axis=1)
y = df['party']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

numeric_features = ['retweet_count', 'favorite_count']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])

text_features = 'text'
text_transformer = FeatureUnion([
    ('tfidf',
     TfidfVectorizer(tokenizer=CustomTokenizer.tweet_identity, lowercase=False, analyzer='word', ngram_range=(1, 5),
                     min_df=1)),
    ('char',
     TfidfVectorizer(tokenizer=CustomTokenizer.tweet_identity, lowercase=False, analyzer='char', ngram_range=(1, 5),
                     min_df=1))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('txt', text_transformer, text_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('clf', SGDClassifier(loss='hinge', random_state=42, max_iter=50, tol=None)),
                      ])

## Test
# clf.fit(X_train, y_train)
# y_predicted = clf.predict(X_test)
#
# precision = sklearn.metrics.precision_score(y_test, y_predicted, average="macro")
# recall = sklearn.metrics.recall_score(y_test, y_predicted, average="macro")
# f1score = sklearn.metrics.f1_score(y_test, y_predicted, average="macro")
# accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)
#
# print(accuracy, f1score)

## Output
clf.fit(X, y)

df = pd.read_csv('../input/test.csv', sep=',', encoding="utf-8-sig")
df = df.fillna({'text': '', 'retweet_count': 0, 'favorite_count': 0})

X = df.drop('Id', axis=1)

y_predicted = clf.predict(X)
df_result = pd.DataFrame(y_predicted, columns=['Prediction'])
df_result.to_csv('../output/out.csv', index_label='Id')
