
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Pre-processing
def prepare_text (lines):
    prepared_text = []
    for line in lines:
        # tokenize line
        tokens = word_tokenize(line)
        # remove stopwords
        stop_words = set(stopwords.words('english'))
        removed_stops = [w for w in tokens if w not in stop_words]
        prepared_text.append(removed_stops)
    return prepared_text


def vectorize(data):
    vectorizer = CountVectorizer(analyzer='word', min_df=10, token_pattern='[a-zA-Z0-9]{3,}')
    data_vectorized= vectorizer.fit_transform(data)
    return data_vectorized
# preliminary analysis

def fit_model ():
    # loop through directory
    lda_model = LatentDirichletAllocation(n_components=100, max_iter=10,learning_method='online')
    lda_output = lda_model.fit_transform(data_vectorized)
    return lda_output

# compute sliding widow durations
# sliding window durations ?
def parse_windows(textlist, wsize):
    windows = []
    window_bounds = []
    for ix in range(1, wsize):
        start, end = 0, ix
        window_bounds.append((start, end))
        windows.append(' '.join(textlist[start:end]))

    for ix in range(len(textlist)):
        start = ix
        end = ix + wsize if ix + wsize <= len(textlist) else len(textlist)
        window_bounds.append((start, end))
        windows.append(' '.join(textlist[start:end]))

    return windows, window_bounds

# computing topic vectors using sliding window durations

# time point by topic matrices
# Resample