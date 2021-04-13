
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import os.path, string, math
import pandas as pd
import scipy.signal as signal
import scipy.interpolate as interpolate
import hypertools as hyp #pip install hypertools
import numpy as np
import scipy as sp
import seaborn as sns

# Preparing data for vectorization
def prepare_text (file):
    # initialize dictionary and list
    prepared_dict  = {}
    token_string = []
    # if file isn't empty
    if os.stat(file).st_size != 0:
        with open(file, 'r', encoding='ISO-8859–1') as f:
            # check if file is emtpy here
            lines = list(f.readlines())
            for line in lines:
                #print("line: " + line)
                # tokenize line (time, comma, word)
                tokens = word_tokenize(line)
                #print("tokens: " + str(tokens))

                stop_words = set(stopwords.words('english'))
                # check that value is not a stop word
                #print("tokens[2]: " + str(tokens[2]))
                if tokens[2] not in stop_words:
                    # check that value doesn't contains integers or symbols
                    if tokens[2].isalpha():
                        # add tokens to one tokens list
                        token_string.append(tokens[2])
                        #print("tokens[0]: " + str(tokens[0]))
                        # move decimal point back 3 numbers (ms to s)
                        timepoint = float(tokens[0])/float(1000)
                        #print("timepoint: " + str(timepoint))
                        # rounding decimals numbers to 2
                        rounded = round(timepoint, 2)
                        #print("rounded: " + str(rounded))
                        # create dictionary with time as key and word as value
                        prepared_dict[rounded] = tokens[2]
                        #print(prepared_dict)

            # join tokens as one string
            final_string = ' '.join(token_string)
            # turn dictionary into data frame
            df = pd.DataFrame(index=prepared_dict.keys())
            df['word'] = prepared_dict.values()

            #print("prepared_text" + str(prepared_text))
    # if tokens aren't empty
    if len(token_string) != 0:
        return final_string, df

# combine data and vectorize
def vectorize(path):
    # initialize lists of data frames and document strings
    all_doc_strings = []
    all_frames = []
    print("all docs:")
    print(all_doc_strings)
    # change directory
    os.chdir(path)
    file_number = 1
    # loop through directory
    for file in os.listdir(path):
        if file.endswith('.vtt'):
            print("file number " + str(file_number))
            print("current file: ")
            print(file)
            # if processed file is not empty. i.e returned None
            if prepare_text(file) is not None:
                file_string, frames = prepare_text(file)
            # if not, continue to next file
            else:
                continue
            #print("prepared string: ")
            #print(string)
            # if returned string with file words is not empty
            if file_string:
                # add it to the docs list
                all_doc_strings.append(file_string)
            # if the data frames are not empty
            if not frames.empty:
                # add it to the frames list
                all_frames.append(frames)
            print("all docs now: ")
            print(all_doc_strings)
            print("all framed now: ")
            print(all_frames)
            file_number += 1
    print("all docs:")
    print(all_doc_strings)
    # convert file into data frame
    print(" Starting Vectorization")
    #all_data = pd.DataFrame(list(zip(all_doc_strings)))
    #print("data frame")
    #print(all_doc_strings)
    # set parameters for CountVectorizer
    vectorizer = CountVectorizer(analyzer='word', lowercase=False, min_df=0.1, max_df=0.75)
    # vectorize list of strings - each doc as one string
    docs_vectorized = vectorizer.fit_transform(all_doc_strings)
    print("docs vectorized")
    print(vectorizer.vocabulary_)
    print(docs_vectorized)
    return vectorizer, docs_vectorized, all_frames

def fit_model (docs_vectorized):
    # set model parameters
    lda_model = LatentDirichletAllocation(n_components=10, max_iter=10,learning_method='online')
    # fit model to the vectorized data
    lda_output = lda_model.fit(docs_vectorized)
    print("lda output: ")
    print(lda_model.components_/lda_model.components_.sum(axis=1)[:, np.newaxis])
    return lda_model, lda_output

###

def get_top_words(lda_model, vectorizer, n_words=30):
    vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    top_words = []
    for k in range(lda_model.components_.shape[0]): #for each topic
        top_words.append([vocab[i] for i in np.argsort(lda_model.components_[k, :])[::-1][:n_words]])
    return top_words

def display_top_words(lda_model, vectorizer, n_words=30):
    for k, w in enumerate(get_top_words(lda_model, vectorizer, n_words=n_words)):
        print(f'topic {k}: {",".join(w)}')
###

# function to find the next start index and end index
def find_index(start_index, desired_interval, index_list):
    #print("inside index list with: " + str(start_index) + "," + str(desired_interval) )
    end_index = start_index + 1
    prev_interval = index_list[end_index] - index_list[start_index]
    for i in range(start_index + 2, len(index_list)):
        curr_interval = index_list[i] - index_list[start_index]
        # check if the current or next item in list is closer to the desired value
        if (abs(desired_interval - curr_interval)) < abs(desired_interval - prev_interval):
            prev_interval = curr_interval
            end_index = i
        else:
            #print("Returning end index: " + str(end_index) + " for start index: " + str(start_index) + "and interval: " + str(prev_interval))
            return end_index
    #print("Returning end index: " + str(end_index) + " for start index: " + str(start_index) + "and interval: " + str(prev_interval))
    return end_index


# given dataframe, divide into sliding windows of 5 seconds and find window mid-points
def sliding_windows(dataframe):
    windows = []
    mid_timepoints = []
    # create list of indices in datafrae (timepoints)
    index_list = dataframe.index.values.tolist()
    #print("index list: " + str(index_list))
    #print("index list from index 55 - 65: " + str(index_list[1260:]))
    # create word list from dataframe
    word_list = dataframe['word'].tolist() # list of index values in dataframe
    #print("word list: " + str(word_list))
    start_index = 0
    end_index = 0
    while end_index < len(index_list) - 1:
        end_index = find_index(start_index, 5, index_list)
        #print("start index: " + str(start_index) + ", end index: " + str(end_index))
        # add words between start index and end index in the word list to windows
        windows.append(' '.join(word_list[start_index: end_index]))
        #print("adding to windows between : start index " + str(
                          #index_list[start_index]) + " end index " + str(index_list[end_index]))
        #print("windows now: " + str(windows))
        #print("start i: " + str(index_list[start_index]) + " end i: " + str(index_list[end_index]))
        mid = round(((index_list[start_index] + index_list[end_index])/2),2)
        mid_timepoints.append(mid)
        start_index = find_index(start_index, 1, index_list)
        #print("start index now: " + str(start_index))
    return windows, mid_timepoints




def topic_trajectories (list_of_dataframes, lda_model, vectorizer):
    trajectories = []
    for i in list_of_dataframes:
        windows_list, time_list = sliding_windows(i)
        # vectorize sliding windows
        print(windows_list, time_list)
        # set parameters for CountVectorizer
        vectorized = vectorizer.transform(windows_list)
        print('vectorized: ')
        print(vectorized)
        windows_trajectory = lda_model.transform(vectorized)#documents by K matrix
        # trajectory_data = pd.Dataframe
        print('windows trajectory: ')
        print(windows_trajectory)
        df = pd.DataFrame(index=time_list, data=windows_trajectory, columns=range(windows_trajectory.shape[1]))
        trajectories.append(df)

    resampled_frames = resample_and_smooth(trajectories)


import datetime as dt

def resample_and_smooth(data, n=1000, wsize=51, order=3, min_val=0):
    if type(data) == list:
        return [resample_and_smooth(d, n=n, wsize=wsize, order=order, min_val=min_val) for d in
                data]
    # create matrix of zeros of the same size as data
    data_resampled = np.zeros([n, data.shape[1]])
    # looking at intervals of time

    t0 = data.index[0]
    x = [(t - t0).total_seconds() for t in data.index]
    xx = np.linspace(0, np.max(x), num=n)

    for i in range(data.shape[1]):
        data_resampled[:, i] = signal.savgol_filter(sp.interpolate.pchip(x, data[:, i])(xx), wsize,
                                                    order)
        data_resampled[:, i][data_resampled[:, i] < min_val] = min_val
    return pd.DataFrame(data=data_resampled, index=[dt.timedelta(seconds=x) + t0 for x in xx], columns=data.columns)

### more code from jeremy
  # to visualize resampled data



#x is your list of topic trajectories (resampled to all have N rows and K columns)-- N = number of resampled timepoints, K = number of topics
#assume x is a list of dataframes
def align(x, n_iter=10):
    sns.heatmap(x)
    aligned_x = x
    for n in range(n_iter):
        aligned_x = hyp.align(aligned_x, align='hyper') #aligned_x is a list of length len(x), where each entry is an N by K numpy array
    sns.heatmap(aligned_x)
    return [pd.DataFrame(data=i, index=j.index, columns=j.columns) for i, j in zip(aligned_x, x)]

# x is list of data frames
# x = np.vstack([d.ravel().T for d in resampled_smoothed_video_data_frames])
# #keep labels:
# x = pd.DataFrame(data=x, index=video_names_or_ids)

def cluster(x, n_clusters=5): #x should be a dataframe with 1 row per video and 1 column per timepoint/topic -- e.g. the result of np.ravel(x0.values).T
    clustered_labels = hyp.cluster(x, cluster='KMeans', n_clusters=5)

    clusters = []
    for k in np.unique(clustered_labels):
        inds = np.where(clustered_labels == k)[0] #might need to change clustered_labels to np.array(clustered_labels) in this line
        clusters.append(x.iloc[inds].copy())
    return clusters


    ### another option
    #kmeans = KMeans(n_clusters=5)
    #kmeans.fit(x)


###

path = '/Users/tehuttesfayebiru/PycharmProjects/Thesis/Test/'
vectorizer, docs_vectorized, all_data = vectorize(path)
#print(" vectorized data: ")o
#print(docs_vectorized)
model, modelled_data = fit_model(docs_vectorized)
display_top_words(model, vectorizer)
topic_trajectories(all_data, model, vectorizer)
# print("modelled data: ")
# print(modelled_data)
# pprint(modelled_data)