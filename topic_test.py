
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import os.path, nltk,string, math, json, pickle
import pandas as pd
import scipy.signal as signal
import scipy.interpolate as interpolate
import hypertools as hyp #pip install hypertools
import numpy as np
import scipy as sp
import seaborn as sns
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import datetime as dt
N_Errors = 0

custom_stops = {"0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't","dont",  "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "itll", "its", "it's", "itself", "iv", "i've", "ive","ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thatll'","thatll","thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "thing","think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're","weren", "werent", "weren't", "we've", "weve","what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y","yeah", "y2", "yes", "yet", "yj", "yl","r", "you", "hes", "youd", "you'd", "you'll", "youll","your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"}
#custom_sets = set(custom_stops)

stop_words = set(stopwords.words('english'))
#print(stop_words)
stop_words.update(custom_stops)
#print("stop words ",stop_words)

# helper function to map treebank tag to wordnet tag for lemmatization
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Preparing data for vectorization
def prepare_text (file):
    # initialize dictionary and list
    #print(file)
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

                # check that value is not a stop word
                #print("tokens[2]: " + str(tokens[2]))
                #print(tokens[2] , tokens[2] in stop_words)
                if tokens[2] in stop_words:
                    continue
                else:
                    # check that value doesn't contains integers or symbols
                    if tokens[2].isalpha():
                        #print(tokens[2] in stop_words)
                        # add tokens to one tokens list
                        # pos tagging the words (tokens)
                        #print('tokens[2] ', tokens[2])
                        tagged = nltk.pos_tag([tokens[2]])
                        #print('tagged ', tagged)
                        word_lemmatizer = WordNetLemmatizer()
                        # lemmatize line by passing word and corresponding tag (call helper function)
                        lemmatized = word_lemmatizer.lemmatize(tagged[0][0], get_wordnet_pos(tagged[0][1]))
                        #print('lemmatized ', lemmatized)
                        token_string.append(lemmatized)
                        #print("tokens[0]: " + str(tokens[0]))
                        # move decimal point back 3 numbers (ms to s)
                        timepoint = float(tokens[0])/float(1000)
                        #print("timepoint: " + str(timepoint))
                        # rounding decimals numbers to 2
                        rounded = round(timepoint, 2)
                        #print("rounded: " + str(rounded))
                        # create dictionary with time as key and word as value
                        prepared_dict[rounded] = lemmatized
                        #print(prepared_dict)

            #print( "before removal ",token_string)

            removed_tokens = [w for w in token_string if not w in stop_words]
            #print("after removal ", removed_tokens)
            keys = list(prepared_dict.keys())
            values = list(prepared_dict.values())
            removed_dict =  {k: a for k, a in zip(keys, values) if not a in stop_words}
            # join tokens as one string
            final_string = ' '.join(removed_tokens)
            # turn dictionary into data frame
            df = pd.DataFrame(index=removed_dict.keys())
            df['word'] = removed_dict.values()


            #print("prepared_text" + str(prepared_text))
    # if tokens aren't empty
    if len(token_string) != 0:
        return final_string, df

# combine data and vectorize
def vectorize(Actual__path):
    # initialize lists of data frames and document strings
    all_doc_strings = []
    file_data_dictionary = {}
    #print("all docs:")
    #print(all_doc_strings)
    # change directory
    os.chdir(Actual__path)
    file_number = 1
    # loop through directory
    for file in os.listdir(Actual__path):
        if file.endswith('.vtt') & os.stat(file).st_size != 0:
            #print("file number " + str(file_number))
            #print("current file: ")
            #print(file)
            # if processed file is not empty. i.e returned None
            #if prepare_text(file) is not None:
            prepared_text = prepare_text(file)
            pass
            if prepared_text is None:
                continue
            else:
                file_string, frames = prepared_text
                #print("file string ", file_string)
                #print("prepared string: ")
                #print(string)
                # if returned string with file words is not empty
                if file_string:
                    # add it to the docs list
                    all_doc_strings.append(file_string)
                # if the data frames are not empty
                if not frames.empty:
                    # add it to the frames list
                    # add file name and frame to dictionary file
                    file_data_dictionary[file[5:]] = frames
                #print("all docs now: ")
                #print(all_doc_strings)
                #print("all framed now: ")
                #print(all_frames)
                file_number += 1
    #print("all docs:")
    #print(all_doc_strings)
    #print("dictionary ", file_data_dictionary)
    # convert file into data frame
    #print(" Starting Vectorization")
    #all_data = pd.DataFrame(list(zip(all_doc_strings)))
    #print("data frame")
    #print(all_doc_strings)
    # set parameters for CountVectorizer
    vectorizer = CountVectorizer(analyzer='word', lowercase=False, min_df=0.1, max_df=0.9)
    # vectorize list of strings - each doc as one string
    docs_vectorized = vectorizer.fit_transform(all_doc_strings)
    #print("docs vectorized")
    #print(vectorizer.vocabulary_)
    #print(docs_vectorized)
    return vectorizer, docs_vectorized, file_data_dictionary

def fit_model (docs_vectorized):
    # set model parameters
    lda_model = LatentDirichletAllocation(n_components=50, max_iter=10,learning_method='online')
    # fit model to the vectorized data
    lda_output = lda_model.fit(docs_vectorized)
    print("lda output: ")
    print(lda_model.components_/lda_model.components_.sum(axis=1)[:, np.newaxis])
    return lda_model, lda_output

###

def get_top_words(lda_model, vectorizer, n_words=10):
    vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    top_words = []
    for k in range(lda_model.components_.shape[0]): #for each topic
        top_words.append([vocab[i] for i in np.argsort(lda_model.components_[k, :])[::-1][:n_words]])
    return top_words

def display_top_words(lda_model, vectorizer, n_words=10):
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
    # create list of indices in dataframe (timepoints)
    index_list = dataframe.index.values.tolist()
    #print("index list: " + str(index_list))
    #print("index list from index 55 - 65: " + str(index_list[1260:]))
    # create word list from dataframe
    word_list = dataframe['word'].tolist() # list of index values in dataframe
    #print("word list: " + str(word_list))
    start_index = 0
    end_index = 0
    while end_index < len(index_list) - 1:
        end_index = find_index(start_index, 60, index_list)
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



# pass in dictionary instead?
def topic_trajectories(dataframes_dictionary, lda_model, vectorizer, **kwargs):
    def df2trajectory(df, **kwargs):
        windows_list, time_list = sliding_windows(df)

        # # vectorize sliding windows
        # print(windows_list, time_list)
        # # set parameters for CountVectorizer
        vectorized = vectorizer.transform(windows_list)
        # print('vectorized: ')
        # print(vectorized)
        windows_trajectory = lda_model.transform(vectorized)#documents by K matrix
        # trajectory_data = pd.Dataframe
        # print('windows trajectory: ')
        # print(windows_trajectory)
        #print("w ", windows_trajectory.shape,windows_trajectory)
        #print("names ", dataframes_dictionary.keys())
        #plot_variance(windows_trajectory, dataframes_dictionary.keys())
        return resample_and_smooth(pd.DataFrame(index=time_list, data=windows_trajectory), **kwargs)

    return {k: df2trajectory(d, **kwargs) for k, d in dataframes_dictionary.items()}

# t_101 = topic_trajectories(df_dict, lda_model, vectorizer, wsize=101)
# t_201 = topic_trajectories(df_dict, lda_model, vectorizer, wsize=201)

def align_dictionary(dicts, n_iter=10):
    keys = list(dicts.keys())
    values = list(dicts.values())
    #values = pd.concat(dicts)
    # constant_filter = VarianceThreshold()
    # constant_filter.fit(values)
    aligned_values = values.copy()
    # for v in aligned_values:
    #     sns.heatmap(v)

    for n in range(n_iter):
        aligned_values = [pd.DataFrame(data=d, index=values[i].index) for i, d in enumerate(hyp.align(aligned_values, align='hyper'))]
    print('pass algin')
    return {k: a for k, a in zip(keys, aligned_values)} #returns dictionary in the same format as dicts, but where trajectories are aligned

def stack_dictionaries(dicts): #dicts should be *aligned dictionaries*
    stacked_trajectories = np.vstack([d.values.ravel() for d in list(dicts.values())])
    print(f'stacked_trajectories shape: {stacked_trajectories.shape}')
    print(f'dictionary keys: {dicts.keys()}')

    return pd.DataFrame(data=stacked_trajectories, index=list(dicts.keys()))

# x = np.vstack([d.ravel().T for d in aligned_frames])4
# #keep labels:
# x = pd.DataFrame(data=x, index=video_names_or_ids)
def resample_and_smooth(data, n=1000, wsize=501, order=3, min_val=0):
    global N_Errors
    try:
        if type(data) == list:
            return [resample_and_smooth(d, n=n, wsize=wsize, order=order, min_val=min_val) for d in
                    data]
        # create matrix of zeros of the same size as data
        data_resampled = np.zeros([n, data.shape[1]])
        # looking at intervals of time
        t0 = data.index[0]
        x = [(t - t0) for t in data.index.values]
        x = np.array(x)
        # equally spaced elements with end point the maximum time in list x and n number of elements
        xx = np.linspace(0, np.max(x), num=n)

        for i in range(data.shape[1]):
            data_resampled[:, i] = signal.savgol_filter(sp.interpolate.pchip(x, data.values[:, i])(xx), wsize,
                                                        order)
            data_resampled[:, i][data_resampled[:, i] < min_val] = min_val
        #print("done with resampling")

        return pd.DataFrame(data=data_resampled, index=[x + t0 for x in xx], columns=data.columns)
    except:
        N_Errors += 1
        print("Error: ",data.shape, N_Errors)  # and some other info about the data if you want...
        if N_Errors > 20:
            raise RuntimeError("too many transcripts errored")

### more code from jeremy
  # to visualize resampled data



#x is your list of topic trajectories (resampled to all have N rows and K columns)-- N = number of resampled timepoints, K = number of topics
#assume x is a list of dataframes

# def align(x, n_iter=10):
#
#
#     aligned_x = x
#     for n in range(n_iter):
#         aligned_x = hyp.align(aligned_x, align='hyper') #aligned_x is a list of length len(x), where each entry is an N by K numpy array
#     import matplotlib.pyplot as plt
#
#     for file in range(5):
#         sns.heatmap(x[file])
#         # fig = plt.figure(figsize=(15, 15))
#         plt.show()
#         plt.clf()
#
#         sns.heatmap(aligned_x[file])
#         plt.show()
#         plt.clf()
#     return [pd.DataFrame(data=i, index=j.index, columns=j.columns) for i, j in zip(aligned_x, x)]


def cluster(x, n_clusters=5): #x should be a dataframe with 1 row per video and 1 column per timepoint/topic -- e.g. the result of np.ravel(x0.values).T
    clustered_labels = hyp.cluster(x, cluster='KMeans', n_clusters=5)

    clusters = []
    for k in np.unique(clustered_labels):
        inds = np.where(clustered_labels == k)[0] #might need to change clustered_labels to np.array(clustered_labels) in this line
        clusters.append(x.iloc[inds].copy())
    return clusters, clustered_labels #clusters[0] is a number-of-cluster_0-videos by timepoints*topics dataframe; clusters[0].iloc[0] is the reshpaed trajectory from the first video from the first cluster (a 1 by timepoints*topics matrix)

#if you have a dataframe x with index = video names and columns ['views', 'likes', 'dislikes'], then
#you can fill in the cluster labels using x['cluster'] = cluster_labels

###
def hyperplot_videos(aligned_frames, clustered_labels):
    trajectories = [v for k, v in aligned_frames.items()]
    print(f'trajectories shapes 0--9: {[t.shape for t in trajectories[:10]]}')
    hyp.plot(trajectories, hue=clustered_labels, ndims=3, reduce='IncrementalPCA', save_path='/Users/tehuttesfayebiru/PycharmProjects/Thesis/Test/' + 'clusterplot.png')
    ### another option
    #kmeans = KMeans(n_clusters=5)
    #kmeans.fit(x)
###

# def average_topics_by_cluster(modelled_data, cluster_labels):
#     topics = np.zeros([len(np.unique(cluster_labels)), modelled_data.shape[1]])
#     for i, c in enumerate(np.unique(cluster_labels)):
#         next_inds = np.where(cluster_labels == c)[0]
#         topics[i, :] = modelled_data(next_inds,:).mean(axis=0)
#     return topics #clusters by number-of-topics




def final_frame(cluster_labels, stacked_frames, path2):
    video_names = stacked_frames.index.values
    view_list = []
    like_list = []
    dislike_list = []
    like_dislike_ratio = []
    for name in video_names:
        json_name = path2 + name[:-6] + 'info.json'
        #print(json_name)
        with open(json_name) as file:
            data = json.load(file)
            curr_views = data["view_count"]
            # if curr_views is None:
            #     print(json_name)
            #     curr_views = 0
            view_list.append(curr_views)
            curr_likes = data["like_count"]
            # if curr_likes is None:
            #     print(json_name)
            #     curr_likes = 0
            like_list.append(curr_likes)
            curr_dislikes = data["dislike_count"]

            dislike_list.append(curr_dislikes)
            print('view count ',data["view_count"],'like count ' ,data["like_count"],'dislike count ',data["dislike_count"])
            if curr_dislikes ==0:
                like_dislike_ratio.append(curr_likes)
            elif curr_dislikes is None or curr_dislikes is None:
                like_dislike_ratio.append(None)
            else:
                like_dislike_ratio.append(float(curr_likes/curr_dislikes))
    df = pd.DataFrame(index=video_names)
    df['view_count'] = view_list
    df['like_count'] = like_list
    df['dislike_count'] = dislike_list
    df['like_dislike_ratio'] = like_dislike_ratio
    df['cluster'] = cluster_labels
    return df

####

def ttest (a, b):
    # perform the t test
    df = (len(a) + len(b)) - 2
    t_statistic, p_statistic = stats.ttest_ind(a,b)
    return p_statistic, t_statistic, df

#x: index = video names, columns = ['views', 'likes', 'dislikes', 'cluster']
def pairwise_ttests(x, column): #need to write "ttest" function to take 2 arrays, a and b, and return p-value, t-values, and degrees of freedom for an *unpaired* 2-tailed t-test:
    p_values = np.zeros([len(np.unique(cluster_labels)), len(np.unique(cluster_labels))])
    t_values = np.zeros_like(p_values)
    df = np.zeros_like(p_values)

    for i in np.unique(cluster_labels):
        for j in np.unique(cluster_labels):
            if i == j:
                pass
            p_values[i, j], t_values[i, j], df[i, j] = ttest(x.dropna(axis=0, subset=[column]).query(f'cluster == {i}')[column].values, x.dropna(axis=0, subset=[column]).query(f'cluster == {j}')[column].values)
    print("t test check point")
    return p_values, t_values, df

#p, t, df = pairwise_ttests(x, 'views')

#averages:
# x.groupby('cluster').mean()

#standard deviations
# x.groupby('cluster').std()

# plot mean values function

def plot_variance(trajectories, bar_labels):
    variances = [np.var(traj, axis=0).mean() for traj in trajectories]
    return plt.bar(bar_labels, variances)


# plot clusters function
# plot frequency distribution
####

# #
Actual__path = '/Users/tehuttesfayebiru/PycharmProjects/Thesis/Thesis_transcripts/Timepoint_files/'
test__path = '/Users/tehuttesfayebiru/PycharmProjects/Thesis/Test/'
path2 = '/Users/tehuttesfayebiru/PycharmProjects/Thesis/Thesis_transcripts/Json_files/'
vectorizer, docs_vectorized, all_data_dictionary = vectorize(test__path)
#print(" vectorized data: ")o
#print(docs_vectorized)
model, modelled_data = fit_model(docs_vectorized) #modelled_data: number-of-videos by number-of-topics
# np.save("all_data_dictionary.npy", all_data_dictionary)
# np.save("vectorizer.npy", vectorizer)
# np.save("model.npy", model)
f = open("model", "wb")
pickle.dump(model, f)
pickle.dump(vectorizer, f)
pickle.dump(all_data_dictionary, f)
f.close()
# model = np.load("/Users/tehuttesfayebiru/PycharmProjects/Thesis/Test/model.npy", allow_pickle=True)
# vectorizer = np.load("/Users/tehuttesfayebiru/PycharmProjects/Thesis/Test/vectorizer.npy", allow_pickle=True)[0]
# all_data_dictionary = np.load("/Users/tehuttesfayebiru/PycharmProjects/Thesis/Test/all_data_dictionary.npy", allow_pickle=True)[0]
#print("data loaded")
#d2=np.load("d1.npy", allow_pickle=True)
# f = open("model", "rb")
# model = pickle.load(f)
# vectorizer = pickle.load(f)
# all_data_dictionary = pickle.load(f)
# f.close()
# print d1.get('key1')
# print d2.item().get('key2')
# save the vectorizer, the lda model and dictionary
display_top_words(model, vectorizer)
resampled_frames = topic_trajectories(all_data_dictionary, model, vectorizer) #returns a dict of resampled and smoothed topic trajectories
print("resample frames ")
#plot_variance(resampled_frames.values., dataframes_dictionary.keys())
aligned_frames = align_dictionary(resampled_frames)
stacked = stack_dictionaries(aligned_frames)
clustered_data, cluster_labels = cluster(stacked)
#np.save('/Users/tehuttesfayebiru/PycharmProjects/Thesis/Test/cluster_labels.npy', cluster_labels)

#hyp.plot(labels=cluster_labels, n_clusters=5)
video_data = final_frame(cluster_labels, stacked, path2)

#print("video data ", video_data)
p_view, t_view, df_view = pairwise_ttests(video_data, 'view_count')
#print('p_view ', p_view, 't_view ', t_view, 'df_view ', df_view)
p_like, t_like, df_like = pairwise_ttests(video_data, 'like_count')
p_dislike, t_dislike, df_dislike = pairwise_ttests(video_data, 'dislike_count')
p_ratio, t_ratio, df_ratio = pairwise_ttests(video_data, 'like_dislike_ratio')
stat_frame = pd.DataFrame(index=['view_count', 'like_count', 'dislike_count', 'like_dislike_ratio'])
stat_frame['p_value'] = [p_view, p_like, p_dislike, p_ratio]
stat_frame['t_stat'] = [t_view, t_like, t_dislike, t_ratio]
stat_frame['d_freedom'] = [df_view, df_like, df_dislike, df_ratio]
stat_frame.to_csv(path_or_buf='video_stat.csv')
# plot the first row of the first cluster's DataFrame,
# reshaping it to the original timepoints-by-topics matrix
hyp.plot(clustered_data[0].iloc[0].values.reshape(-1, 50), save_path='/Users/tehuttesfayebiru/PycharmProjects/Thesis/graphs/traj_plot.pdf')
# where each item is a 10,000-timepoint by 10-topic matrix
plot_data = list(np.concatenate(tuple(df.values.reshape(-1, 1000, 50) for df in clustered_data)))
# create the list of labels to pass to hypertools.plot's "hue" argument: a flattened list
# of the cluster number each timepoint of each video belongs to
cluster_hue = np.concatenate([[i+1] * clustered_data[i].shape[0] * 1000 for i in range(len(clustered_data))])
# plot all 150 videos, colored by cluster
hyp.plot(
    plot_data,
    hue=cluster_hue,
    legend=True,
    size=[8,8], save_path='/Users/tehuttesfayebiru/PycharmProjects/Thesis/graphs/cluster_plot.pdf'
)
plt.close('all')

means_frame = video_data.groupby('cluster').mean()
means_frame.plot.bar()
plt.savefig('bar_plot.png', bbox_inches='tight')
# std_frame = video_data.groupby('cluster').std()
# bins = ['01', '02', '03', '04', '05']
# plt.xlabel("cluster")
# plt.ylabel("mean_views")
# print("mean views: ")
# print(means_frame['view_count'].tolist())
# plt.bar(bins, means_frame['view_count'].tolist())
# plt.savefig('bar_view_count.png', bbox_inches='tight')
# plt.ylabel("mean_likes")
# print("mean likes: ")
# print(means_frame['like_count'].tolist())
# plt.bar(bins, means_frame['like_count'].tolist())
# plt.savefig('bar_like_count.png', bbox_inches='tight')
# plt.ylabel("mean_dislikes")
# print("mean dislikes: ")
# print(means_frame['dislike_count'].tolist())
# plt.bar(bins, means_frame['dislike_count'].tolist())
# plt.savefig('bar_dislike_count.png', bbox_inches='tight')
# plt.ylabel("mean_ratio")
# print("mean ratio: ")
# print(means_frame['like_dislike_ratio'].tolist())
# plt.bar(bins, means_frame['like_dislike_ratio'].tolist())
# plt.savefig('bar_ratio.png', bbox_inches='tight')

for j in np.unique(cluster_labels):
    a = video_data.dropna(axis=0, subset=['view_count']).query(f'cluster == {j}')['view_count'].values
    #print(j, " view means ", a)
    sns.displot(data=a, kind="kde")
    # save
    plt.savefig("view_count_" + str(j), bbox_inches='tight')
    # the same for like/dislike ratio
    l = video_data.dropna(axis=0, subset=['like_count']).query(f'cluster == {j}')['like_count'].values
    #print(j," like means ", l)
    suml = l.sum()
    lp = [round((i/suml), 2) * 100 for i in l]
    #print("like % ", lp)
    sns.displot(data=lp, kind="kde")
    # save
    plt.savefig("like_count_" + str(j), bbox_inches='tight')
    d = video_data.dropna(axis=0, subset=['dislike_count']).query(f'cluster == {j}')['dislike_count'].values
    #print(j, " dislike means ", d)
    sumd = d.sum()
    dp = [(round(i/sumd),2) * 100 for i in d]
    #print("dislike % ", dp)
    sns.displot(data=dp, kind="kde")
    # save
    plt.savefig("dislike_count_" + str(j), bbox_inches='tight')
    r = video_data.dropna(axis=0, subset=['like_dislike_ratio']).query(f'cluster == {j}')['like_dislike_ratio'].values
    #print(j, " ratio means ", r)
    sns.displot(data=r, kind="kde")
    # save
    plt.savefig("ratio_" + str(j), bbox_inches='tight')


# import seaborn as sns
# sns.displot(data=X, kind=“kde")

# plt.savefig(‘name_of_graph.png’, bbox_inches='tight')
# #hyperplot_videos(aligned_frames, cluster_labels)
# df['view_count'] = view_list
# df['like_count'] = like_list
# df['dislike_count'] = dislike_list
# df['like_dislike_ratio'] = like_dislike_ratio
# ts = t_values
# ts[p_values > 0.05] = np.nan
# sns.heatmap(ts)