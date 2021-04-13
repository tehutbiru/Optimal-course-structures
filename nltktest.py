import nltk, string
import os, os.path, shutil
from nltk import *
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
global curr_time_stamp
global total_tokens
global secondLine
global doubleLine
global count
global error_list


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

def pre_process():
    count = 0
    error_files = 0
    # changing the directory
    os.chdir('/Users/tehuttesfayebiru/PycharmProjects/Thesis/Thesis_transcripts/')
    # loop through files in directory
    print("start ")
    print(len([name for name in os.listdir('.') if os.path.isfile(name) and name.endswith('.vtt')]))
    for file in os.listdir('/Users/tehuttesfayebiru/PycharmProjects/Thesis/Thesis_transcripts/'):
        try:
            # if it is a video transcript file

            if file.endswith('.vtt'):
                count += 1
                print("file number " + str(count) + ": " + file)
                save_new_path = '/Users/tehuttesfayebiru/PycharmProjects/Thesis/Thesis_transcripts/Pre-processed_files/'
                save_time_path = '/Users/tehuttesfayebiru/PycharmProjects/Thesis/Thesis_transcripts/Timepoint_files/'
                save_tagged_path = '/Users/tehuttesfayebiru/PycharmProjects/Thesis/Thesis_transcripts/Tagged_files/'

                # read that file, open 3 files to write pre-processed transcript, pos tagged transcript
                # and transcript separated by time point
                with open(file, 'r') as fin, open(save_new_path + 'new-' + file, 'w') as fout, open(save_tagged_path + 'tagd-' + file, 'w')\
                        as fout2, open(save_time_path + 'time-' + file, 'w') as fout3:
                    # loop through lines in file
                    lines = list(fin.readlines()[4:])
                    # use enumerator to loop through files and access the next line
                    for i, line in enumerate(lines):
                        #print(line)
                        # if the line is empty, move on to next line
                        if not line.strip():
                            continue
                        else:
                            # lowercase everything
                            line = line.lower()
                            # remove punctuation
                            removed_punc = ''.join([char for char in line if char not in string.punctuation])
                            # tokenize
                            tokens = word_tokenize(removed_punc)
                            # if it is a time stamp,
                            if (all(x.isdigit() for x in tokens)):
                                # save the time stamp to a global variable for later use
                                curr_time_stamp = tokens # change name
                                # reset variables total tokens to non and secondLine to false
                                total_tokens = None
                                secondLine = False
                                # move on to next line
                                continue

                            # else - if it is not a time stamp
                            else:
                                # to make sure the index is still viable when checking for next line
                                if (i + 1) <= len(lines):
                                    # setting next line using index
                                    next_line = lines[i + 1]
                                    # If the next line is empty
                                    if next_line == "\n":
                                        # There isn't a second line
                                        secondLine = False
                                    # if the next line isn't empty
                                    else:
                                        # set second line to true
                                        secondLine = True
                                        # assign the tokens to global variable total tokens
                                        # to save for use later
                                        total_tokens = tokens
                                        # we are looking at a double line
                                        doubleLine = True

                                    # identify time point increments by dividing the interval with 1 +
                                    # the number of words (tokens) in the following line
                                    # if total tokens hasn't been assigned yet, assign it to total tokens
                                    if total_tokens == None:
                                        total_tokens = tokens
                                    # If there isn't a second line
                                    if secondLine is False:
                                        # Calculate the time interval in the time stamps
                                        if len(curr_time_stamp)> 1:
                                            diff = int(curr_time_stamp[1]) - int(curr_time_stamp[0])
                                        else:
                                            diff = int(curr_time_stamp)
                                        # if we are in a double line, join the current tokens to the end
                                        # of the tokens in the previous line
                                        if doubleLine:
                                            total_tokens.extend(tokens)
                                            # set double line to false (already joined)
                                            doubleLine = False
                                        # use the total tokens to calculate time point
                                        time_point = diff/(len(total_tokens) + 1)
                                        # iterate up to the length of the tokens
                                        for t in range(0, len(total_tokens)):
                                            # write to a file - timepoint, token in corresponding indices
                                            fout3.write(str(int(curr_time_stamp[0]) + t*time_point) + ', ' + total_tokens[t] + '\n')

                                    # pos tagging the words (tokens)
                                    tagged = nltk.pos_tag(tokens)
                                    # writing the tagged line to a file
                                    fout2.write(''.join(str(tagged)) + '\n')



                                    word_lemmatizer = WordNetLemmatizer()
                                    # lemmatize line by passing word and corresponding tag (call helper function)
                                    lemmatized = [word_lemmatizer.lemmatize(tag[0], get_wordnet_pos(tag[1])) for tag in tagged]

                                    # write pre-processed line to a file
                                    fout.write(' '.join(lemmatized) + '\n')
        # handle any error files
        except:
            error_files += 1
            print("error with file: " + file)
            # set source and destination path of problem file
            source_path = '/Users/tehuttesfayebiru/PycharmProjects/Thesis/Thesis_transcripts/' + file
            destination_path = '/Users/tehuttesfayebiru/PycharmProjects/Thesis/Thesis_transcripts/Problem_files/' + file
            # move problem file to a specified folder
            shutil.move(source_path, destination_path)


pre_process()

