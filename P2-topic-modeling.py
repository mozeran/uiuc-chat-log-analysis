# Practicum: Chat Reference Analysis
# Xinyu Tian
# xt5@illinois.edu
# comments come before codes

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import GibbsSamplingDMM as DMM
import subprocess
import CreateBatchFile
from ModelEvaluation import topicCoherence
import ast
from collections import Counter
from itertools import chain


class chatText(object):
    corpus = []
    corpusTra = []
    corpusTes = []
    outputTopicWords = []

    def __init__(self, body, n_topics):
        self.body = body
        self.n_topics = n_topics

    def preprocessing(self, removeRareWords = False, rareWordThres=1):
        """
        Given a list containing the chats cleaned, perform tokenizing, stopword removing and stemming.
        :param body: a list whose items are strings of chats
        :return: a list of processed words
        """
        # remove all numbers and punctuations
        print('\nPreparing the conversation texts...')
        print('Removing numbers and punctuations...')
        tokenizer = RegexpTokenizer(r'[a-zA-Z]\w+\'?\w*')
        tokenizedTexts = [tokenizer.tokenize(str(raw).lower()) for raw in self.body]
        # remove common stopwords
        print('Removing stopwords...')
        stopword = stopwords.words('english')
        stoppedTexts = [[word for word in text if word not in stopword] for text in tokenizedTexts]
        if removeRareWords:
            corpusWordCount = dict(Counter(list(chain.from_iterable(stoppedTexts))))
            rareWords = [item[0] for item in corpusWordCount.items() if item[1] <= rareWordThres]
            stoppedTexts = [[word for word in text if word not in rareWords] for text in stoppedTexts]

        # stemming: reduce derived words to the root/stem words, including remove 'ed', 's', 'ing', 'ly', etc.
        # focus on semantics and help removing rare words
        print('Transforming words to their word stem...')
        pStemmer = PorterStemmer()
        stemmedTexts = [[pStemmer.stem(i) for i in token] for token in stoppedTexts]
        # get the corpus by joining the tokens
        self.corpus = [' '.join(token) for token in stemmedTexts]

    def trainingTestSplit(self, trainingRatio=0.7):
        """
        Given a corpus as a list, separate it to training and test set by random sampling.
        :param corpus: a list of corpus from preprocessing()
        :return: a tuple of lists, training set takes first and test one is the second
        """
        # use np.random to create a series of random integers as the index of the training cases
        print('\nSeparating the training and test set...')
        traSample = np.random.randint(0, len(self.corpus), int(len(self.corpus) * trainingRatio))
        # slice the training corpus
        self.corpusTra = [self.corpus[index] for index in traSample]
        # choose the remaining cases as the test corpus
        self.corpusTes = list(set(self.corpus) - set(self.corpusTra))

    @staticmethod
    def getTopicalWords(model, featureNames, n_topWords: int):
        """
        The function is used with LDA, NMF and other sklearn-based models only.
        Given a model object, feature names and the number of top words, return a list of topic words.
        :param model: LDA or NMF fitted.
        :param featureNames: feature names by either tf_vectorizer or tfidf_vectorizer.
        :param n_topWords: number of words in every topic
        :return: a list of topic words: [['w1', 'w2', ..., 'wn'], ['w1', 'w2', ..., 'wn'], ... ]
        """
        top_word = []
        for topic_idx, topic in enumerate(model.components_):
            topic_top_word = [featureNames[i] for i in topic.argsort()[:-n_topWords - 1:-1]]
            top_word.append(topic_top_word)
        return top_word

    def runLDAModel(self, n_topics, n_topWords=10, n_iters=30, max_df=0.95, min_df=2):
        """
        Given a training corpus, return a list of the topic words by LDA
        :param n_topics: number of topics
        :param n_topWords: the number of words in a topic
        :param n_iters: the maximum number of iterations
        :param max_df: when building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
                       If float, the parameter represents a proportion of documents, integer absolute counts
        :param min_df: when building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
                       This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts
        :return: the list of top topic words
        """
        print('\nBuilding LDA model...')
        vectorizerTF = CountVectorizer(max_df=max_df, min_df=min_df, max_features=None)
        corpusTF = vectorizerTF.fit_transform(self.corpusTra)
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=n_iters, learning_method='batch')
        lda.fit(corpusTF)
        featureNamesTF = vectorizerTF.get_feature_names()
        topicalWords = self.getTopicalWords(lda, featureNamesTF, n_topWords)
        return topicalWords

    def runNMFModel(self, n_topics, n_topWords=10, n_iters=1000, max_df=0.2, min_df=2):
        """
        Given a training corpus, return a list of the topic words by NMF
        :param n_topics: number of topics
        :param n_topWords: the number of words in a topic
        :param n_iters: the maximum number of iterations
        :param max_df: When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
                       If float, the parameter represents a proportion of documents, integer absolute counts.
        :param min_df: When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
                       This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts.
        :return: the list of top topic words
        """
        print('\nBuilding NMF model...')
        vectorizerTFIDF = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=None)
        corpusTFIDF = vectorizerTFIDF.fit_transform(self.corpusTra)
        nmf = NMF(n_components=n_topics, random_state=1,
                  beta_loss='kullback-leibler', solver='mu', max_iter=n_iters, alpha=0.1,
                  l1_ratio=0.5)
        nmf.fit(corpusTFIDF)
        featureNamesTFIDF = vectorizerTFIDF.get_feature_names()
        topicalWords = self.getTopicalWords(nmf, featureNamesTFIDF, n_topWords)
        return topicalWords

    def runDMMModel(self, n_topics, n_topWords=10, n_iters=2000, alpha=0.3, beta=0.8):
        """
        Given a training corpus, return a list of the topic words by DMM
        :param n_topics: number of topics
        :param n_topWords: the number of words in a topic
        :param n_iters: the maximum number of iterations
        :param alpha: the default value is 0.1.
        :param beta: the default value is 0.01. the users may consider to the beta value of 0.1 for short texts.
        :return: the list of top topic words
        """
        print('\nBuilding DMM model...')
        dmm = DMM.GibbsSamplingDMM(corpus=self.corpusTra, ntopics=n_topics, niters=n_iters, twords=n_topWords,
                                   alpha=alpha, beta=beta)
        dmm.analyseCorpus()
        dmm.topicAssigmentInitialise()
        topicalWords = dmm.GetTopicalWordList()
        return topicalWords

    def runPhraseLDAModel(self, n_topics, min_support=30, max_pattern=2, n_iters=1000):
        # to run phrase-LDA models, make sure you have installed Java and added the path to your local environment variable
        # see README for more details.
        """
        Given a training corpus, return a list of the topic words by phrase-LDA
        :param n_topics: integer, number of topics
        :param min_support: integer, minimum times a phrase candidate should appear in the corpus to be significant
        :param max_pattern: integer, max size you would like a phrase to be (if you don't want too long of phrases that occasionally occur)
        :param iter_times: the maximum number of iterations
        :return: the list of top topic words
        """
        # write the datasets into separate file for phrase-LDA
        with open('texts_tra.txt', 'w', encoding='utf8') as corpusTraF:
            print('\n'.join(self.corpusTra), file=corpusTraF)

        print('\nBuilding phrase-LDA model...')
        CreateBatchFile.write(n_topics, min_support, max_pattern, n_iters)

        proc = subprocess.Popen('win_run.bat', stdout=subprocess.PIPE)
        output = proc.communicate()[0]

        with open('./output/outputFiles/topicalPhrases.txt') as pLDA:
            pLDAResult = pLDA.read()
        return pLDAResult

    def getOutput(self, modelName):
        # you can change the parameters right here
        models = {'LDA': 'self.runLDAModel(self.n_topics, n_topWords=10, n_iters=30, max_df=0.95, min_df=2)',
                  'NMF': 'self.runNMFModel(self.n_topics, n_topWords=10, n_iters=1000, max_df=0.2, min_df=2)',
                  'DMM': 'self.runDMMModel(self.n_topics, n_topWords=10, n_iters=2000, alpha=0.3, beta=0.8)',
                  'phrase-LDA': 'self.runPhraseLDAModel(self.n_topics, min_support=30, max_pattern=2, n_iters=1000)'}
        self.outputTopicWords = eval(models[modelName])
        if modelName == 'phrase-LDA':
            self.outputTopicWords = ast.literal_eval(self.outputTopicWords)
        # print the topics in separate lines
        i = 1
        for topic in self.outputTopicWords:
            print('Topic' + str(i) + ': ' + ' '.join(topic))
            i += 1
        # calculate nPMI score as a evaluation metric
        tc = topicCoherence(self.outputTopicWords, self.corpusTes)
        print(modelName + ' Mean Topic Coherence:', str(tc.getNPMI()))
        return self.outputTopicWords


if __name__ == '__main__':
    # ask users to input the name of the csv file cleaned, make sure it contains the column of 'body'
    prompt = 'Welcome.\nStep 1. Please input the cleaned .csv file including the column "body"\
              \nFor example: "chats_spring2017_all_cleaned_body".\nPress Enter to continue.\n'
    while True:
        try:
            finName = str(input(prompt)) + '.csv'
            # read the csv file
            chats = pd.read_csv(finName, sep=',', na_values='NA')
            break
        except FileNotFoundError:
            prompt = 'The file does not exist. Please check the file name and input the correct one.\n'

    # ask users about the preference on different topic models
    models = ['LDA', 'NMF', 'DMM', 'phrase-LDA']
    modelSelected = []
    result = {}
    print('\nStep 2. Please select the topic models to use.')
    for model in models:
        while True:
            modelPref = input('try ' + model + '? (Y/N)\n')
            if modelPref in ['Y', 'y']:
                modelSelected.append(model)
                break
            elif modelPref in ['N', 'n']:
                break
            else:
                continue
    n_topics = eval(input('\nStep 3. Please specify the number of topics. (You may want to get started from 10)\n'))

    # pass the text and n_topics into the object
    SP17corpus = chatText(body=[chat for chat in chats['body']], n_topics=n_topics)
    # separate the training and test data
    SP17corpus.preprocessing()
    # split training and test set
    SP17corpus.trainingTestSplit(trainingRatio=0.6)

    for model in modelSelected:
        result[model] = SP17corpus.getOutput(modelName=model)