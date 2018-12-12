# adapted from Jey Han Lau [https://github.com/jhlau/topic_interpretability]
# simplified by using the fixed window size (all words in an chat/article) and restructured as object-oriented
# supports counting the multi-word phrases in topics (e.g. search_box, faculti_member, ...). The default maxPattern is set as 3. To handle ngrams where n > 3, please change the value of maxPattern.

import math
import numpy as np
from itertools import combinations_with_replacement as CwR
from collections import defaultdict

class topicCoherence(object):
    topicUnigrams = set()
    topicID2Unigram = {}
    topicUnigram2ID = {}
    topicalWordPairs = {}
    wordCount = defaultdict(int)
    topicCoherence = defaultdict(int)

    def __init__(self, topics, refCorpus, concatColSep = '_', maxPattern = 3):
        """
        The initialization function where important parameters are assigned.
        :param topics: the list of topical words/phrases as the output of a given model in list of lists
        For example:
        [['interlibrari_loan', 'lose_chat', 'chat_servic', 'lower_level', 'chat_open', 'writer_workshop', 'spring_break', 'studi_room', 'call_ugl', 'add_chat'],
        ['good_night', 'great_day', 'good_day', 'good_luck', 'drop_menu', 'sound_good', 'nice_day', 'ye_great', 'remov_thank_welcom', 'make_sens'], ...]
        :param refCorpus: preprocessed corpus used to calculate the word count, list of texts
        :param concatColSep: separation symbol used for topic phrases, or concatenated collocations
        :param maxPattern: the maximum number of words in a phrase, if available. In case the model takes phrases with more than 3 words, please specify
        """
        self.topics = topics
        self.refCorpus = refCorpus
        self.concatColSep = concatColSep
        self.maxPattern = maxPattern

    def __words2IDs(self, words):
        indices = []
        for word in words.split():
            if word in self.topicUnigram2ID:
                indices.append(self.topicUnigram2ID[word])
            else:
                indices.append(0)
        return indices

    def __getNGrams(self, wordIDs):
        uniGrams = []
        allNGrams = []
        # get all 1-gram words first
        for i in range(len(wordIDs)):
            if wordIDs[i] != 0:
                uniGrams.append(self.topicID2Unigram[wordIDs[i]])
        allNGrams += uniGrams
        # find all possible k-gram words and check if they are in the topical words
        for k in range(2, self.maxPattern + 1):
            possibleKGrams = [self.concatColSep.join(item) for item in list(CwR(uniGrams, k))]
            # screen out those appear in the topical words
            validKGrams = [kgram for kgram in possibleKGrams if kgram in self.topicalWordPairs]
            allNGrams += validKGrams
        allNGrams = list(set(allNGrams))
        return allNGrams

    def __getWordCount(self):
        for line in self.refCorpus:
            refChatWordIDs = self.__words2IDs(line)
            nGrams = self.__getNGrams(refChatWordIDs)

            for nGram in nGrams:
                if nGram in self.topicalWordPairs:
                    self.wordCount[nGram] += 1
            # create word combinations
            for i1 in range(len(nGrams) - 1):
                for i2 in range(i1 + 1, len(nGrams)):
                    if (nGrams[i1] in self.topicalWordPairs and nGrams[i2] in self.topicalWordPairs[nGrams[i1]])\
                        or (nGrams[i2] in self.topicalWordPairs and nGrams[i1] in self.topicalWordPairs[nGrams[i2]]):
                        # save ordered paris to avoid duplication
                        if nGrams[i1] < nGrams[i2]:
                            combined = nGrams[i2] + '|' + nGrams[i1]
                        else:
                            combined = nGrams[i1] + '|' + nGrams[i2]
                        self.wordCount[combined] += 1
        return self.wordCount

    def __getWordAssoc(self, word1, word2):
        # two types of combinations
        combined1 = word1 + '|' + word2
        combined2 = word2 + '|' + word1
        n_combined = 0
        # get the number from word counts
        if combined1 in self.wordCount:
            n_combined = self.wordCount[combined1]
        elif combined2 in self.wordCount:
            n_combined = self.wordCount[combined2]
        # ignore the irrelevant words
        n_word1 = self.wordCount[word1] if word1 in self.wordCount else 0
        n_word2 = self.wordCount[word2] if word2 in self.wordCount else 0
        n_total = len(self.refCorpus)
        # simplified, if each of the count is zero, the result will be zero
        if n_word1 == 0 or n_word2 == 0 or n_combined == 0:
            result = 0.0
        else:
            result = math.log((float(n_combined) * float(n_total)) / \
                              float(n_word1 * n_word2), 10)
            try:
                result = result / (-1.0 * math.log(float(n_combined) / n_total, 10))
            except ZeroDivisionError:
                result = 0
        return result

    def __getTopicCoherence(self, topic):
        topicAssoc = []
        # go through all the word pairs
        for wordID1 in range(0, len(topic) - 1):
            word1 = topic[wordID1]
            for wordID2 in range(wordID1 + 1, len(topic)):
                word2 = topic[wordID2]
                if word2 != word1:
                    topicAssoc.append(self.__getWordAssoc(word1, word2))
        # calculate the mean as the score of a topic
        return np.mean(topicAssoc)

    def getNPMI(self):
        for topic in self.topics:
            for word1 in topic:
                # separate the multi-word phrases by the symbol given
                for unigram in word1.split(self.concatColSep):
                    # check if the word really exists
                    if len(unigram) > 0:
                        self.topicUnigrams.add(unigram)     # add the uni-gram word into a set to avoid duplication
                # search all the relevant words appear in the same topic. The both words are called 'word pairs'
                # for example: {'request_item': {'check_book', 'physic_copi', 'place_request', ...}
                for word2 in topic:
                    if word2 != word1:
                        wordPairs = set()
                        if word1 in self.topicalWordPairs:
                            wordPairs = self.topicalWordPairs[word1]
                        if word2 != word1:
                            wordPairs.add(word2)
                            self.topicalWordPairs[word1] = wordPairs
        # number uni-gram words: {0: 'request_item', ...}
        self.topicID2Unigram = dict(enumerate(sorted(list(self.topicUnigrams))))
        # reverse the dictionary {'request_item': 0, ...}
        self.topicUnigram2ID = {unigram: id for id, unigram in self.topicID2Unigram.items()}
        self.wordCount = self.__getWordCount()
        topicList = [','.join(topic) for topic in self.topics]

        for i in range(len(topicList)):
            self.topicCoherence[i] = self.__getTopicCoherence(topicList[i].split(','))
        # calculate the mean score for all topics
        NPMI = np.mean(list(self.topicCoherence.values()))
        return NPMI
