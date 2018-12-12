# adapted from atefm (https://github.com/atefm/pDMM)

import random

class GibbsSamplingDMM(object):
	numDocuments = 0
	numWordsInCorpus = 0
	word2IdVocabulary = {}
	id2WordVocabulary = {}
	documents = []
	occurenceToIndexCount = []
	topicAssignments = []
	docTopicCount = []
	topicWordCount = []
	sumTopicWordCount = []
	multiPros = []
	betaSum=0.

	def __init__(self, corpus, ntopics=20, alpha=0.1, beta=0.001, niters=2000, twords=20):
		super(GibbsSamplingDMM, self).__init__()
		self.corpus = corpus
		self.ntopics = ntopics
		self.alpha = alpha
		self.beta = beta
		self.niters = niters
		self.twords = twords

	def analyseCorpus(self):
		indexWord=0

		for doc in self.corpus:
			document = []
			wordOccurenceToIndexInDocCount = {}
			wordOccurenceToIndexInDoc = []
			if doc.rstrip!=None:
				words = doc.rstrip().split()
				for word in words:

					if word in self.word2IdVocabulary:
						document.append(self.word2IdVocabulary[word])
					else:
						self.word2IdVocabulary[word]=indexWord
						self.id2WordVocabulary[indexWord]=word
						document.append(indexWord)
						indexWord+=1

					if word in wordOccurenceToIndexInDocCount:
						wordOccurenceToIndexInDocCount[word]+=1
					else:
						wordOccurenceToIndexInDocCount[word]=1

					wordOccurenceToIndexInDoc.append(wordOccurenceToIndexInDocCount[word])

				self.numWordsInCorpus+=len(document)
				self.numDocuments+=1
				self.documents.append(document)
				self.occurenceToIndexCount.append(wordOccurenceToIndexInDoc)

		self.betaSum=len(self.word2IdVocabulary)*self.beta


	def topicAssigmentInitialise(self):
		self.docTopicCount = [0 for x in range(self.ntopics)]
		self.sumTopicWordCount = [0 for x in range(self.ntopics)]

		for i in range(self.ntopics):
				self.topicWordCount.append([0 for x in range(len(self.word2IdVocabulary))])

		for i in range (self.numDocuments):
			topic = random.randint(0,self.ntopics-1)
			self.docTopicCount[topic]+=1
			
			for j in range (len(self.documents[i])):
				self.topicWordCount[topic][self.documents[i][j]]+=1
				self.sumTopicWordCount[topic]+=1

			self.topicAssignments.append(topic)

	def nextDiscrete(self,a):
		b = 0.

		for i in range(len(a)):
			b+=a[i]

		r = random.uniform(0.,1.)*b
		
		b=0.
		for i in range (len(a)):
			b+=a[i]
			if(b>r):
				return i
		return len(a)-1

	def sampleInSingleIteration(self,x):
		print ("iteration: "+str(x))
		for d in range(self.numDocuments):
			topic = self.topicAssignments[d]
			self.docTopicCount[topic]-=1
			docSize = len(self.documents[d])
			document = self.documents[d]

			for w in range(docSize):
				word = document[w]
				self.topicWordCount[topic][word]-=1
				self.sumTopicWordCount[topic]-=1

			for t in range(self.ntopics):
				self.multiPros[t] = self.docTopicCount[t]+self.alpha

				for w in range(docSize):
					word = document[w]
					self.multiPros[t] *= (self.topicWordCount[t][word]+self.beta+self.occurenceToIndexCount[d][w]-1)/(self.sumTopicWordCount[t]+w+self.betaSum)

			topic = self.nextDiscrete(self.multiPros)

			self.docTopicCount[topic]+=1

			for w in range(docSize):
				word = document[w]
				self.topicWordCount[topic][word]+=1
				self.sumTopicWordCount[topic]+=1

			self.topicAssignments[d] = topic

	def GetTopicalWordList(self):
		all_list = []
		for t in range(self.ntopics):
			list = []
			wordCount = {w: self.topicWordCount[t][w] for w in range(len(self.word2IdVocabulary))}

			count = 0

			for index in sorted(wordCount, key=wordCount.get, reverse=True)[50:]:
				if count < self.twords:
					list.append(self.id2WordVocabulary[index])
					count += 1
			all_list.append(list)
		return all_list
