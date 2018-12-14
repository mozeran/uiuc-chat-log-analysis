# Chat Reference Analysis 

The program is developed for the Library at the University of Illinois at Urbana-Champaign to improve its chat reference services by analyzing the chat reference conversations and extract the topics.
It integrates 4 topic models and calculates nPMI score as a reference for users to evaluate the performance of models.

Models available:
* Latent Dirichlet Allocation (LDA)
* Non-negative Matrix Factorization (NMF)
* Dirichlet Multinomial Mixture (DMM)
* phrase-LDA

## Getting Started

### Files

There are 2 main components of the program: Phase 1 P1-data-cleaning and Phase 2 P2-topic-modeling.
In phase 2, the program works with 4 supportive modules: CreateBatchFile.py, GibbsSamplingDMM.py, ModelEvaluation.py and TopicalPhrases (TopicalPhrases, output and win_run_default.bat).

Please make sure all of these files above are placed in the same directory, as well as the raw chat log in a .csv format to process.

### Input Format

Please check the structure of the raw chat log. It should be a csv file where each line of a chat conversation is an individual line in the csv, and the csv contains the following columns:

| File  | conversationID | fromJID | fromJIDResource | body | Time | Date |
|-------|----------------|---------|-----------------|------|------|------|

* File: The original filename of the data; this is a carryover from using OpenRefine for some initial cleaning of ASCII characters.
* conversationID: A unique identifier for each chat conversation.
* fromJID: The username (if staff) or unique identifier of the person who typed into the chatbox.
* fromJIDResource: Identifies whether the person who typed was staff (string "Smack") or a patron (any other string).
* body: The actual text written into the chatbox.
* Time: The time a line of chat was sent.
* Date: The date a line of chat was sent.

(The above format is based on the way the chat software was set up at the time of this project.)

### Installing

Before running the program, please install all Python packages used:

* numpy
* pandas
* nltk
* sklearn
* langdetect
* langid

You may either use Anaconda or pip command to install the packages mentioned.

```
pip install pandas
```

In case of NLTK stopword failure, please download the stopword files from NLTK manually. You may want to run the following commands in Python console:

```
import nltk
nltk.download('stopwords')
```

Due to limited time, we call the phrase-LDA module directly, which was initially implemented in both Python and Java. See [Original Code by Ahmed El-Kishky](http://web.engr.illinois.edu/~elkishk2/code/ToPMine.zip).
To run phrase-LDA model, please [install Java](https://www.java.com/en/download/), and add '$Path$\Java\jre\bin' into PATH list of environment variables:

* [Windows](https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/)
* [MacOS](https://gist.github.com/patriciogonzalezvivo/77da993b14a48753efda)

## Running

Phase 1. run P1-data-cleaning -> Phase 2. run P2-topic-modeling.
Please follow the instructions to type in.

### Phase 1 Data Cleaning

Step 1. Please input the languages you would like to detect, all the possible languages used or the languages to consider.

By default, you can just input "en es" for English and Spanish. For other languages, please key in [ISO 639-1 codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) instead of their names.

It helps improving the accuracy of language detection, which adds two new columns in -body.csv lang1 by langdetect and lang2 by langid. Generally, lang2 tends to be more conservative and reliable.
lang1 is more likely to produce false negative cases by mistaking English chats as non-English ones. In this way, you may want to change the rule for making a final decision. 
Now, the rule checks the result by 'lang2' first and see whether it is consistent with that given by 'lang1'.

Step 2. Please input the .csv file to process. Just copy and paste the name of csv file. 

```
Loading the dataset...
89100 rows have been read from chats_spring2017_all_cleaned.csv

Numbering all the library staffs...
There are 74 library staffs.

Cleaning the rows...
Personally identifiable information removed, now shown as 'REMOVED'.
All URLs and standardized greetings are now removed.

Grouping all the conversations...
Conversations with less than 5 words are neglected.
There are 5706 chats.

Detecting languages...
There are 2 chats in other languages.

The grouped conversations are exported to chats_spring2017_all_cleaned_body.csv. Non-English rows are separated out.
```

At last, you will get 3 new .csv files: -body, -en and -non-en. The -body .csv will be used in the Phase 2 to perform the text mining.

### Phase 2 Topic Modeling

Step 1. Input the name of the cleaned -body.csv file, created by Phase 1.

Step 2. Select the models you would like to run by inputting Y or N.

Step 3. Assume the number of topics, usually ranging from 3 to 20.

```
Preparing the conversation texts...
Removing numbers and punctuations...
Removing stopwords...
Transforming derived words to the stem form...

Separating the training and test set...

Building LDA model...
Topic1: see ok click articl page go one citat thank look
Topic2: account librari ok thank number reserv tri help use check
Topic3: thank help remov look inform find sourc resourc librari would
Topic4: access tri link remov thank librari use get campu page
Topic5: book librari thank request help look check copi catalog remov
Topic6: articl access journal thank text full ok look find tri
Topic7: thank grainger room floor ugl hello hi help level studi
Topic8: search databas look articl find help okay thank use tri
Topic9: articl thank journal help let look remov click find link
Topic10: thank good chat okay one night help know open oh
LDA Mean Topic Coherence: 0.1610832050527773
```

The nPMI score ("Mean Topic Coherence") is only a reference metric that evaluates the re-appearance of topic words in the reference corpus. In general, the larger nPMI is, the better the model performs. However, studies suggested that there are significant gap between the score and human interpretability. Because of that, we didn't do parameter tuning based on the nPMI score.

*A quick note about phrase-LDA: This topic model, as we implemented it, required many files and subfolders. It was also intricately connected with our data. We have tried to share the structure and all necessary components here, without sharing the raw data. It's likely that in the translation process, phrase-LDA will have some bugs.*

### Key Parameters

Apart from the argument number of topics, the following parameters are crucial to the result. To simplify the program, these parameters are not prompted but subject to change directly.

#### Maximum document frequency (max_df)

It is an important argument for both the count vectorizer and TF-IDF vectorizer. By definition, it is a threshold to ignore terms when their document frequency is higher than that.
If float, the parameter represents the proportion of documents, while if integer, it indicates the absolute counts.

The value of max_df affects the result by properly filtering out the functional/common words, for example, ok, thank, good, etc, whose document frequency are considerably high.
Therefore, in case the result contains too much useless words, you may consider decreasing the value of max_df.

#### Number of top words (n_topWords)

It controls the number of words for each topic. From the experience of experiments, NMF prefers lower values.

#### Number of iterations (n_iters)

The number of maximum iteration times. You may want to reduce it if the process takes too much time. See codes for default settings.

#### Reference Corpus

As the size of chat log increases, it is recommended to use the external data, just like the chat history of another semester as the test set to obtain a better validity. You can change the settings in the function training_test_split in Phase 2.

Additionally, to change the learning method in [LDA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation), or the solver and beta loss function in [NMF](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html), please see the original documents.

## Built With

* [pandas](https://pandas.pydata.org/) - data manipulation 
* [nltk](https://www.nltk.org/) - NLP preprocessing
* [scikit-learn](http://scikit-learn.org/stable/documentation.html) - LDA and NMF model
* [langdetect](https://github.com/Mimino666/langdetect) - language detection
* [langid](https://github.com/saffsd/langid.py) - language detection

## Authors

* [Xinyu Tian](https://github.com/tianxiny)
* [Megan Ozeran](https://github.com/mozeran)

## Acknowledgments

* Ahmed El-Kishky et al, Scalable Topical Phrase Mining from Text Corpora, http://www.vldb.org/pvldb/vol8/p305-ElKishky.pdf
* atefm, pDMM: Python implemetation for Dirichlet Multinomial Mixture (DMM) model, https://github.com/atefm/pDMM
* jhlau, Computation of the semantic interpretability of topics produced by topic models, https://github.com/jhlau/topic_interpretability
