# Practicum: Chat Reference Analysis
# Xinyu Tian
# xt5@illinois.edu
# comments come before codes

import re
import pandas as pd
import langdetect
import langid

def readChatLog(csvFilePath: str) -> pd.DataFrame:
    """
    Read the chat log from the csv file.
    :param csvFilePath: the path of the csv file, use either an absolute or a relative path
    :return: a DataFrame
    """
    print("\nLoading the dataset...")
    # please specify the columns to read by changing the items in the list 'header'
    header = ['File', 'conversationID', 'fromJID', 'fromJIDResource', 'sentDate', 'body', 'Time', 'Date']
    # data type of each column, strings as default
    df = pd.read_csv(csvFilePath, na_values='NA', encoding='utf8', usecols=header,
                     dtype={'File': str, 'conversationID': str, 'fromJID': str, 'fromJIDResource': str,
                                  'sentDate': str, 'body': str, 'Time': str, 'Date': str})
    print(str(len(df.index)),"rows have been read from", csvFilePath)
    return df

def getStaffDict(df: pd.DataFrame) -> dict:
    """
    Number the library staffs by their working emails to remove the PII of library staffs.
    :param df: the DataFrame read by readChatLog()
    :return: a dictionary whose keys are emails and values are automatically generated numbers of staffs
    """
    print("\nNumbering all the library staffs...")
    # use a set to acquire all unique values of emails
    staffSet = set()
    # go through every row of the DataFrame
    for index, row in df.iterrows():
        # 'Smack' is a pattern to identify the staffs from all senders
        if row['fromJIDResource'] == 'Smack':
            staffSet.add(row['fromJID'])
    # number the staffs by enumerate(), returns a dictionary just like
    # {0: 'aaa@chat.library.illinois.edu', 1: 'bbb@chat.library.illinois.edu', ... }
    ID2Staff = dict(enumerate(s for s in staffSet))
    # switch the keys and values in the dictionary to map the emails to the autonomous names (e.g. staff01)
    # {'aaa@chat.library.illinois.edu': 'staff001', 'bbb@chat.library.illinois.edu': 'staff002', ... }
    # use zfill to make sure that all staffs have a 3-digit number
    Staff2ID = dict((staff, str(ID + 1).zfill(3)) for ID, staff in ID2Staff.items())
    print("There are " + str(len(Staff2ID)) + " library staffs.")
    return Staff2ID

def cleanChatText(df: pd.DataFrame, staff2ID: dict) -> pd.DataFrame:
    """
    Remove netID, library ID, UIN, all URLs and standardized greetings.
    :param df: the DataFrame read by readChatLog()
    :param staff2ID: a dictionary whose keys are emails and values are automatically generated numbers of staffs
    :return: the cleaned DataFrame with all PII removed
    """
    print("\nCleaning the rows...")
    for index, row in df.iterrows():
        # change all staff usernames to generic/autonomous ones by using staff2ID to map
        if row['fromJID'] in staff2ID:
            row['fromJID'] = 'staff' + str(staff2ID[row['fromJID']])
        else:
            # assume all other senders are patron
            row['fromJID'] = 'patron'
        # enforce the value in each row of row['body'] to be a string
        row['body'] = str(row['body'])
        # use Regex to remove PII
        # remove netID
        row['body'] = re.sub(r'[Nn][Ee][Tt][Ii][Dd]([:,]?\s|[:,]\s?)[A-Za-z]+\d', 'netID REMOVED', row['body'])
        # remove UIN and library ID - (!) might result in a few false positives
        row['body'] = re.sub(r'(\s\d{9}|\s\d{14})', 'REMOVED', row['body'])
        # remove URLs
        row['body'] = re.sub(r'(<a\s.+</a>|(http://|https://).+)', 'REMOVED', row['body'])
        # remove standardized greetings
        row['body'] = re.sub(r'.* [Aa]sk [Aa] [Ll]ibrarian [Ss]ervice.*', '', row['body'])
    print("Personally identifiable information removed, now shown as 'REMOVED'.\n"
          "All URLs and standardized greetings are now removed.")
    return df

def aggregateChat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the cleaned DataFrame, group the chats by their conversationID.
    :param df: the cleaned DataFrame with sensitive and useless information removed
    :return: a DataFrame that each row shows a single chat
    """
    print("\nGrouping all the conversations...")
    print("Conversations with less than 5 words are neglected.")
    # use the groupby method of the DataFrame, reset the index so that the result becomes a new DataFrame
    chatsGrouped = df.groupby('conversationID')['body'].apply(lambda row: ', '.join(row)).reset_index()
    # assume that all valid conversations should contains at least 5 words, throw away those too short chats
    chatsGrouped = chatsGrouped[chatsGrouped['body'].apply(lambda x: len(x.split()) >= 5)]
    print("There are " + str(len(chatsGrouped.index)) + " chats.")
    return chatsGrouped

def detectLanguage(df: pd.DataFrame):
    """
    Given a DataFrame of conversations grouped, detect the language of texts by using langdetect and langid.
    Use a rule to compare their results and make the decision.
    :param df: the DataFrame where a single row consists of only a conversation
    :return: two lists, one stores all conversations in English while the other stores those in other languages
    """
    print("\nDetecting languages...")
    # langdetect (https://github.com/Mimino666/langdetect) works better on long texts, which supports:
    # af, ar, bg, bn, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gu, he,
    # hi, hr, hu, id, it, ja, kn, ko, lt, lv, mk, ml, mr, ne, nl, no, pa, pl,
    # pt, ro, ru, sk, sl, so, sq, sv, sw, ta, te, th, tl, tr, uk, ur, vi, zh - cn, zh - tw
    df['lang1'] = df.apply(lambda row: langdetect.detect(row['body']), axis=1)
    # langid (https://github.com/saffsd/langid.py) works better on short texts,
    # and performs way better when a consideration set is given, which supports:
    # af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo,
    # fr, ga, gl, gu, he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt,
    # lv, mg, mk, ml, mn, mr, ms, mt, nb, ne, nl, nn, no, oc, or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk,
    # sl, sq, sr, sv, sw, ta, te, th, tl, tr, ug, uk, ur, vi, vo, wa, xh, zh, zu
    df['lang2'] = df.apply(lambda row: langid.classify(row['body'])[0], axis=1)
    nonEN = []
    # 'lang1' and 'lang2' are now 2 new columns at right hand
    for reset_index, row in df.iterrows():
        # define the criteria here
        # since 'lang2' tends to be more reliable, so the rule checks the result by 'lang2' first
        # and see whether it is consistent with that given by 'lang1'
        if row['lang2'] != 'en' and row['lang1'] == row['lang2']:
            nonEN.append(row['conversationID'])
    print("There are " + str(len(nonEN)) + " chats in other languages.")
    # separate the DataFrame into two, the English and non-English
    chatsEN = df[df.apply(lambda row: row['conversationID'] not in nonEN, axis=1)]
    chatsNonEN = df[df.apply(lambda row: row['conversationID'] in nonEN, axis=1)]
    return chatsEN, chatsNonEN

if __name__ == '__main__':
    # allow users to decide which languages to consider
    langsSetting = str(input('Welcome.\n'
                              'Step 1. Please \n'
                              'Use language codes: en = English, es = Spanish.\n'
                              'Use space to separate them, just like "en es".\n'
                              'Press Enter to continue.\n'))
    # input the path and name of the original csv file
    finName = str(input('Step 2. Please input the .csv file to process.\n'
                         'For example: "chats_spring2017_all_cleaned".\n'
                         'Press Enter to continue.\n'))
    # generate the derived file names
    fin = finName + ".csv"
    foutBody = finName + "_body.csv"
    foutEN = finName + "_en.csv"
    foutNonEN = finName + "_non_en.csv"
    # parse the input of the languages to consider
    langsSettings = [lang for lang in langsSetting.split(' ')]
    # set the config of langid
    langid.set_languages(langsSettings)
    # read the csv file to the DataFrame
    dfInput = readChatLog(fin)
    # number the library staffs, convert the staff usernames to generic ones
    staff2ID = getStaffDict(dfInput)
    # remove useless columns
    dfOutput = pd.DataFrame(dfInput.drop(columns=['fromJIDResource']))
    # clean the DataFrame by removing PII, links and standardized greetings
    dfOutput = cleanChatText(df=dfOutput, staff2ID=staff2ID)
    # group the logs into conversations by conversation ID
    chatsGrouped = aggregateChat(dfOutput)
    # separate non-English conversations
    chatsEN, chatsNonEN = detectLanguage(chatsGrouped)
    # write the manipulated DataFrames into different csv files
    print("\nThe grouped conversations are exported to " + finName + "_body.csv. Non-English rows are separated out.")
    chatsGrouped.to_csv(foutBody, encoding='utf8')
    chatsEN.to_csv(foutEN, encoding='utf8')
    chatsNonEN.to_csv(foutNonEN, encoding='utf8')
