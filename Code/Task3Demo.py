import time
import nltk
from itertools import islice
import re
import pandas as pd
import spacy
from nltk.corpus import wordnet as wn
import time

nltk.download('wordnet')


nlp = spacy.load("en_core_web_sm")
sp = spacy.load('en_core_web_sm')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

"""# ALL Functions to Create the dataframes and Parse the Data"""

# Creates the Dictionaries from the Path of the File

def create_dicts(path):
    """
      Returns 2 Dictionaries. Once for the Line and One For the Relation in the File.
      This is based on how the Data is organized in the supplied Data Files.
    """
    line_d = {}
    rel_d = {}

    with open(path) as f:
        for line in islice(f, 0, None, 4):
            lister = line.split('"')
            line_number = int(lister[0].split('\t')[0])
            line_d[line_number] = ''.join(str(s) for s in lister[1:])

    with open(path) as f:
        for i, line in enumerate(islice(f, 1, None, 4)):
            rel_d[i] = line.split('\n')[0]

    return (line_d, rel_d)


def create_dataframe(dictionary_to_convert, cols):
    """
      From a Dictionary which is passed, and the desired column to create, this function
      returns a Dataframe.
    """

    dataframe_converted = pd.DataFrame.from_dict(dictionary_to_convert, orient='index', columns=cols)
    dataframe_converted = dataframe_converted.reset_index()
    dataframe_converted = dataframe_converted.drop(columns=['index'])

    return dataframe_converted


def parse_data(path_to_file):
    """
      Invokes the Create Dict and Create Data Frame Function.
      This function is designed to create the Line and Relation Dataframe
    """

    line_dict, rel_dict = create_dicts(path_to_file)

    line_df = create_dataframe(line_dict, ['line'])
    rel_df = create_dataframe(rel_dict, ['relation'])

    line_df['relation'] = rel_df['relation']

    return (line_df, rel_df)


"""# Window Creation"""


def create_window(dataframe):
    """

      For A DataFrame with the column 'line', this function will create a Window
      of the words from E1 Tag - 1 to E2 Tag + 1 words.

      This window will be added as a New Column Named 'Line Window' in the DataFrame and will be returned

    """

    window_dict = {}
    for i, val in enumerate(dataframe['line']):
        e1 = re.findall('<e1>(.*?)</e2>', val)
        before = re.findall('\w* ?<e1>', val)
        after = re.findall('</e2> ?\w*', val)
        bef = before[0].replace('<e1>', '')
        aft = after[0].replace('</e2>', '')
        s = e1[0].replace('</e1>', '').replace('<e2>', '')
        window_dict[i] = bef + s + aft

    window_dataframe = create_dataframe(window_dict, ['window'])

    dataframe['line window'] = window_dataframe['window']
    return dataframe


"""# Tokenization"""


# Adding a column of tokens to the dataframe
def create_tokens(dataframe):
    """

      For A DataFrame with the column 'line', this function will create tokens
      of the words in that line

      These tokens will be added as a New Column Named 'Tokens' in the DataFrame and will be returned

    """

    tokenize_dict = {}
    iterator = dataframe.to_dict('dict')['line']

    for key, val in iterator.items():
        tokenize_dict[key] = nltk.word_tokenize(val)

    for key, val in tokenize_dict.items():
        l = []
        for i in range(len(val)):
            if val[i] == '<':
                val[i] = ''.join(val[i:i + 3])

            l = [e for e in val if e not in ('e1', 'e2', '/e1', '/e2', '>')]
            tokenize_dict[key] = ', '.join(str(s) for s in l)

    tokenize_dataframe = create_dataframe(tokenize_dict, ['token'])

    dataframe['tokens'] = tokenize_dataframe['token']

    return dataframe


"""# POS And Dep Parse"""


def create_pos_dep(dataframe, col):
    """

      For A DataFrame with the window created, this function will add the POS and Dep Tags of those words.

      These values will be added as Two Columns Named 'pos' and 'dep' in the DataFrame and will be returned.

    """
    pos_dict = {}
    dep_dict = {}
    p = []
    d = []
    for i, val in enumerate(dataframe[col]):
        s = sp(''.join(val).replace(',', ''))
        for word in s:
            p.append(word.pos_)
            d.append(word.dep_)
        pos_dict[i] = ', '.join(str(s) for s in p)
        dep_dict[i] = ', '.join(str(s) for s in d)
        p = []
        d = []

    colname1 = col + '_pos' if col in ['e1', 'e2'] else 'pos'
    colname2 = col + '_dep' if col in ['e1', 'e2'] else 'dep'
    pos_dataframe = create_dataframe(pos_dict, [colname1])
    dep_dataframe = create_dataframe(dep_dict, [colname2])

    dataframe[colname1] = pos_dataframe[colname1]
    dataframe[colname2] = dep_dataframe[colname2]

    return dataframe


"""# NER"""


def create_NER(dataframe):
    """

      For A DataFrame with line, this function will extract both the entities.

      These values will be added as Two Columns Named 'e1' and 'e2' in the DataFrame and will be returned.

    """

    dataframe['entities'] = dataframe['line']
    entity_dict = {}
    for i, val in enumerate(dataframe['entities']):
        e1 = re.findall('<e1>(.*?)</e1>', val)
        e2 = re.findall('<e2>(.*?)</e2>', val)
        entity_dict[i + 1] = (str(e1[0]), str(e2[0]))

    entity_dataframe = create_dataframe(entity_dict, ['e1', 'e2'])
    dataframe = dataframe.drop(columns=['entities'])
    dataframe['e1'] = entity_dataframe['e1']
    dataframe['e2'] = entity_dataframe['e2']

    return dataframe


"""# HyperNym and HoloNym"""


def create_syn(dataframe, col):
    """

      For A DataFrame with col, this function will extract synsets of both the entities.

      These values will be added as Two Columns Named 'e1_syn' and 'e2_syn' in the DataFrame and will be returned.

    """

    hypernym_e = {}
    hyper = []
    for i, val in enumerate(dataframe[col]):
        if wn.synsets(val):
            syn = wn.synsets(val)[0]
            s = syn.hypernyms()
            if s:
                for val in s:
                    hyper.append(str(val)[8:-3].split('.')[0])

                hypernym_e[i] = ', '.join(v for v in hyper)
                hyper = []
            else:
                hypernym_e[i] = 'None'
        else:
            hypernym_e[i] = 'None'

    colname = col + '_syn'

    hypernym_e_dataframe = create_dataframe(hypernym_e, [colname])
    dataframe[colname] = hypernym_e_dataframe[colname]

    return dataframe


# Creation of the Test and Training Data. Just Pass in the Path to the respective files

line_dataframe, relation_dataframe = parse_data("semeval_train.txt")
line_test_dataframe, relation_test_dataframe = parse_data("semeval_test.txt")

# Splitting the Relation Column into just the relation, for Accuracy Checking Purpose

line_dataframe['just_relation'] = [x.split('(')[0] for x in line_dataframe['relation']]
line_test_dataframe['just_relation'] = [x.split('(')[0] for x in line_test_dataframe['relation']]

line_dataframe = create_window(line_dataframe)
line_test_dataframe = create_window(line_test_dataframe)

# Adding the Tokens Column

line_dataframe = create_tokens(line_dataframe)
line_test_dataframe = create_tokens(line_test_dataframe)

# Adding POS and Dep Columns

line_dataframe = create_pos_dep(line_dataframe, 'line window')
line_test_dataframe = create_pos_dep(line_test_dataframe, 'line window')

# Entity Extraction

line_dataframe = create_NER(line_dataframe)
line_test_dataframe = create_NER(line_test_dataframe)

line_dataframe = create_syn(line_dataframe, 'e1')
line_dataframe = create_syn(line_dataframe, 'e2')
line_test_dataframe = create_syn(line_test_dataframe, 'e1')
line_test_dataframe = create_syn(line_test_dataframe, 'e2')


# Create the Model

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

# Initializing count vectorizer
cv = CountVectorizer()

line_dataframe_tr = line_dataframe.iloc[0:7000, :]
line_dataframe_test = line_test_dataframe.iloc[0:5000, :]

# Training and Testing dataframe extraction
x_train_df = line_dataframe_tr[["pos", "dep", "e1_syn", "e2_syn"]]
x_test_df = line_dataframe_test[["pos", "dep", "e1_syn", "e2_syn"]]

# Transforming input columns in training and testing dataframe.
cvec = cv.fit(x_train_df["pos"])
pos_train = pd.DataFrame(cvec.transform(x_train_df["pos"]).todense())
pos_test = pd.DataFrame(cvec.transform(x_test_df["pos"]).todense())

cvec = cv.fit(x_train_df["dep"])
dep_train = pd.DataFrame(cvec.transform(x_train_df["dep"]).todense())
dep_test = pd.DataFrame(cvec.transform(x_test_df["dep"]).todense())

cvec = cv.fit(x_train_df["e1_syn"])
e1_train = pd.DataFrame(cvec.transform(x_train_df["e1_syn"]).todense())
e1_test = pd.DataFrame(cvec.transform(x_test_df["e1_syn"]).todense())

cvec = cv.fit(x_train_df["e2_syn"])
e2_train = pd.DataFrame(cvec.transform(x_train_df["e2_syn"]).todense())
e2_test = pd.DataFrame(cvec.transform(x_test_df["e2_syn"]).todense())

x_train = pd.concat([dep_train, e1_train, e2_train, pos_train], axis=1)
x_test = pd.concat([dep_test, e1_test, e2_test, pos_test], axis=1)

le = LabelEncoder()
le.fit(line_dataframe_tr["relation"])
y_train = le.transform(line_dataframe_tr["relation"])
y_test = le.transform(line_dataframe_test["relation"])

# Decision tree model
"""
start = time.time()
model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=30, class_weight='balanced')
model.fit(x_train, y_train)
test_predictions = model.predict(x_test)
end = time.time()
precisionScore = precision_score(y_test, test_predictions, average='micro')
recallScore = recall_score(y_test, test_predictions, average='micro')
f1Score = f1_score(y_test, test_predictions, average='micro')
accuracyScore = accuracy_score(y_test, test_predictions)
print("Decision tree model:")
print("\n")
print("Accuracy for relations with direction: ")
print(accuracyScore*100)
print("\n")
print("Precision: ")
print(precisionScore)
print("\n")
print("Recall: ")
print(recallScore)
print("\n")
print("F1-Score ")
print(f1Score)
print("\n")
print("Time taken for model fitting and prediction in seconds: ")
print(end-start)


start = time.time()
model = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=30, class_weight='balanced')
model.fit(x_train, y_train)
test_predictions = model.predict(x_test)
end = time.time()
precisionScore = precision_score(y_test, test_predictions, average='micro')
recallScore = recall_score(y_test, test_predictions, average='micro')
f1Score = f1_score(y_test, test_predictions, average='micro')
accuracyScore = accuracy_score(y_test, test_predictions)
print("Decision tree model:")
print("\n")
print("Accuracy for relations with direction: ")
print(accuracyScore*100)
print("\n")
print("Precision: ")
print(precisionScore)
print("\n")
print("Recall: ")
print(recallScore)
print("\n")
print("F1-Score ")
print(f1Score)
print("\n")
print("Time taken for model fitting and prediction in seconds: ")
print(end-start)


start = time.time()
model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=30, class_weight='balanced')
    , n_estimators=5)
model.fit(x_train, y_train)
test_predictions = model.predict(x_test)
end = time.time()
precisionScore = precision_score(y_test, test_predictions, average='micro')
recallScore = recall_score(y_test, test_predictions, average='micro')
f1Score = f1_score(y_test, test_predictions, average='micro')
accuracyScore = accuracy_score(y_test, test_predictions)
print("\n")
print("\n")
print("Bagging Classifier (Decision tree) model:")
print("\n")
print("Accuracy for relations with direction: ")
print(accuracyScore*100)
print("\n")
print("Precision: ")
print(precisionScore)
print("\n")
print("Recall: ")
print(recallScore)
print("\n")
print("F1-Score ")
print(f1Score)
print("\n")
print("Time taken for model fitting and prediction in seconds: ")
print(end-start)


start = time.time()
model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=30, class_weight='balanced')
    , n_estimators=5)
model.fit(x_train, y_train)
test_predictions = model.predict(x_test)
end = time.time()
precisionScore = precision_score(y_test, test_predictions, average='micro')
recallScore = recall_score(y_test, test_predictions, average='micro')
f1Score = f1_score(y_test, test_predictions, average='micro')
accuracyScore = accuracy_score(y_test, test_predictions)
print("\n")
print("\n")
print("Bagging Classifier (Decision tree) model:")
print("\n")
print("Accuracy for relations with direction: ")
print(accuracyScore*100)
print("\n")
print("Precision: ")
print(precisionScore)
print("\n")
print("Recall: ")
print(recallScore)
print("\n")
print("F1-Score ")
print(f1Score)
print("\n")
print("Time taken for model fitting and prediction in seconds: ")
print(end-start)
"""

start = time.time()
model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=50, class_weight='balanced')
    , n_estimators=10)
model.fit(x_train, y_train)
test_predictions = model.predict(x_test)
end = time.time()
precisionScore = precision_score(y_test, test_predictions, average='micro')
recallScore = recall_score(y_test, test_predictions, average='micro')
f1Score = f1_score(y_test, test_predictions, average='micro')
accuracyScore = accuracy_score(y_test, test_predictions)
print("\n")
print("\n")
print("Boosting Classifier (Decision tree) model:")
print("\n")
print("Accuracy for relations with direction: ")
print(accuracyScore*100)
print("\n")
print("Precision: ")
print(precisionScore)
print("\n")
print("Recall: ")
print(recallScore)
print("\n")
print("F1-Score ")
print(f1Score)
print("\n")
print("Time taken for model fitting and prediction in seconds: ")
print(end-start)

# Printing the output for the test file.
y_test = y_test.tolist()
test_predictions = test_predictions.tolist()

actual_relation_test = [le.inverse_transform([val])[0] for val in y_test]
predicted_relation_test = [le.inverse_transform([val])[0] for val in test_predictions]

actual_relationNoDir_test = [x.split('(')[0] for x in actual_relation_test]
predicted_relationNoDir_test = [x.split('(')[0] for x in predicted_relation_test]

line_test_dataframe["PredictedRelationWD"] = predicted_relation_test
line_test_dataframe["PredictedRelationWOD"] = predicted_relationNoDir_test

accuracyScore = accuracy_score(actual_relationNoDir_test, predicted_relationNoDir_test)

print("\n")
print("Accuracy for just relations: ")
print(accuracyScore*100)
print("\n")

for index in line_test_dataframe.index:
    print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")
    print("Sentence :", index + 1, "\n")
    print("Sentence             : ", line_test_dataframe['line'][index], "\n")
    print("Relation             : ", line_test_dataframe['relation'][index], "\n")
    print("Predicted Relation   : ", line_test_dataframe['PredictedRelationWD'][index], "\n")
    print("\n----------------------------------------------------------------------------------------------------------------------------------------\n")
