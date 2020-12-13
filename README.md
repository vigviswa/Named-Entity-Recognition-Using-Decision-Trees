# Named Entity Recognition Using Decision Trees

This repositiory covers how to extract the Named Entities in the SemEval2010 - Task 8 Dataset. You can look up the Dataset [here](https://github.com/vigviswa/Named-Entity-Recognition-Using-Decision-Trees/tree/main/data). Also, the SemEval2010 - Task 8 statement can be read [here](https://semeval2.fbk.eu/semeval2.php?location=tasks). 


### Quickstart and Summary

For the implementation, the task has been divided into two parts:

### Part 1 and Part 2:

Creation of the Corpus Reader Class to parse the data from the dataset into a DataFrame for the Model.

Some Common NLP Tasks such as: POS Identification, Dependency Parsing, Full Synctactic Parsing, HyperNym, HoloNym, MeroNym and HypoNym Extraction, etc.

### Part 3:

Creation of the Decision Tree Model to perform the Relation Classification and Identification.

### Steps to Run the Code:

#### For Task 1 & 2:

1) Download the Submission. The code file for this task is called, `Task1_2Demo.py`. 

2) Create a File called "test_sentence.txt" containing the test sentences for which you want to 
run the Task 1 and Task 2.

3) Save this file in the same directory as the code, i.e., `/Code`.

4) run 

```
pip install -r requirements.txt
```

run  
```
python -m spacy download en_core_web_sm
```

To Download all packages and dependencies

5) Run the Code as `python Task1_2Demo.py` and you will see all the outputs printed on the console or on
the IDE you are using.

6) This code can also be run on Google Colab as:

a) Open Google Colab, upload the test_sentence.txt

b) Import the Task1_2Demo.ipynb on Colab. You should be able to run all the tasks.

c) The Notebook Link is clickable [here:](https://colab.research.google.com/drive/1gNE2BdGURa12U1-2Ai8JZEyXkeAyUXG9?usp=sharing)


#### For Task 3:

1) Open the IDE of your choice. And run the Command `python -m spacy download en_core_web_sm`

2) Download the SemEval dataset available on E-Learning and put in the same directory, i.e, `/Code`
and name it as `semeval_train.txt`

3) Put all your test sentences in a file called `semeval_test.txt` or to run this on the entire test set,
Download the file available on e-learning and rename as `semeval_test.txt`

4) Once you copy the test sentences, just make sure that you have run the pip install step as before.

5) Run the code as `python Task3Demo.py`
