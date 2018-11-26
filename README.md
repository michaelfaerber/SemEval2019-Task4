Created: 26 November 2017

**see also the todo_issues.txt**


The method used for the hyperpartisan classification problem is two-fold.
__PART 1__: 2 shallow neural networks based on Mikolov/Le's Paragraph vectors are trained using the gensim library's Doc2Vec class
__PART 2__: The embeddings obtained from Doc2Vec are combined and an SVC model is used for classification


Now it's time to look at the steps to execute the program. An accompanying description will be provided for each step. \
The docstrings (module and function level) and inline comments in the modules provide additional explanations.


**INPUT FILES:**
The input files are XML files. The ground truth is separated from the articles.
The Article files contain the title and the content, along with an identifier for each article.
The Ground truth files contain a hyperpartisan indicator (true or false), a bias indicator (left, right, left-centre, right-centre),
a URL (link to the article) and an identifier.
This identifier is the key which is used to join each Article with its corresponding ground truth.

There are 3 such pairs of files which have been supplied:

1. articles-training-byarticle-20181122.xml, ground-truth-training-byarticle-20181122.xml (Small data set annotated by a crowdsourcing effort arranged by the organizers of SemEval 4)
2. articles-training-bypublisher-20181122.xml, ground-truth-training-bypublisher-20181122.xml
3. articles-validation-bypublisher-20181122.xml, ground-truth-validation-bypublisher-20181122.xml

The 2nd and the 3rd pairs are large training and validation files from the Buzzfeed data set.


## STEP 1: 
Insert the data from one of the ground truth XML files (training/test/validation/Crowd-sourced train/crowd-sourced test) into a SQLITE3 database.
```
python(3) ground_truth_sqlite.py [-h] [--drop] [--nodrop]
                  {training,validation,test,crowdsourced_train,crowdsourced_test}
```
## STEP 2: 
combine the data from one of the XML article files (training/test/validation/crowd-sourced train/crowd-sourced test)
with the ground truth from a SQLITE3 table together in a TSV file. It also changes hyperpartisan = true/false in the
ground truth sqlite table to 1/0.
```
python create_unified_tsv.py [-h]
                {training,validation,test,crowdsourced_train,crowdsourced_test}
```
## STEP 3: 
train embeddings based on Doc2Vec. Two separate embeddings are obtained for the title and the content.
These are then used to train an SVC model. Both embedding models and the SVC model are committed to disk.

NOTE: this uses the clean_shuffle module to create a dataframe in which data is cleaned and shuffled.
      It creates paragraph vectors in the para2vec module, which is explained at the end.
```
python training.py
```
## STEP 4: 
performs validation using the provided validation set, and calculates a number of metrics. Further, the hand-prepared training file with 645 records is used as a second validation file, mimicking the 2 test files
```
python validation.py
```

MODULES which are used internally.

1. *clean_shuffle.py*: create a dataframe from the input file in which data is cleaned and shuffled

2. *para2vec.py*: The most important of all the modules. 
It creates 2 doc2vec dbow models for the content and the title. Docs (title/content) are tagged to their hyperpartisan
indicator (0/1 after Step 2), and only then is the model trained. 
It also includes functions to combine embeddings and to map document vectors to their original labels.

Both doc2vec models have been tested extensively with various hyperparameters by splitting the training set (not on the validation set), and this should be the best possible (the only thing that might be reduced is the 'sample' hyperparameter for the content model)

**VERY IMPORTANT:** By tagging documents with their hyperpartisan indicator (which has 2 values), we may be restricting the output vectors. Instead, it might be possible to tag with the article ids and then introduce the labels at the SVC stage.
SEE todo_issues.txt (2nd issue) for more details.
