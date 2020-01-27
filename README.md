**This repository contains the source code developed by the team "Peter Brinkmann" for participating in the [SemEval 2019 Task 4: Hyperpartisan News Detection](https://webis.de/events/semeval-19/). More information about the implemented approach is available in our [paper](http://dbis.informatik.uni-freiburg.de/content/team/faerber/papers/SemEval_2019_Task_4.pdf).**

Note: The docstrings (module and function level) and inline comments in the modules provide additional explanations.


**INPUT FILES**
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


## STEP 1: Import Data into DB
Insert the data from one of the ground truth XML files (training/test/validation/Crowd-sourced train/crowd-sourced test) into a SQLITE3 database.
```
python(3) ground_truth_sqlite.py [-h] [--drop] [--nodrop]
                  {training,validation,test,crowdsourced_train,crowdsourced_test}
```
## STEP 2: Combine Articles with Ground Truth Information
combine the data from one of the XML article files (training/test/validation/crowd-sourced train/crowd-sourced test)
with the ground truth from a SQLITE3 table together in a TSV file. It also changes hyperpartisan = true/false in the
ground truth sqlite table to 1/0.
```
python create_unified_tsv.py [-h]
                {training,validation,test,crowdsourced_train,crowdsourced_test}
```
## STEP 3: Train the Deep Learning Model
```
python(3) train_words_dl_model.py -p {data directory}
```
This will train a CNN model using Google's words2vec word vectors. 
 - Use `-a` argument to train a different model: 
    1. `-a 1` - a hybrid CNN-LSTM model, or 
    2. `-a 2` - a LSTM model.
 - Use `-w` argument to use a different embedding layer, like custom pretrained word2vec, Stanford's Glove or Fasttext.

The script works through these steps: 
  1. Read the processed articles and train a tokenizer on them. Store the trained tokenizer on disk. Next time the tokenizer will be loaded from disk.
  2. The articles will then be converted to sequences using the trained tokenizer.
  3. Define the DL model (CNN, LSTM or CNN-LSTM hybrid based on the -a algorithm).
  4. Train the model until convergence. Store the trained tokenizer on disk. Next time the model will be loaded from disk.

To evaluate the model, use `-e` option.  This will run the script on evaluation mode, which loads the trained model from disk and runs it against the validation data to get the model's evaluation metrics. The metrics will be printed in a log file.

## References & Contact
If you use our code or would like to referene our work, please cite it as follows:
```
@inproceedings{Faerber2018SemEval,
  author    = {Michael F{\"{a}}rber and
               Agon Qurdina and
               Lule Ahmedi},
  title     = "{Team Peter Brinkmann at SemEval-2019 Task 4: Detecting Biased News
               Articles Using Convolutional Neural Networks}",
  booktitle = "{Proceedings of the 13th International Workshop on Semantic Evaluation}",
  series    = "{SemEval@NAACL-HLT'19}",
  location  = "{Minneapolis, MN, USA}",
  pages     = {1032--1036},
  year      = {2019},
  url       = {https://www.aclweb.org/anthology/S19-2180/}
}
```
Feel free to get in touch with us in case of questions of requests:
[Michael Färber](https://sites.google.com/view/michaelfaerber/), Karlsruhe Institute of Technology, Germany
