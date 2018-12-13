#---------------------------------------------------------------------------------------
# Name:        XML and SQLITE data into TSV
# Purpose:     This module is used to combine the data from one of the XML article
#              files (training/test/validation/crowd-sourced train/crowd-sourced test)
#              with the ground truth from a SQLITE3 table together in a TSV file.
#              The 'crowd sourced' XML files are xml files whose data has
#              been manually annotated for the SemEval 2019 Task 4.
#              IMPORTANT NOTE: this module changes hyperpartisan = true/false in the
#              ground truth sqlite table to 1/0.
#
# Execution:   python create_unified_tsv.py [-h]
#                {training,validation,test,crowdsourced_train,crowdsourced_test}
#
# Author:      Ashwath Sampath
#
# Created:     08-11-2018 (V1.0): Initial version which inserts only training
#                                 data into into the file
# Revisions:   22-11-2018 (V1.1): Added options to insert different XML data into
#                                 different files.
#                                 Code improved and made more user-friendly
#                                 Moved select_from_ground_truth to ground_truth_sqlite 
#                                 Stop writing bias, url and id into the output file
#              28-11-2018 (V1.2): Adding id to output file again, sem_eval_dir_path
#                                 option added, paths cleaned up.
#                                 
#              12-12-2018 (V1.3): Not adding hyperpartisan to the file in write_to_tsv
#                                 We don't have the ground truth in the test file.
#---------------------------------------------------------------------------------------

from lxml import etree
from bs4 import BeautifulSoup
import sqlite3
import argparse
import csv
import os
import ground_truth_sqlite

def get_xml_root(xml_filepath):
    """ Gets the root of the xml tree and returns it."""
    doc = etree.parse(xml_filepath)
    root = doc.getroot()
    return root

def get_xml_file_path(dataset_type, sem_eval_dir_path):
    """ Uses the dataset type to get the appropriate article XML file"""
    # NOTE: This function needs to be edited when the test set ground truth is available
    basepath = os.path.join(sem_eval_dir_path, 'data', 'Articles')
    if dataset_type == 'training':
        # Buzzfeed training ground truth
        return '{}/articles-training-bypublisher-20181122.xml'.format(basepath)
    elif dataset_type == 'validation':
        # Buzzfeed validation ground truth
        return '{}/articles-validation-bypublisher-20181122.xml'.format(basepath)
    elif dataset_type == 'crowdsourced_train':
        # Crowd-sourced training ground truth
        return '{}/articles-training-byarticle-20181122.xml'.format(basepath)
    # Buzzfeed test ground truth
    elif dataset_type == 'crowdsourced_test':
        return 'Dummy: not yet available'
    # Crowd-sourced testing ground truth
    return 'Dummy: not yet available'

def set_output_file(dataset_type, sem_eval_dir_path):
    """ Uses the dataset type to set the appropriate article output (tsv) filename"""
    # NOTE: This function needs to be edited when the test set ground truth is available
    basepath = os.path.join(sem_eval_dir_path, 'data', 'IntegratedFiles')
    if dataset_type == 'training':
        # Buzzfeed training ground truth
        return '{}/buzzfeed_training_withid.tsv'.format(basepath)
    elif dataset_type == 'validation':
        # Buzzfeed validation ground truth
        return '{}/buzzfeed_validation_withid.tsv'.format(basepath)
    elif dataset_type == 'crowdsourced_train':
        # Crowd-sourced training ground truth
        return '{}/crowdsourced_train_withid.tsv'.format(basepath)
    # Buzzfeed test ground truth
    elif dataset_type == 'crowdsourced_test':
        return '{}/buzzfeed_test_withid.tsv'.format(basepath)

    # Crowd-sourced testing ground truth
    return '{}/crowdsourced_test_withid'.format(basepath)


def write_to_tsv(output_tsv, xml_file):   
    """ Read from large xml file and a sqlite table containing additional fields from the
    ground truth (both based on table_name), write the results to to a tsv file."""

    #fieldnames = ['id', 'title', 'content', 'hyperpartisan', 'bias', 'url']
    # DO NOT WRITE THE BIAS AND THE URL into the output file. They won't be used as I think
    # they would increase the bias of the Buzzfeed journalists who created the data set.

    fieldnames = ['id', 'title', 'content']
    training_tsv = open(output_tsv, 'w', encoding='utf-8')
    training_writer = csv.DictWriter(training_tsv, delimiter='\t', fieldnames=fieldnames)
    #training_writer.writeheader()
    la_belle_soupe = BeautifulSoup(open(xml_file).read(),"lxml-xml")
    articles_list = la_belle_soupe.find_all("article")
    for i in range(len(articles_list)):
        current_article = articles_list[i]
        # article now contains all the fields of one article
        title = current_article['title'].replace('\t', ' ').replace('\n',' ')
        
        identifier = current_article['id']
        #published_date = current_article['published-at']
        # \n is present in the content. Remove it, or there will be major issues.
        text = current_article.get_text().replace('\t', ' ').replace('\n',' ')
        training_writer.writerow({'id': identifier, 'title': title, 'content': text})

def main():
    """ Main function which parses command-line arguments, 
    and inserts data from the appropriate ground truth xml file into it"""
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['training', 'validation', 'test',
                                        'crowdsourced_train', 'crowdsourced_test'],
                        help='Select the type of dataset to fetch from XML and SQLite')
    parser.add_argument("--path",'-p', default="/home/ashwath/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    args = parser.parse_args()
    sem_eval_dir_path = args.path
    db_path = os.path.join(sem_eval_dir_path, 'data', 'Databases', 'ground_truth.sqlite3')
    connection = ground_truth_sqlite.db_connect(db_path)
    # Results should be fetchable using column names as keys
    connection.row_factory = sqlite3.Row
    table_name = ground_truth_sqlite.set_sqlite_table_name(args.type)
    xml_file = get_xml_file_path(args.type, sem_eval_dir_path)
    output_tsv = set_output_file(args.type, sem_eval_dir_path)
    write_to_tsv(connection, output_tsv, table_name, xml_file)

if __name__ == '__main__':
    main()
