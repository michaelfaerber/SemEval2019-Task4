#------------------------------------------------------------------------------------
# Name:        Ground Truth XML to database
# Purpose:     This module is used to insert the data from one of the ground
#              truth XML files (training/test/validation/Crowd-sourced train/
#              crowd-sourced test) into a SQLITE3 database.
#              The 'Crowd-sourced' XML file are small xml files whose data has
#              been manually annotated for the SemEval 2019 Task 4.
#
# Execution:   python ground_truth_sqlite.py [-h] [--drop] [--nodrop]
#                  {training,validation,test,crowdsourced_train,crowdsourced_test}
#
# Author:      Ashwath Sampath
#
# Created:     08-11-2018 (V1.0): Initial version which inserts only training
#                                 ground truths into into SQLITE3
# Revisions:   22-11-2018 (V1.1): Added options to insert different ground truth
#                                 XML data into different SQLITE3 databses.
#                                 Code improved and made more user-friendly
#                                 Moved select_from_ground_truth from 
#                                 create_integrated_tsv
#              04-12-2018 (V1.2): Paths cleaned up
#
#------------------------------------------------------------------------------------

from lxml import etree
import sqlite3
import os
import sys
import argparse

def get_xml_root(xml_filepath):
    """ Gets the root of the xml tree and returns it."""
    doc = etree.parse(xml_filepath)
    root = doc.getroot()
    return root

def get_xml_file_path(dataset_type, sem_eval_dir_path):
    """ Uses the dataset type to get the appropriate ground truth XML file"""
    # NOTE: This function needs to be edited when the test set ground truth is available
    basepath = os.path.join(sem_eval_dir_path, 'data', 'GroundTruth')
    if dataset_type == 'training':
        # Buzzfeed training ground truth
        return '{}/ground-truth-training-bypublisher-20181122.xml'.format(basepath)
    elif dataset_type == 'validation':
        # Buzzfeed validation ground truth
        return '{}/ground-truth-validation-bypublisher-20181122.xml'.format(basepath)
    elif dataset_type == 'crowdsourced_train':
        # Crowd-sourced training ground truth
        return '{}/ground-truth-training-byarticle-20181122.xml'.format(basepath)
    # Buzzfeed test ground truth
    elif dataset_type == 'crowdsourced_test':
        return 'Dummy: not yet available'
    # Crowd-sourced testing ground truth
    return 'Dummy: not yet available'

def set_sqlite_table_name(dataset_type):
    """ Uses the dataset type to set the appropriate sqlite3 table name """
    if dataset_type == 'training':
        # Buzzfeed training ground truth
        return 'ground_truth_training'
    elif dataset_type == 'validation':
        # Buzzfeed validation ground truth
        return 'ground_truth_validation'
    elif dataset_type == 'crowdsourced_train':
        # Crowd-sourced training ground truth
        return 'ground_truth_crowdsourced_train'
    # Buzzfeed test ground truth
    elif dataset_type == 'test':
        return 'ground_truth_test'
    # Crowd-sourced testing ground truth
    return 'ground_truth_crowdsourced_test'

def db_connect(db_path):
    """ Connects to/creates the sqlite3 database at the path supplied
    as an argument """
    connection = sqlite3.connect(db_path, timeout=10,
                                 detect_types=sqlite3.PARSE_DECLTYPES)
    return connection

def create_ground_truth_table(conn, table_name):
    """ Function which takes a sqlite3 connection object and creates a table
    of the name table_name if it doesn't already exist. """
    cur = conn.cursor()
    ground_truth_sql = """
        CREATE TABLE {} (
                id text NOT NULL,
                hyperpartisan text,
                bias text,
                url text,
                PRIMARY KEY(id)
                )""".format(table_name)
    
    try:
        cur.execute(ground_truth_sql)
        print("Created {}".format(table_name))
    except sqlite3.OperationalError:
        print("Table '{}' already exists. Did you forget to use the -d or --drop option?"\
            .format(table_name))
        sys.exit()

def insert_into_ground_truth(conn, xml_root, table_name):
    """ Function which inserts into the appropriate ground_truth table.
    It takes a sqlite3 conn object, the root of the file's xml tree, and
    the table name as arguments. The xml elements have attributes, which
    are accessed like a dict using .get() and inserted into the table"""

    cur = conn.cursor()

    insert_ground_truth_sql = """
    INSERT INTO {}(id, hyperpartisan, bias, url)
    VALUES (?, ?, ?, ?) """.format(table_name)

    # Buzzfeed ground truth is of the foll. form:
    #<article id='0000037' hyperpartisan='false' bias='right-center'
    # url='https://cfr.org/report/bipartisan-work-plan' labeled-by='publisher'/>

    # Crowd-sourced ground truth is if the foll. form (no bias):
    # <article hyperpartisan="false" id="0000093" labeled-by="article"
    # url="http://merriam-webster.com/dictionary/freedom"/>


    for child in xml_root:
        # Get the id, title, url from the xml element article's attributes
        # For crod sourced, get('bias') will return None and will insert null
        # into the DB
        cur.execute(insert_ground_truth_sql, (child.get('id'),
                                              child.get('hyperpartisan'),
                                              child.get('bias') ,
                                              child.get('url')))
    try:
        conn.commit()
    except:
        print("Something went wrong while committing, attempting to rollback!")
        conn.rollback()

def get_count_star_table(conn, table_name):
    """ Gets the number of rows in a particular table"""
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM {}'.format(table_name))
    return cur.fetchone()[0]

def drop_ground_truth(conn, table_name):
    """ Drops the table mentioned in the argument if it exists"""
    cur = conn.cursor()
    cur.execute('drop table if exists {}'.format(table_name))
    print("Dropped {}".format(table_name))

def select_from_ground_truth(conn, identifier, table_name):
    """ This function queries the sqlite3 table specified in the arguments
    for the identifier, and returns the results in a tuple of the form
    (id, bias, hyperpartisan, url) """
    cur = conn.cursor()
    query = """
    SELECT * 
    FROM {} 
    WHERE id = '{}' """.format(table_name, identifier)
    # df = pd.read_sql_query(query, conn)
    cur.execute(query)
    # Returns a tuple of form (id, bias, hyperpartisan, url)
    return cur.fetchone()

def select_id_hyperpartisan_mappings(sem_eval_dir_path, table_name):
    """ This function queries the sqlite3 table specified in the arguments
    for a mapping of article ids to hyperpartisans, and returns the results in a dictionary where key is the id and hyperpartisan value is the value """
    db_path = os.path.join(sem_eval_dir_path, 'data', 'Databases', 'ground_truth.sqlite3')
    conn = db_connect(db_path)
    cur = conn.cursor()
    query = """
    SELECT id, hyperpartisan 
    FROM {} """.format(table_name)
    # df = pd.read_sql_query(query, conn)
    cur.execute(query)

    mappings = {}
    for row in cur:
        mappings[int(row[0])] = row[1]
        
    return mappings

def main():
    """ Main function which creates the appropriate ground truth sqlite table based
    on command-line args, and inserts data from the appropriate ground truth xml file
    into it"""
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['training', 'validation', 'test',
                                        'crowdsourced_train', 'crowdsourced_test'],
                        help='Select the type of dataset to insert into SQLITE3')
    parser.add_argument("--drop", '-d', action="store_true", default="False",
                        help="Use this argument to drop the table")
    parser.add_argument("--nodrop",'-n', action="store_true", default="True",
                        help="Use this argument to not drop the table (this is the "
                        "default behaviour)") 
    parser.add_argument("--path",'-p', default="/home/ashwath/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    args = parser.parse_args()
    sem_eval_dir_path = args.path
    db_path = '{}/data/Databases/ground_truth.sqlite3'.format(sem_eval_dir_path)
    db_path = os.path.join(sem_eval_dir_path, 'data', 'Databases', 'ground_truth.sqlite3')
    connection = db_connect(db_path)
    # Create appropriate ground truth table based on command line argument
    table_name = set_sqlite_table_name(args.type)
    if args.drop is True:
        if input("Are you sure you want to drop the table '{}' (y or n)? "\
            .format(table_name)).lower() == 'y':
            drop_ground_truth(connection, table_name)
        else:
            print("Exiting program.")
            sys.exit()
    create_ground_truth_table(connection, table_name)
    # Get xml file path and name based on 'type' argument and get the xml root
    ground_truth_xmlpath = get_xml_file_path(args.type, sem_eval_dir_path)
    ground_root = get_xml_root(ground_truth_xmlpath)
    # Insert into appropriate ground truth table
    insert_into_ground_truth(connection, ground_root, table_name)
    print("No. of rows inserted into table '{}' = {} ".format(table_name, \
        get_count_star_table(connection, table_name)))
    connection.close()

if __name__ == '__main__':
    main()