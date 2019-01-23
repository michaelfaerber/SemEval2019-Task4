import pandas as pd
import os
import clean_shuffle
import create_unified_tsv
import getopt
import sys
from time import sleep
import logging
from keras.models import load_model
from data_loaders import TokenizerLoader, TextSequencesLoader
from datetime import datetime

inputFileName = 'articles-validation-bypublisher-20181122'
runOutputFileName = "prediction.txt"
sem_eval_path = '/home/peter-brinkmann/'
logging.basicConfig(filename='{}/logs/info_log.log'.format(sem_eval_path), filemode='w', 
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
seq_len = 5000
model_file_name = 'words_conv_model_w0_v5.h5'

def _parse_options():
    """Parses the command line options."""
    try:
        long_options = ["inputDataset=", "outputDir="]
        opts, _ = getopt.getopt(sys.argv[1:], "d:o:", long_options)
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    inputDataset = "undefined"
    outputDir = "undefined"

    for opt, arg in opts:
        if opt in ("-d", "--inputDataset"):
            inputDataset = arg
        elif opt in ("-o", "--outputDir"):
            outputDir = arg
        else:
            assert False, "Unknown option."
    if inputDataset == "undefined":
        sys.exit("Input dataset, the directory that contains the articles XML file, is undefined. Use option -d or --inputDataset.")
    elif not os.path.exists(inputDataset):
        sys.exit("The input dataset folder does not exist (%s)." % inputDataset)

    if outputDir == "undefined":
        sys.exit("Output path, the directory into which the predictions should be written, is undefined. Use option -o or --outputDir.")
    elif not os.path.exists(outputDir):
        os.mkdir(outputDir)

    return (inputDataset, outputDir)

def _convert_xml_to_tsv(inputDataset):
    xml_file = os.path.join(inputDataset, '{}.xml'.format(inputFileName))
    tsv_file = os.path.join(sem_eval_path, 'data', 'IntegratedFiles', '{}.tsv'.format(inputFileName))
    logging.info('Converting xml to tsv. XML file: {}. TSV file: {}.'.format(xml_file, tsv_file))
    create_unified_tsv.write_to_tsv(tsv_file, xml_file)
    sleep(2)
    return tsv_file

def _convert_tsv_to_dataframe(tsv_file):
    df_location = os.path.join(sem_eval_path, 'data', 'Pickles', '{}_df.pickle'.format(inputFileName))
    logging.info('Converting tsv to dataframe. Df location: {}.'.format(df_location))
    df = clean_shuffle.read_prepare_test_df(tsv_file, file_path=df_location)
    logging.info('Dataframe created. Shape: {}.'.format(df.shape))
    # df.sort_values('id', inplace=True)
    return df

def _convert_texts_to_sequences(df):
    texts = df['content'] + df['title']
 
    logging.info('Loading tokenizer')
    tokenizer = TokenizerLoader(texts, sem_eval_path).load()
 
    logging.info('Converting texts to sequences')
    sequences_loader = TextSequencesLoader(tokenizer, seq_len)
    X_test = sequences_loader.load(texts)

    return X_test

def _predict(model, X_val):
    """ Predicts the labels for the validation set using the given model
    ARGUMENTS: model: an sklearn model
               X_val: the validation matrix for which labels have to be predicted
    RETURNS: y_pred: predicted labels Pandas series"""
    logging.info('Predicting values')
    predicted_values = model.predict_classes(X_val)
    formatted_pred = predicted_values.reshape((-1,))
    return pd.Series(formatted_pred)

def _create_output_dataframe(input_df, y_pred):
    logging.info('Creating output dataframe')
    # Convert 0 and 1 back to true and false (as it was in the xml file)
    # ATTENTION: we don't need to convert it to 0 and 1 in the previous step any more.
    truefalsedict = {0: 'false', 1: 'true'}
    y_pred_df = pd.DataFrame(y_pred, columns=['predicted_hyperpartisan'])
    y_pred_df['predicted_hyperpartisan'] = y_pred_df['predicted_hyperpartisan'].map(truefalsedict, na_action=None)
    # The order of ids will be the same, also add leading zeros (to make it like the input dataset)
    y_pred_df['id'] = input_df['id'].astype(str).str.zfill(7)
    # Reorder the columns
    y_pred_df = y_pred_df[['id', 'predicted_hyperpartisan']]
    return y_pred_df

def _write_output_dataframe_to_file(y_pred_df, outfile):
    logging.info('Writing output dataframe to file')
    y_pred_df.to_csv(outfile, sep=' ', index=False, header=False)

########## MAIN ##########
# Total script duration on shetland: 00h:56m:30s
def main(inputDataset, outputDir):
    startTime = datetime.now()

    global inputFileName
    for file in os.listdir(inputDataset):
        if file.endswith(".xml"):
            # inputFileName = inputDataset + "/" + file
            inputFileName = file.replace('.xml', '')
            logging.info('File: {}'.format(inputFileName))

    # Convert xml test file to tsv format
    # Duration on shetland: 00h:02m:23s
    # Duration on TIRA: 00h:16m:30s
    input_tsv = _convert_xml_to_tsv(inputDataset)

    # Convert tsv file to dataframe
    # Duration on shetland: 00h:33m:29s
    # Duration on TIRA: 04h:53m:30s
    input_df = _convert_tsv_to_dataframe(input_tsv)

    # Convert texts to sequences
    # Duration on shetland: 00h:01m:33s 
    # Duration on TIRA: 00h:06m:28s 
    X_test = _convert_texts_to_sequences(input_df)

    # Load model
    model = load_model(os.path.join(sem_eval_path, 'models', model_file_name))

    # Do the prediction
    # Duration on shetland: 00h:19m:52s
    # Duration on TIRA: 00h:24m:52s
    y_pred = _predict(model, X_test)

    # Create output dataframe to write on disk
    y_pred_df = _create_output_dataframe(input_df, y_pred)
    
    # Write preditions df to file
    outfile = outputDir + "/" + runOutputFileName
    _write_output_dataframe_to_file(y_pred_df, outfile)
    print("The predictions have been written to the output folder.")

    # Write preditions df to SemEval directory
    outfile = '{}/data/'.format(sem_eval_path) + runOutputFileName
    _write_output_dataframe_to_file(y_pred_df, outfile)
    print("The predictions have been written to SemEval folder.")

    print(datetime.now() - startTime)


if __name__ == '__main__':
    main(*_parse_options())
