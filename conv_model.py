from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Activation, Embedding, Flatten
from keras.engine.input_layer import Input
import argparse
import clean_shuffle
from para2vec import ParagraphVectorModel, get_vector_label_mapping
from gensim.models.doc2vec import Doc2Vec
import numpy
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sem_eval_path = ''
dataset_type = ''


def load_trained_embeddings():
    model_content_dbow = Doc2Vec.load('{}/embeddings/doc2vec_dbow_model_content'.format(sem_eval_path))
    model_title_dbow = Doc2Vec.load('{}/embeddings/doc2vec_dbow_model_title'.format(sem_eval_path))
    return model_title_dbow, model_content_dbow

def get_tsv_path(mode='train'):
    basepath = '{}/data/IntegratedFiles'.format(sem_eval_path)
    if dataset_type == 'training':
        if mode == 'train':
            return '{}/buzzfeed_training.tsv'.format(basepath)
        else:
            return '{}/buzzfeed_validation.tsv'.format(basepath)
    elif dataset_type == 'crowdsourced_training':
        return  '{}/crowdsourced_train.tsv'.format(basepath)
    return 'Dummy: not yet available'

def generate_data():
    data_file = get_tsv_path()
    
    val_df = clean_shuffle.read_prepare_df(data_file)
    
    print('df created')

    model_title_dbow, model_content_dbow = load_trained_embeddings()
    
    print('embeddings loaded')

    pv = ParagraphVectorModel(val_df, init_models=False)
    pv.get_tagged_docs()
    print('get_tagged_docs()')
    pv.model_content_dbow = model_content_dbow
    pv.model_title_dbow = model_title_dbow
    print('Starting get_vector_label_mapping()')
    X_train, y_train = get_vector_label_mapping(pv)
    print('get_vector_label_mapping() completed')

    # Needed to reshape from (n, 300) to (n, 300, 1) to serve as an input to the Conv layer
    X_train = numpy.atleast_3d(X_train)
    print('df reshaped')

    return (X_train, y_train)

def load_test_data():
    data_file = get_tsv_path(mode='test')
    
    val_df = clean_shuffle.read_prepare_df(data_file)
    
    print('df created')

    model_title_dbow, model_content_dbow = load_trained_embeddings()
    
    print('embeddings loaded')

    pv = ParagraphVectorModel(val_df, init_models=False)
    pv.get_tagged_docs()
    print('get_tagged_docs()')
    pv.model_content_dbow = model_content_dbow
    pv.model_title_dbow = model_title_dbow
    print('Starting get_vector_label_mapping()')
    X_test, y_test = get_vector_label_mapping(pv)
    print('get_vector_label_mapping() completed')

    # Needed to reshape from (n, 300) to (n, 300, 1) to serve as an input to the Conv layer
    X_test = numpy.atleast_3d(X_test)
    print('df reshaped')

    return (X_test, y_test)

def load_pretrained_data(generate_new_data):
    X_train_path = '{}/final_data/X_train'.format(sem_eval_path)
    y_train_path = '{}/final_data/y_train'.format(sem_eval_path)

    if generate_new_data is True:
        X_train, y_train = generate_data()
        
        print('Dumping data to: {}'.format(X_train_path))
        pickle.dump(X_train, open(X_train_path, 'wb'))
        pickle.dump(y_train, open(y_train_path, 'wb'))
    else:
        X_train = pickle.load(open(X_train_path, 'rb'))
        y_train = pickle.load(open(y_train_path, 'rb'))
    
    boundary = int(0.8*X_train.shape[0])
    return X_train[:boundary], y_train[:boundary], X_train[boundary:], y_train[boundary:]

def main():
    batch_size = 32 # default
    filters = 250
    kernel_size = 5
    hidden_dims = 500
    epochs = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=['training', 'crowdsourced_training'],
                        help='Select the type of dataset to use for training')
    parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    parser.add_argument("--generate", '-g', action="store_true", default="False",
                        help="Use this argument to generate new data. By default it will use the loaded data")
    parser.add_argument("--epochs", '-e', default="10",
                        help="Use this argument to set the number of epochs. Default: 10")
    parser.add_argument("--filters", '-f', default="250",
                        help="Use this argument to set the number of filters. Default: 100")
    parser.add_argument("--kernel", '-k', default="4",
                        help="Use this argument to set the size of kernels. Default: 4")
    args = parser.parse_args()
    
    global dataset_type
    dataset_type = args.type
    
    global sem_eval_path
    sem_eval_path = args.path

    generate = args.generate
    filters = int(args.filters)
    kernel_size = int(args.kernel)
    epochs = int(args.epochs)

    X_train, y_train, X_val, y_val = load_pretrained_data(generate_new_data=generate)
    X_test, y_test = load_test_data()

    # max_features = 5000
    # maxlen = 400
    # embedding_dims = 300


    model = Sequential()

    model.add(Conv1D(filters,
                    kernel_size,
                    activation='relu',
                    input_shape=X_train[0].shape))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(hidden_dims, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    model.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(X_val, y_val))

    scores = model.evaluate(X_test, y_test, batch_size=batch_size)
    for idx, metric in enumerate(model.metrics_names):
        print('{}: {}'.format(metric, scores[idx]))

if __name__ == "__main__":
    main()