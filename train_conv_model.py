from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Activation, Embedding, Flatten, GlobalMaxPooling1D
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

def build_pv_models(df, sem_eval_path):
    """ Function which builds the paragraph vector models (title and content models) based on the 
    data in the data frame df. Also pickles the paragraph vector instance.
    ARGUMENTS: df, Pandas Dataframe which has already been shuffled.
    RETURNS: ParagraphVectorModel object pv, which has 2 Doc2Vec models, 2 TaggedDocuments, and
             a dataframe as its members
    DETAILS: The 2 Doc2Vec models can be accessed by pv.model_content_dbow and pv.model_title_dbow.
             These are committed to disk when build_doc2vec_content_model and build_doc2vec_title_model are
              called (as Embeddings/doc2vec_dbow_model_content_idtags and Embeddings/doc2vec_dbow_model_title_idtags resp.)"""
    pv = ParagraphVectorModel(df, sem_eval_dir_path=sem_eval_path)
    # Remove df to save memory
    del df
    # Get docs of form [Word list, tag]: title and content tagged separately
    pv.get_tagged_docs()
    # Each of the models created in the foll. statemw
    pv.build_doc2vec_content_model()
    pv.build_doc2vec_title_model()
    pv_location = os.path.join(sem_eval_path, 'models', 'pv_object.pickle')
    with open(pv_location, 'wb') as pfile:
        pickle.dump(pv, pfile, pickle.HIGHEST_PROTOCOL)
    return pv

def get_data():
    pv_location = os.path.join(sem_eval_path, 'models', 'pv_object.pickle')
    try:
        # If a paragraph vector has already been pickled, load it in.
        with open(pv_location, 'rb') as pfile:
            print("Loading paragraph vector instance from pickle...")
            pv = pickle.load(pfile)
    except FileNotFoundError:
        filename = os.path.join(sem_eval_path, 'data', 'IntegratedFiles', 'buzzfeed_training_withid.tsv')
        df_location = os.path.join(sem_eval_path, 'data', 'Pickles', 'training_df.pickle')
        # Doc2Vec training required
        df = clean_shuffle.read_prepare_df(filename, file_path=df_location)
        import sys
        sys.exit()
        print("Training paragraph vectors...")
        pv = build_pv_models(df, sem_eval_path)

    X_train, y_train_df = get_vector_label_mapping(pv, 'concat')

    # Needed to reshape from (n, 300) to (n, 300, 1) to serve as an input to the Conv layer
    X_train = numpy.atleast_3d(X_train)

    return X_train, y_train_df.hyperpartisan

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",'-p', default="/home/agon/Files/SemEval",
                        help="Use this argument to change the SemEval directory path (the default path is: '/home/ashwath/Files/SemEval')")
    parser.add_argument("--epochs", '-e', default="10",
                        help="Use this argument to set the number of epochs. Default: 10")
    parser.add_argument("--filters", '-f', default="64",
                        help="Use this argument to set the number of filters. Default: 64")
    parser.add_argument("--kernel", '-k', default="4",
                        help="Use this argument to set the size of kernels. Default: 4")
    args = parser.parse_args()
    
    global sem_eval_path
    sem_eval_path = args.path

    # Hyperparameters
    filters = int(args.filters)
    kernel_size = int(args.kernel)
    epochs = int(args.epochs)
    hidden_dims = 250
    batch_size = 32 # default

    # Get data
    X_train, y_train = get_data()

    # Model definition
    model = Sequential()

    model.add(Conv1D(filters,
                    kernel_size,
                    activation='relu',
                    input_shape=X_train[0].shape))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=4))

    model.add(Conv1D(filters,
                    kernel_size,
                    activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=4))

    model.add(GlobalMaxPooling1D())
    # model.add(Flatten())

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
                verbose=2)

    conv_model_location = os.path.join(sem_eval_path, 'models', 'conv_embeddings.h5')
    model.save(conv_model_location)

if __name__ == "__main__":
    main()