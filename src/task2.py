import os

import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow_addons.metrics import F1Score
from sklearn.preprocessing import LabelBinarizer

from utils import set_seeds, load_data, get_argument_parser


# gets the dense word vector for a particular word from trained word2vec model.
def get_vector(cfg, word, word_vectors):
    isPresent = False
    try:
        word_vector = word_vectors.get_vector(word)
        isPresent = True
    except KeyError:
        word_vector = np.zeros((cfg['vector_size'],))

    return isPresent, word_vector


# gets the sentence embedding which is 
# the average of each word vector in the sentence.
def get_sentence_embedding(cfg, row, word_vectors):
    sent_vector = np.zeros((cfg['vector_size'],))
    present_word_count = 0

    for temp_w in row.split():
        isPresent, word_vector = get_vector(cfg, temp_w, word_vectors)
        if isPresent:  # only sum non-zero words
            present_word_count += 1
            sent_vector += word_vector
    
    if present_word_count != 0:
        sent_vector = sent_vector/present_word_count

    return sent_vector


# trains word2vec model using gensim library
def train_word2vec(cfg):
    train_df = load_data(cfg=cfg, split_type='train', cached=cfg['cached'])

    w2v_model = Word2Vec(sentences=[sent.split() for sent
                         in train_df['cleaned_text'].values],
                         min_count=5, workers=cfg['num_workers'],
                         sg=1, vector_size=cfg['vector_size'],
                         epochs=cfg['w2v_epochs'])

    print('Word vectors trained.')
    return w2v_model.wv  # return keyedVectors


# loads data from disk and prepares it task-specifically.
def prepare_data(cfg, word_vectors):

    train_df = load_data(cfg=cfg, split_type='train', cached=cfg['cached'])
    dev_df = load_data(cfg=cfg, split_type='dev', cached=cfg['cached'])
    test_df = load_data(cfg=cfg, split_type='test', cached=cfg['cached'])

    lb = LabelBinarizer()
    y_train = lb.fit_transform(train_df['label'])
    y_dev = lb.transform(dev_df['label'])
    y_test = lb.transform(test_df['label'])

    X_train = np.stack(train_df['cleaned_text'].apply
                       (lambda row: get_sentence_embedding(cfg, row,
                                                           word_vectors)))
    X_dev = np.stack(dev_df['cleaned_text'].apply
                     (lambda row: get_sentence_embedding(cfg, row,
                                                         word_vectors)))
    X_test = np.stack(test_df['cleaned_text'].apply
                      (lambda row: get_sentence_embedding(cfg, row, 
                                                          word_vectors)))

    return X_train, X_dev, X_test, y_train, y_dev, y_test, lb


# trains the task 2 model with provided training arguments.
def train(cfg, X_train, X_dev, X_test, y_train, y_dev, y_test):

    inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))
    x = tf.keras.layers.Dense(units=256, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(units=5, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate'])

    loss = tf.keras.losses.CategoricalCrossentropy()

    f1_score = F1Score(num_classes=5, average='weighted', 
                       name='weighted_f1_score')

    file_path = os.path.join(cfg['checkpoint_dir'], 'task2', 'best_model')
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                                      monitor='val_loss', 
                                                      verbose=1,
                                                      save_best_only=True)
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0, 
                                                     patience=5, verbose=2)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                        factor=0.2, 
                                                        patience=3, verbose=1)

    model.compile(optimizer=optimizer, loss=loss, metrics=[f1_score])

    model.fit(x=X_train, y=y_train, epochs=cfg['epochs'], verbose=1,
              shuffle=True, callbacks=[early_stopper, lr_scheduler, 
              checkpointer], validation_data=(X_dev, y_dev))

    model.evaluate(x=X_test, y=y_test)

    return


if __name__ == '__main__':
    
    cfg = get_argument_parser().parse_args().__dict__
    
    set_seeds(cfg)

    word_vectors = train_word2vec(cfg)
    
    X_train, X_dev, X_test, y_train,\
        y_dev, y_test, label_binarizer = prepare_data(cfg, word_vectors)
    
    train(cfg, X_train, X_dev, X_test, y_train, y_dev, y_test)