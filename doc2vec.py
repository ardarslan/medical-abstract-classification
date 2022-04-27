import os
import time
import pickle
import pprint
from functools import partial
from uuid import uuid4

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC

import gensim

import tensorflow as tf
from tensorflow_addons.metrics import F1Score

from utils import set_seeds, load_data, get_argument_parser
from utils import log as full_log
from utils import WeightedCategoricalCrossEntropy


def extend_argument_parser(parser):
    # Custom configurations for task 1.
    parser.add_argument(
        "--classifier",
        type=str,
        default="svm",
        choices=["svm", "mlp"],
        help="Classifier to use after with tf-idf features",
    )
    parser.add_argument(
        "--mlp_epochs",
        type=int,
        default=20,
        help="Number of epochs to train MLP.",
    )
    parser.add_argument(
        "--doc2vec_epochs",
        type=int,
        default=20,
        help="Number of epochs to train doc2vec.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Minimum number of occurences for a word to be considered in embedding",
    )
    parser.add_argument(
        "--dm",
        type=int,
        default=1,
        choices=[0, 1],
        help="dm=0: distributed bag of words, dm=1: distributed memory",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="The maximum distance between the current and predicted word within a sentence.",
    )
    return parser


def get_log_path(cfg):
    path = os.path.join(
        cfg["checkpoint_dir"],
        "doc2vec",
        cfg["experiment_time"]
    )
    os.makedirs(path, exist_ok=True)
    return path


def get_log_filename(cfg, unique_id):
    path = get_log_path(cfg)

    filename_parts = []
    filename_parts.append(f"epochs-{cfg['epochs']}")
    filename_parts.append(f"vector-size-{cfg['vector_size']}")
    filename_parts.append(f"min-count-{cfg['min_count']}")
    filename_parts.append(unique_id)

    filename = "-".join(filename_parts)
    return os.path.join(path, filename + ".log")


def save_cfg(cfg):
    log_path = get_log_path(cfg)
    config_path = os.path.join(log_path, "cfg.txt")
    with open(config_path, "w") as file_handler:
        file_handler.write(pprint.pformat(cfg, indent=4))


def prepare_corpus(row, tokens_only=False):
    token = row["cleaned_text"].split()
    if tokens_only:
        return token
    else:
        # For training data, add tags
        return gensim.models.doc2vec.TaggedDocument(token, [row.name])


def train_doc2vec(cfg, log, model_path, train_df):
    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=cfg["vector_size"],
        min_count=cfg["min_count"],
        epochs=cfg["doc2vec_epochs"],
        seed=cfg["seed"],
        window=cfg['window_size'],
        dm=cfg['dm']
    )

    log(msg="Building vocabulary for Doc2Vec...")
    model.build_vocab(train_df["prepared_text"])

    log(msg="Training Doc2Vec...")
    model.train(
        train_df["prepared_text"],
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    log(msg="Saving Doc2Vec...")
    model.save(model_path)


def load_doc2vec(log, model_path):
    log(msg="Loading Doc2Vec...")
    return gensim.models.doc2vec.Doc2Vec.load(fname=model_path)


def embed_split(model, df, train_spit=False):
    if train_spit:
        df_mapped = df["prepared_text"].apply(
            lambda v: model.infer_vector(v[0])
        )
    else:
        df_mapped = df["prepared_text"].apply(lambda v: model.infer_vector(v))
    return np.stack(df_mapped.tolist())


def save_dataset(cfg, X, y, split):
    log_path = get_log_path(cfg)
    dataset_path = os.path.join(log_path, f"{split}_dataset.pickle")
    with open(dataset_path, "wb") as file_handler:
        pickle.dump(obj={"X": X, "y": y}, file=file_handler, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_data(cfg, log, model_id):
    train_df = load_data(cfg=cfg, split_type="train", cached=cfg["cached"])
    dev_df = load_data(cfg=cfg, split_type="dev", cached=cfg["cached"])
    test_df = load_data(cfg=cfg, split_type="test", cached=cfg["cached"])
    log(msg="Data loaded.")

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_df["label"].values), y=train_df["label"].values)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(train_df["label"])
    y_dev = lb.transform(dev_df["label"])
    y_test = lb.transform(test_df["label"])

    log(msg="Preparing train set for Doc2Vec...")
    train_df["prepared_text"] = train_df.apply(prepare_corpus, axis=1)
    log(msg="Preparing dev set for Doc2Vec...")
    dev_df["prepared_text"] = dev_df.apply(
        partial(prepare_corpus, tokens_only=True), axis=1
    )
    log(msg="Preparing test set for Doc2Vec...")
    test_df["prepared_text"] = test_df.apply(
        partial(prepare_corpus, tokens_only=True), axis=1
    )

    log_path = get_log_path(cfg)
    model_path = os.path.join(log_path, f"doc2vec-{model_id}.model")

    train_doc2vec(
        cfg=cfg, log=log, model_path=model_path, train_df=train_df
    )

    doc2vec = load_doc2vec(log=log, model_path=model_path)

    log(msg="Embedding train set...")
    X_train = embed_split(model=doc2vec, df=train_df, train_spit=True)
    log(msg="Embedding dev set...")
    X_dev = embed_split(model=doc2vec, df=dev_df)
    log(msg="Embedding test set...")
    X_test = embed_split(model=doc2vec, df=test_df)

    log(msg="Data preparation done.")

    log(msg="Saving train dataset...")
    save_dataset(cfg, X_train, y_train, split="train")
    log(msg="Saving dev dataset...")
    save_dataset(cfg, X_dev, y_dev, split="dev")
    log(msg="Saving test dataset...")
    save_dataset(cfg, X_test, y_test, split="test")

    log(msg="Saved datasets to disk.")
    return X_train, X_dev, X_test, y_train, y_dev, y_test, lb, class_weights


def train_svm(log, model_path, X_train, y_train):
    model = SVC(class_weight="balanced")
    y_train = y_train.argmax(axis=1)

    log(msg="Training SVM classifier...")
    model.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(obj=model, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def test_svm(log, model_path, X_test):
    log(msg="Loading best SVM classifier for evaluation...")
    with open(model_path, "rb") as f:
        model = pickle.load(file=f)

    return model.predict(X_test)


def build_mlp(cfg, input_shape, class_weights):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(units=256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(units=5, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])

    loss = WeightedCategoricalCrossEntropy(weights=class_weights)

    f1_score = F1Score(
        num_classes=5, average="weighted", name="weighted_f1_score"
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=[f1_score])

    return model


def train_mlp(cfg, log, model, model_path, X_train, y_train, X_valid, y_valid):
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
    )
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=5, verbose=2
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, verbose=1
    )

    log(msg="Training MLP...")
    model.fit(
        x=X_train,
        y=y_train,
        epochs=cfg["mlp_epochs"],
        verbose=1,
        shuffle=True,
        callbacks=[early_stopper, lr_scheduler, checkpointer],
        validation_data=(X_valid, y_valid),
    )


def test_mlp(log, model, model_path, X_test):
    log(msg="Loading best MLP for evaluation...")
    model.load_weights(model_path)
    y_test_pred = model.predict(x=X_test)
    return y_test_pred


def evaluate_predictions(log, y_true, y_pred):
    score = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    log(msg=f"Weighted F1 score: {score}")


def main():
    parser = get_argument_parser()
    parser = extend_argument_parser(parser)
    cfg = parser.parse_args().__dict__
    cfg["experiment_time"] = str(int(time.time()))

    save_cfg(cfg)

    set_seeds(cfg)

    unique_id = uuid4().hex
    log_filename = get_log_filename(cfg, unique_id=unique_id)
    partial_log = partial(full_log, filename=log_filename)
    partial_log(msg=pprint.pformat(cfg, indent=4))

    model_id = unique_id
    log_path = get_log_path(cfg)

    (X_train, X_dev, X_test, y_train, y_dev, y_test, _, class_weights) = prepare_data(
        cfg=cfg, log=partial_log, model_id=model_id
    )

    if cfg["classifier"] == "svm":
        model_path = os.path.join(log_path, f"best-svm-{model_id}.pickle")

        train_svm(
            log=partial_log,
            model_path=model_path,
            X_train=X_train,
            y_train=y_train,
        )

        y_train_pred = test_svm(
            log=partial_log, model_path=model_path, X_test=X_train
        )
        y_dev_pred = test_svm(
            log=partial_log, model_path=model_path, X_test=X_dev
        )
        y_test_pred = test_svm(
            log=partial_log, model_path=model_path, X_test=X_test
        )
    elif cfg["classifier"] == "mlp":
        model = build_mlp(cfg, input_shape=(X_train.shape[1],), class_weights=class_weights)
        model.summary(print_fn=lambda s: partial_log(msg=s))
        model_path = os.path.join(log_path, f"best-model-{model_id}")

        train_mlp(
            cfg=cfg,
            log=partial_log,
            model=model,
            model_path=model_path,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_dev,
            y_valid=y_dev,
        )

        y_train_pred = test_mlp(
            log=partial_log, model=model, model_path=model_path, X_test=X_train
        )
        y_dev_pred = test_mlp(
            log=partial_log, model=model, model_path=model_path, X_test=X_dev
        )
        y_test_pred = test_mlp(
            log=partial_log, model=model, model_path=model_path, X_test=X_test
        )

        y_train_pred = y_train_pred.argmax(axis=1)
        y_dev_pred = y_dev_pred.argmax(axis=1)
        y_test_pred = y_test_pred.argmax(axis=1)
    else:
        raise ValueError(
            f"{cfg['classifier']} is not a valid classifier option,"
            + "please choose either 'svm' or 'mlp'"
        )

    y_train = y_train.argmax(axis=1)
    y_dev = y_dev.argmax(axis=1)
    y_test = y_test.argmax(axis=1)

    partial_log(msg="Evaluating train split...")
    evaluate_predictions(log=partial_log, y_true=y_train, y_pred=y_train_pred)

    partial_log(msg="Evaluating dev split...")
    evaluate_predictions(log=partial_log, y_true=y_dev, y_pred=y_dev_pred)

    partial_log(msg="Evaluating test split...")
    evaluate_predictions(log=partial_log, y_true=y_test, y_pred=y_test_pred)


if __name__ == "__main__":
    main()
