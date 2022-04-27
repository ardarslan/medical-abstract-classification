import os
import re
import pickle
import pprint
from functools import partial
from uuid import uuid4

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from lightgbm import LGBMClassifier

import tensorflow as tf
from tensorflow_addons.metrics import F1Score

from utils import set_seeds, load_data, get_argument_parser
from utils import log as full_log


def extend_argument_parser(parser):
    # Custom configurations for task 1.
    parser.add_argument(
        "--classifier",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "mlp"],
        help="Classifier to use after with tf-idf features",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=-1,
        help="ID of the model that should be loaded in case --retrain_task is not set.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top k features to select for model, -1 to select all",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of training data to be used for training",
    )
    parser.add_argument(
        "--tfidf_min_df",
        type=float,
        default=3.0,
        help="min_df parameter for TfidfVectorizer, same possible values",
    )
    parser.add_argument(
        "--no_tfidf_binary",
        action="store_true",
        help="'binary' option for TfidfVectorizer",
    )
    parser.add_argument(
        "--norm_paper_ids",
        action="store_true",
        help="normalize paper IDs into unified token",
    )
    parser.add_argument(
        "--norm_units",
        action="store_true",
        help="normalize units into unified token",
    )
    parser.add_argument(
        "--norm_tech_abbr",
        action="store_true",
        help="normalize technical abbreviations into unified token",
    )
    return parser


def get_log_path(cfg):
    path = os.path.join(
        cfg["checkpoint_dir"],
        "task1",
        cfg["classifier"],
    )
    os.makedirs(path, exist_ok=True)
    return path


def get_log_filename(cfg, unique_id):
    path = get_log_path(cfg)

    filename_parts = []
    filename_parts.append(f"lemma-{'y' if cfg['lemmatize'] else 'n'}")
    filename_parts.append(f"num-{'y' if cfg['number2hashsign'] else 'n'}")
    filename_parts.append(f"paperids-{'y' if cfg['norm_paper_ids'] else 'n'}")
    filename_parts.append(f"units-{'y' if cfg['norm_units'] else 'n'}")
    filename_parts.append(f"techabbr-{'y' if cfg['norm_tech_abbr'] else 'n'}")
    filename_parts.append(unique_id)

    filename = "-".join(filename_parts)
    return os.path.join(path, filename + ".log")


def build_normalizer(cfg):
    matchers = []

    # Article ID matchers.
    matcher_nct = re.compile(r"nct[0-9]+")
    matcher_isrctn = re.compile(r"isrctn[0-9]+")
    matcher_ntr = re.compile(r"ntr[0-9]+")
    if cfg["norm_paper_ids"]:
        matchers.append((matcher_nct, "PAPER-ID"))
        matchers.append((matcher_isrctn, "PAPER-ID"))
        matchers.append((matcher_ntr, "PAPER-ID"))
    # Unit matchers.
    matcher_units = re.compile(r"[0-9]+(mg|ml|mw|ms|mmhg|min|months|years)")
    if cfg["norm_units"]:
        matchers.append((matcher_units, "UNITS-ID"))
    # Medical abbreviation matchers.
    matcher_tech_abbr = re.compile(r"[0-9]*[a-z]+[0-9]+[a-z]*[0-9]*")
    if cfg["norm_tech_abbr"]:
        matchers.append((matcher_tech_abbr, "TECH-ABBR-ID"))
    # Number matchers.
    matcher_num = re.compile(r"\b[0-9]+\b")
    if cfg["number2hashsign"]:
        matchers.append((matcher_num, "ZAHL-ID"))

    def normalizer(s: str) -> str:
        for matcher, replacement in matchers:
            s = matcher.sub(replacement, s)
        return s

    return normalizer


def prepare_data(cfg, log):
    train_df = load_data(cfg=cfg, split_type="train", cached=cfg["cached"])
    dev_df = load_data(cfg=cfg, split_type="dev", cached=cfg["cached"])
    test_df = load_data(cfg=cfg, split_type="test", cached=cfg["cached"])
    log(msg="Data loaded.")

    lb = LabelBinarizer()
    y_train = lb.fit_transform(train_df["label"])
    y_dev = lb.transform(dev_df["label"])
    y_test = lb.transform(test_df["label"])

    # Bag of Word.
    if cfg["tfidf_min_df"] > 1.0:
        min_df = int(cfg["tfidf_min_df"])
    else:
        min_df = cfg["tfidf_min_df"]
    bow_model = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        binary=not cfg["no_tfidf_binary"],
    )

    normalizer = build_normalizer(cfg)

    X_train = bow_model.fit_transform(
        train_df["cleaned_text"].apply(normalizer)
    )
    X_dev = bow_model.transform(dev_df["cleaned_text"].apply(normalizer))
    X_test = bow_model.transform(test_df["cleaned_text"].apply(normalizer))
    log(msg="Tf-Idf features extracted.")

    log(msg=f"No. of extracted Tf-Idf features: {X_train.shape[1]}")

    if cfg["top_k"] != -1 and cfg["top_k"] < X_train.shape[1]:
        k_best = SelectKBest(chi2, k=cfg["top_k"])
        X_train = k_best.fit_transform(X_train, y_train)
        X_dev = k_best.transform(X_dev)
        X_test = k_best.transform(X_test)
        log(msg=f"Top {cfg['top_k']} features selected.")

    if cfg["train_fraction"] < 1.0:
        train_labels = y_train.argmax(axis=1)
        _, X_train, _, y_train = train_test_split(
            X_train,
            y_train,
            test_size=cfg["train_fraction"],
            stratify=train_labels,
            random_state=cfg["seed"],
        )
        log(msg=f"Split off {cfg['train_fraction']} fraction from tain data.")

    log(msg="Data preparation done.")
    return X_train, X_dev, X_test, y_train, y_dev, y_test, lb


def train_lgbm(log, model_path, X_train, y_train):
    model = LGBMClassifier()
    y_train = y_train.argmax(axis=1)

    log(msg="Training LGBM classifier...")
    model.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(obj=model, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def test_lgbm(log, model_path, X_test):
    log(msg="Loading best LGBM classifier for evaluation...")
    with open(model_path, "rb") as f:
        model = pickle.load(file=f)

    feature_importance = model.booster_.feature_importance()
    n_selected_features = feature_importance[feature_importance != 0].shape[0]
    log(msg=f"No. of selected features: {n_selected_features}")

    return model.predict(X_test)


def build_mlp(cfg, input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(units=256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(units=5, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])

    loss = tf.keras.losses.CategoricalCrossentropy()

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
        x=X_train.todense(),
        y=y_train,
        epochs=cfg["epochs"],
        verbose=1,
        shuffle=True,
        callbacks=[early_stopper, lr_scheduler, checkpointer],
        validation_data=(X_valid.todense(), y_valid),
    )


def test_mlp(log, model, model_path, X_test):
    log(msg="Loading best MLP for evaluation...")
    model.load_weights(model_path)
    y_test_pred = model.predict(x=X_test.todense())
    return y_test_pred


def evaluate_predictions(log, y_true, y_pred):
    score = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    log(msg=f"Weighted F1 score: {score}")


def main():
    parser = get_argument_parser()
    parser = extend_argument_parser(parser)
    cfg = parser.parse_args().__dict__

    set_seeds(cfg)

    unique_id = uuid4().hex
    log_filename = get_log_filename(cfg, unique_id=unique_id)
    partial_log = partial(full_log, filename=log_filename)
    partial_log(msg=pprint.pformat(cfg, indent=4))

    (X_train, X_dev, X_test, y_train, y_dev, y_test, _) = prepare_data(
        cfg, partial_log
    )

    model_id = unique_id if cfg["id"] == -1 else cfg["id"]
    log_path = get_log_path(cfg)

    if cfg["classifier"] == "lightgbm":
        model_path = os.path.join(log_path, f"best-lgbm-{model_id}.pickle")

        if cfg["retrain_task"]:
            train_lgbm(
                log=partial_log,
                model_path=model_path,
                X_train=X_train,
                y_train=y_train,
            )

        y_train_pred = test_lgbm(
            log=partial_log, model_path=model_path, X_test=X_train
        )
        y_dev_pred = test_lgbm(
            log=partial_log, model_path=model_path, X_test=X_dev
        )
        y_test_pred = test_lgbm(
            log=partial_log, model_path=model_path, X_test=X_test
        )
    elif cfg["classifier"] == "mlp":
        model = build_mlp(cfg, input_shape=(X_train.shape[1],))
        model.summary(print_fn=lambda s: partial_log(msg=s))
        model_path = os.path.join(log_path, f"best-model-{model_id}")

        if cfg["retrain_task"]:
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
            + "please choose either 'lightgbm' or 'mlp'"
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
