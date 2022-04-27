import os
import random
import argparse
import ray
import spacy
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K


def set_seeds(cfg):
    seed = cfg["seed"]
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# loads data from disk.
# if preprocessed file not available, preprocesses it first.
def load_data(cfg, split_type, cached=True):
    if cached:
        file_path = os.path.join(
            cfg["data_dir"], split_type + "_cleaned.pickle"
        )
        if os.path.isfile(file_path):
            df = pd.read_pickle(file_path)
        else:
            df = preprocess_raw(cfg=cfg, split_type=split_type)
    else:
        df = preprocess_raw(cfg=cfg, split_type=split_type)

    print(f"{split_type} data loaded.")
    return df


def read_raw(cfg, split_type):
    read_dir = os.path.join(cfg["data_dir"], split_type + ".txt")

    with open(read_dir) as f:
        lines = f.readlines()

    label_set = [
        "OBJECTIVE",
        "BACKGROUND",
        "METHODS",
        "RESULTS",
        "CONCLUSIONS",
    ]
    label_text_tuples = []

    for line in lines:
        lbl, _, txt = line.partition("\t")

        if lbl in label_set:
            label_text_tuples.append((lbl, txt))

    df = pd.DataFrame(label_text_tuples, columns=["label", "text"])
    return df


# preprocessing step. depends on options arguments passed.
def preprocess_raw(cfg, split_type):
    df = read_raw(cfg, split_type)

    write_dir = os.path.join(cfg["data_dir"], split_type + "_cleaned.pickle")
    print(f"Preprocessing raw {split_type} data from scratch...")
    nlp = spacy.load("en_core_web_sm")

    @ray.remote
    def preprocess_f(row, cfg):
        processed_token_list = []

        def remove_f(cfg, t, type):
            if type == "stop":
                if cfg["remove_stop"]:
                    return not t.is_stop
                else:
                    return True
            else:  # punc
                if cfg["remove_punc"]:
                    return not t.is_punct
                else:
                    return True

        for t in nlp(row):
            if (
                remove_f(cfg, t, "stop")
                and remove_f(cfg, t, "punc")
                and not t.is_space
            ):
                if (
                    len(t) >= cfg["min_token_len"]
                    and len(t) <= cfg["max_token_len"]
                ):
                    if t.like_num and cfg["number2hashsign"]:
                        processed_token_list.append("#")
                    else:
                        if cfg["lemmatize"]:
                            processed_token_list.append(t.lemma_.lower())
                        else:
                            try:
                                processed_token_list.append(t.lower())
                            except TypeError:
                                processed_token_list.append(str(t).lower())

        return " ".join(processed_token_list)

    ray.init()
    df["cleaned_text"] = ray.get(
        [preprocess_f.remote(row, cfg) for row in df["text"]]
    )
    ray.shutdown()

    df = df[["label", "cleaned_text"]]
    df.to_pickle(write_dir, protocol=4)

    return df


def log(filename, msg):
    print(msg)
    with open(filename, "a") as f:
        f.write(msg + "\n")


"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
def WeightedCategoricalCrossEntropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = tf.cast(y_true, tf.float32) * K.log(tf.cast(y_pred, tf.float32) * tf.cast(weights, tf.float32))
        loss = -K.sum(loss, -1)
        return loss

    return loss


# gets arguments passed to CLI
def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="../data/PubMed_200k_RCT"
    )
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--cached", type=bool, default=True)
    parser.add_argument("--vector_size", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=15)
    parser.add_argument("--w2v_epochs", type=int, default=5)
    parser.add_argument("--retrain_task", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--finetune_bert", action="store_true")
    parser.add_argument("--num_bert_layers_to_fine_tune", type=int, default=1)
    parser.add_argument("--previous_checkpoint_abs_path", type=str, default=None)
    parser.add_argument(
        "--bert_base",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",  # distilbert-base-uncased  # bert-base-uncased
    )
    parser.add_argument(
        "--bert_base_output_mode",
        type=str,
        default="cls",
    )
    parser.add_argument(
        "--bert_base_sentence_max_length",
        type=int,
        default=256,
    )
    # below are preprocessing options
    parser.add_argument("--remove_stop", type=bool, default=True)
    parser.add_argument("--remove_punc", type=bool, default=True)
    parser.add_argument("--number2hashsign", action="store_true")
    parser.add_argument("--min_token_len", type=int, default=1)
    parser.add_argument("--max_token_len", type=int, default=15)
    parser.add_argument("--lemmatize", action="store_false")

    return parser
