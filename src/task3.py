import os
import time
import pprint

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
import numpy as np
import pandas as pd
import keras
from tensorflow_addons.metrics import F1Score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import TFAutoModel, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from utils import set_seeds, get_argument_parser, read_raw, WeightedCategoricalCrossEntropy
from tensorflow.python.training.tracking.data_structures import NoDependency


# defines the bert model
class BertBasedClassifier(tf.keras.Model):
    def __init__(self, cfg, fine_tune=False):
        super().__init__()
        self.cfg = NoDependency(cfg)

        self.bert_base = TFAutoModel.from_pretrained(cfg["bert_base"], from_pt=True)

        if fine_tune:
            if cfg["bert_base"] == "distilbert-base-uncased":
                self.bert_base.distilbert.embeddings.trainable = False

                for index, layer in enumerate(self.bert_base.distilbert.transformer.layer):
                    if index >= len(self.bert_base.distilbert.transformer.layer) - cfg["num_bert_layers_to_fine_tune"]:
                        layer.trainable = True
                    else:
                        layer.trainable = False
                
            elif cfg["bert_base"] in ["emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased"]:
                self.bert_base.bert.embeddings.trainable = False
                self.bert_base.bert.pooler.trainable = False

                for index, layer in enumerate(self.bert_base.bert.encoder.layer):
                    if index >= len(self.bert_base.bert.encoder.layer) - cfg["num_bert_layers_to_fine_tune"]:
                        layer.trainable = True
                    else:
                        layer.trainable = False
            else:
                raise Exception(f"Not a valid bert_base {cfg['bert_base']}")
        else:
            self.bert_base.layers[0].trainable = False

        self.dense1 = tf.keras.layers.Dense(
            units=768, activation="linear"
        )
        self.dense2 = tf.keras.layers.Dense(units=5, activation="softmax")

    def call(self, x):
        if self.cfg["bert_base"] == "distilbert-base-uncased":
            output = self.bert_base.distilbert(
                input_ids=x["input_ids"], attention_mask=x["attention_mask"]
            )
        elif self.cfg["bert_base"] in ["emilyalsentzer/Bio_ClinicalBERT", "bert-base-uncased"]:
            output = self.bert_base.bert(
                input_ids=x["input_ids"], attention_mask=x["attention_mask"]
            )
        else:
            raise Exception(f"Not a valid bert_base {bert_base}.")

        last_hidden_state = output[0]
        if self.cfg["bert_base_output_mode"] == "cls":
            output = last_hidden_state[:, 0, :]  # first token is cls token
        elif self.cfg["bert_base_output_mode"] == "mean":
            output = tf.math.reduce_mean(last_hidden_state[:, 1:, :], axis=1)
        else:
            raise Exception(
                f"Not a valid bert_base_output_mode {self.cfg['bert_base_output_mode']}."
            )

        output = self.dense1(output)
        output = self.dense2(output)

        return output


def prepare_data(cfg):
    def tokenize_function(row):
        return {**tokenizer(row["text"], padding="max_length", truncation=True, max_length=cfg["bert_base_sentence_max_length"]), "label": row["label"]}

    tokenizer = AutoTokenizer.from_pretrained(cfg["bert_base"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            return_tensors="tf",
                                            max_length=cfg["bert_base_sentence_max_length"])

    lb = LabelBinarizer()
    train_df = read_raw(cfg, split_type="train")
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_df["label"].values), y=train_df["label"].values)
    dev_df = read_raw(cfg, split_type="dev")
    test_df = read_raw(cfg, split_type="test")
    y_train = lb.fit_transform(train_df["label"].values).astype(np.float32)
    y_dev = lb.transform(dev_df["label"].values).astype(np.float32)
    y_test = lb.transform(test_df["label"].values).astype(np.float32)

    def prepare_data_helper(cfg, split, data_collator):
        if split == "train":
            df = train_df
            y = y_train
        elif split == "dev":
            df = dev_df
            y = y_dev
        elif split == "test":
            df = test_df
            y = y_test
        else:
            raise Exception(f"Not a valid split {split}.")
        
        y = [y[i, :] for i in range(y.shape[0])]
        
        ds = Dataset.from_dict({"text": df["text"].tolist(), "label": y}).map(tokenize_function, batched=True)

        ds = ds.to_tf_dataset(
                    columns=["attention_mask", "input_ids"],
                    label_cols=["label"],
                    shuffle=(split == "train"),
                    collate_fn=data_collator,
                    batch_size=cfg["batch_size"])
        return ds

    train_ds = prepare_data_helper(cfg, "train", data_collator)
    dev_ds = prepare_data_helper(cfg, "dev", data_collator)
    test_ds = prepare_data_helper(cfg, "test", data_collator)

    return train_ds, dev_ds, test_ds, class_weights, lb


# trains the task 3 model with provided training arguments.
def train(cfg):
    train_ds, dev_ds, test_ds, class_weights, lb = prepare_data(cfg)

    if cfg["previous_checkpoint_abs_path"] is not None:  # load a compiled model from a previous training
        cfg["experiment_time"] = cfg["previous_checkpoint_abs_path"].split("/")[1]  # use experiment_time of previous training
        save_dir = os.path.join(cfg["checkpoint_dir"], "task3", cfg["experiment_time"])
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "best_model")
        loss = WeightedCategoricalCrossEntropy(class_weights)
        model = keras.models.load_model(best_model_path, custom_objects={'loss': loss}) # this is already compiled and it has the last optimizer state
        model.cfg = cfg
    else:
        model = BertBasedClassifier(cfg, fine_tune=cfg["finetune_bert"])
        save_dir = os.path.join(cfg["checkpoint_dir"], "task3", cfg["experiment_time"])  # use experiment_time of current training
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "best_model")

        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
        loss = WeightedCategoricalCrossEntropy(class_weights)
        f1_score = F1Score(num_classes=5, average="weighted", name="weighted_f1_score")
        model.compile(optimizer=optimizer, loss=loss, metrics=[f1_score])
    
    
    pprint.pprint(cfg)
    
    test_predictions_path = os.path.join(save_dir, "test_predictions.csv")
    scores_path = os.path.join(save_dir, "scores.txt")
    config_path = os.path.join(save_dir, "config.txt")
    model_summary_path = os.path.join(save_dir, "model_summary.txt")

    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=5, verbose=2, restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, verbose=1
    )

    model.fit(
        train_ds,
        epochs=cfg["epochs"],
        verbose=1,
        shuffle=True,
        callbacks=[early_stopper, lr_scheduler],
        validation_data=dev_ds
    )

    model.summary()

    with open(model_summary_path, 'w') as writer:
        model.summary(print_fn=lambda x: writer.write(x + '\n'))

    model.save(best_model_path)  # this already saves the model with best weights because restore_best_weights is True in early_stopper.
                                 # this will overwrite the previous saved model if there was already one at the same path.
                                 # this also saves optimizer state.

    test_yhat = np.array(model.predict(test_ds))  # (num_samples, 5)
    test_y = np.concatenate([y for x, y in test_ds], axis=0)  # (num_samples, 5)
    classes = lb.classes_
    df = pd.DataFrame(np.hstack((test_y, test_yhat)), columns=[f"y_{classes[i]}" for i in range(5)] +
                                                              [f"yhat_{classes[i]}" for i in range(5)])
    df.to_csv(test_predictions_path, index=False)  # save test ground truths and predictions

    def evaluate(model, ds, split):
        print(f"Evaluating {split} dataset...")
        scores = model.evaluate(ds, verbose=2)
        loss_value = scores[0]
        weighted_f1 = scores[1]
        return loss_value, weighted_f1
    
    train_loss_value, train_weighted_f1 = evaluate(model, train_ds, "train")
    dev_loss_value, dev_weighted_f1 = evaluate(model, dev_ds, "dev")
    test_loss_value, test_weighted_f1 = evaluate(model, test_ds, "test")

    with open(scores_path, "w") as writer:
        writer.write(f"Train loss: {train_loss_value}, Train weighted f1: {train_weighted_f1}\n")
        writer.write(f"Dev loss: {dev_loss_value}, Dev weighted f1: {dev_weighted_f1}\n")
        writer.write(f"Test loss: {test_loss_value}, Test weighted f1: {test_weighted_f1}\n")
    
    with open(config_path, "w") as writer:
        writer.write(pprint.pformat(cfg, indent=4))


if __name__ == "__main__":
    cfg = get_argument_parser().parse_args().__dict__
    cfg["experiment_time"] = str(int(time.time()))

    set_seeds(cfg)

    train(cfg)
