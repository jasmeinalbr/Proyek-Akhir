"""Transform module for Adult Income Dataset"""

import tensorflow as tf
import tensorflow_transform as tft

# Categorical features and approximate vocab size (+1 buffer for OOV)
CATEGORICAL_FEATURES = {
    "workclass": 9,        # Private, Self-emp, Govt, etc.
    "education": 16,       # 16 unique education levels
    "marital.status": 7,   # Married, Divorced, etc.
    "occupation": 15,      # Occupation types
    "relationship": 6,     # Husband, Wife, etc.
    "race": 5,             # White, Black, etc.
    "sex": 2,              # Male, Female
    "native.country": 42   # Countries
}

# Numerical features
NUMERICAL_FEATURES = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

# Label key
LABEL_KEY = "income"

def transformed_name(key: str) -> str:
    """Renaming transformed features"""
    return key + "_xf"


def convert_num_to_one_hot(label_tensor, num_labels: int):
    """Convert int to one-hot tensor"""
    one_hot_tensor = tf.one_hot(label_tensor, num_labels, dtype=tf.float32)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features"""
    outputs = {}

    # One-hot encode categorical (replace nulls / "?" with "unknown")
    for key, dim in CATEGORICAL_FEATURES.items():
        input_tensor = inputs[key]

        # Hapus tanda kutip di awal/akhir
        cleaned = tf.strings.regex_replace(input_tensor, '"', '')

        # Ganti missing / "?" jadi "unknown"
        cleaned = tf.where(
            tf.logical_or(tf.equal(cleaned, "?"), tf.equal(cleaned, "")),
            tf.constant("unknown", dtype=tf.string),
            cleaned,
        )

        int_value = tft.compute_and_apply_vocabulary(
            cleaned,
            top_k=dim,              # max vocab size
            num_oov_buckets=1       # 1 bucket for OOV
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    # Scale numeric (replace NaN with 0.0)
    for feature in NUMERICAL_FEATURES:
        input_tensor = tf.cast(inputs[feature], tf.float32)  # pastikan float
        cleaned = tf.where(tf.math.is_nan(input_tensor), 0.0, input_tensor)
        outputs[transformed_name(feature)] = tft.scale_to_0_1(cleaned)

    # Label (income: <=50K -> 0, >50K -> 1)
    label_str = inputs[LABEL_KEY]  # income: string ">50K" / "<=50K"
    label = tf.where(tf.equal(label_str, ">50K"), 1, 0)  # ubah jadi 0/1
    label = tf.cast(tf.reshape(label, [-1, 1]), tf.float32)  # reshape + cast ke float
    outputs[transformed_name(LABEL_KEY)] = label

    return outputs