"""Shared feature list for Orange → sklearn → Gradio."""

FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_impact",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

TARGET = "koi_disposition"
POSITIVE = "CONFIRMED"
NEGATIVE = "FALSE POSITIVE"
