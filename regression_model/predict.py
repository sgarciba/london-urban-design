import typing as t

import numpy as np
import pandas as pd

from regression_model import __version__ as _version
from regression_model.processing.data_manager import load_pipeline

pipeline_file_name = f"regression_model_output_v{_version}.pkl"
_lst_pipeline = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    results = {"predictions": None, "version": _version}

    predictions = _lst_pipeline.predict(
        X=input_data
    )
    results = {
        "predictions": predictions.tolist(),
        "version": _version
    }

    return results
