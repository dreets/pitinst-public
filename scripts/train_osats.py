"""
Train OSATS models using kinematic summaries.
Runs various regression models to predict OSATS scores from kinematic summaries.

To execute this script, run:
    python scripts/train_osats.py --int_model <model_number>
Example:
    python scripts/train_osats.py --int_model 1
"""
import argparse
import joblib
import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import utils

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate OSATS Kinematic Summaries from Tracked Polygons.")

    parser.add_argument("--int_model", type=str, required=True, 
                        help="Model number for saved annotations found in results.")
   
    return parser


def main(argv: list[str] | None = None) -> None:
    logger.info("=" * 70)
    logger.info("LOADING PARAMETERS...")
    args = build_parser().parse_args(argv)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    dct_df_splits: dict[str, pd.DataFrame] = create_osats_df(int_model=args.int_model)

    logger.info("=" * 70)
    logger.info("RUNNING MODELS...")
    loo = LeaveOneOut()
    dct_base_models: dict = {
        "independent_elasticnet": make_pipeline(StandardScaler(), ElasticNetCV(cv=loo, max_iter=10000)),
        "linear_svr": make_pipeline(StandardScaler(), SVR(kernel="linear", C=0.1)),
        "random_forest": RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42),
    }

    logger.info("=" * 70)
    logger.info("EVALUATING MODELS...")
    dct_fitted_models: dict[str, dict[str, sklearn.base.RegressorMixin]] = {name: {} for name in dct_base_models.keys()}
    X_train = dct_df_splits["train"][utils.Osats.lst_kinematics_attributes].values
    for str_name, base_model in dct_base_models.items():
        for str_attribute in utils.Osats.lst_osats_attributes:
            y_train_attribute = dct_df_splits["train"][str_attribute].values
            final_attr_model = clone(base_model) # type: ignore
            final_attr_model.fit(X_train, y_train_attribute)
            dct_fitted_models[str_name][str_attribute] = final_attr_model

    evaluate_split(dct_df_splits=dct_df_splits, dct_models=dct_fitted_models, pth_osats_predictions=utils.Directories.pth_osats / f"{args.int_model}_predictions.csv")
    extract_explicit_importances(
        dct_fitted_models,
        pth_osats_importances=utils.Directories.pth_osats / f"{args.int_model}_importances.csv",
        X_train=X_train,
        df_train=dct_df_splits["train"],
    )
    save_models(
        dct_models=dct_fitted_models,
        pth_osats_dir=utils.Directories.pth_osats,
        int_model=int(args.int_model),
    )

    logger.info("=" * 70)
    logger.info("RUN COMPLETED!!!")


def save_models(dct_models: dict, pth_osats_dir: Path, int_model: int) -> None:
    """
    Saves each fitted model separately to disk using joblib.
    Args:
        dct_models: A dictionary of fitted models per model type and attribute.
        pth_osats_dir: Directory path where individual model files will be saved.
        int_model: Model identifier.
    Returns:
        None but saves each fitted model to its own joblib file.
    """
    pth_osats_dir.mkdir(parents=True, exist_ok=True)
    for str_name, dict_of_attr_models in dct_models.items():
        for str_attribute, model in dict_of_attr_models.items():
            pth_model = pth_osats_dir / f"{int_model}_{str_name}_{str_attribute}.joblib"
            joblib.dump(model, pth_model)
            logger.info(f"Saved model to {pth_model}.")


def evaluate_split(
        dct_df_splits: dict[str, pd.DataFrame],
        dct_models: dict,
        pth_osats_predictions: Path,
        lst_kinematics_attributes: list[str] = utils.Osats.lst_kinematics_attributes, 
        lst_osats_attributes: list[str] = utils.Osats.lst_osats_attributes, 
) -> None:
    """
    Evaluates the explicitly trained per-attribute models on given splits.
    Args:
        dct_df_splits: A dictionary mapping split names (e.g. "train", "val", "test") to their DataFrames.
        dct_models: A dictionary of fitted models per attribute.
        pth_osats_predictions: Path to save the OSATS predictions CSV.
        lst_kinematics_attributes: List of kinematics attribute names (default: utils.Osats.lst_kinematics_attributes).
        lst_osats_attributes: List of OSATS attribute names (default: utils.Osats.lst_osats_attributes).
    Returns:
        None but saves the OSATS predictions Mean Absolute Error (MAE) to the specified CSV path, with a STR_SPLIT column.
    """
    subscore_attributes = [attr for attr in lst_osats_attributes if attr != "INT_TOTAL"]
    lst_all_rows: list[dict] = []
    for str_split, df_split in dct_df_splits.items():
        X_split = df_split[lst_kinematics_attributes].values

        for str_model, dict_of_attr_models in dct_models.items():
            dict_row: dict = {"STR_SPLIT": str_split, "STR_MODEL": str_model}
            lst_pred_subscores: list[np.ndarray] = []
            for str_attribute in subscore_attributes:
                model = dict_of_attr_models[str_attribute]
                y_attr = df_split[str_attribute].values
                pred_attr = np.clip(model.predict(X_split), 1, 5) # clip subscores to [1, 5]
                dict_row[str_attribute] = mean_absolute_error(y_attr, pred_attr) # type: ignore
                lst_pred_subscores.append(pred_attr)

            pred_total_sum = np.sum(lst_pred_subscores, axis=0)
            y_total = df_split["INT_TOTAL"].values
            dict_row["INT_TOTAL_SUM"] = mean_absolute_error(y_total, pred_total_sum) # type: ignore

            model_total = dict_of_attr_models["INT_TOTAL"]
            pred_total_direct = np.clip(model_total.predict(X_split), len(subscore_attributes) * 1, len(subscore_attributes) * 5)
            dict_row["INT_TOTAL"] = mean_absolute_error(y_total, pred_total_direct) # type: ignore

            lst_all_rows.append(dict_row)

    df_osats_predictions: pd.DataFrame = pd.DataFrame(lst_all_rows)
    df_osats_predictions.to_csv(pth_osats_predictions, index=False)


def extract_explicit_importances(
        dct_models: dict,
        pth_osats_importances: Path,
        X_train: np.ndarray,
        df_train: pd.DataFrame,
        lst_kinematics_attributes: list[str] = utils.Osats.lst_kinematics_attributes,
        lst_osats_attributes: list[str] = utils.Osats.lst_osats_attributes,
    ) -> None:
    """
    Extracts signed feature weights/directions per attribute for each model type and saves them to a single CSV.
    Args:
        dct_models: A dictionary of fitted models per attribute.
        pth_osats_importances: Path to save the OSATS feature importances CSV.
        X_train: Training feature matrix (optional, used for signing tree feature importances).
        df_train: Training DataFrame containing OSATS attributes.
        lst_kinematics_attributes: List of kinematics attribute names (default: utils.Osats.lst_kinematics_attributes).
        lst_osats_attributes: List of OSATS attribute names (default: utils.Osats.lst_osats_attributes).
    Returns:
        None but saves the signed OSATS feature importances to the specified CSV path.
    """
    lst_all_rows: list[dict] = []
    for str_name, dict_of_attr_models in dct_models.items():
        for str_attribute in lst_osats_attributes:
            model = dict_of_attr_models[str_attribute]
            estimator = model[-1] if hasattr(model, "__getitem__") else model

            if hasattr(estimator, "coef_"):
                # Retain positive / negative sign for linear models
                weights = estimator.coef_.flatten()
            elif hasattr(estimator, "feature_importances_"):
                weights = estimator.feature_importances_.copy()
                # Sign tree importances by marginal correlation with target if data is provided
                if X_train is not None and df_train is not None:
                    y_attr = df_train[str_attribute].values
                    for idx in range(X_train.shape[1]):
                        std_x = np.std(X_train[:, idx])
                        std_y = np.std(y_attr) # type: ignore
                        if std_x > 0 and std_y > 0:
                            corr = np.corrcoef(X_train[:, idx], y_attr)[0, 1] # type: ignore
                            weights[idx] *= np.sign(corr) if not np.isnan(corr) else 1.0
            else:
                weights = np.zeros(len(lst_kinematics_attributes))

            dict_row: dict = {"STR_MODEL": str_name, "STR_ATTRIBUTE": str_attribute}
            dict_row.update(dict(zip(lst_kinematics_attributes, weights)))
            lst_all_rows.append(dict_row)

    df_imp: pd.DataFrame = pd.DataFrame(lst_all_rows)
    df_imp.to_csv(pth_osats_importances, index=False)


def create_osats_df(int_model: int) -> dict[str, pd.DataFrame]:
    """
    Creates the combined dataframe of osats and kinematics if it doesn't already exist.
    Args:
        int_model (int): The integer identifier for the model, used to locate the corresponding OSATS and kinematics CSV files.
    Returns:
        dict: A dictionary mapping split names ("train", "val", "test") to their corresponding DataFrames containing both OSATS and kinematics data for the specified model.
    """
    pth_osats: Path = utils.Directories.pth_osats / f"{int_model}.csv"
    if not utils.Osats.pth_osats_metadata.exists():
        raise FileNotFoundError(f"OSATS CSV file not found at {utils.Osats.pth_osats_metadata}")
    df_osats: pd.DataFrame = pd.read_csv(utils.Osats.pth_osats_metadata)

    pth_kinematics: Path = utils.Directories.pth_kinematics / str(int_model)
    if not pth_kinematics.exists():
        raise FileNotFoundError(f"Kinematics CSV file not found at {pth_kinematics}")
    
    lst_rows: list[dict] = []
    for str_video in utils.Videos.dct_split_videos["ss_val_test"]:
        pth_kinematics_video: Path = pth_kinematics / f"{str_video}.csv"
        if not pth_kinematics_video.exists():
            raise FileNotFoundError(f"Kinematics CSV file for video {str_video} not found at {pth_kinematics_video}")
        df_kinematics_video: pd.DataFrame = pd.read_csv(pth_kinematics_video)

        df_nonzero: pd.Series = df_kinematics_video["FLT_TIME"] > 0.002
        df_totals: pd.Series = df_kinematics_video[df_kinematics_video["INT_CLASS"] == 0].iloc[0]
        str_split = "train" if str_video in utils.Videos.lst_ss_train else "val" if str_video in utils.Videos.lst_val else "test"
        lst_rows.append({
            "STR_SPLIT": str_split,
            "STR_VIDEO": str_video,
            "FLT_TIME": df_totals["FLT_TIME"],
            "FLT_DISTANCE": df_totals["FLT_DISTANCE"],
            "FLT_SPEED": df_totals["FLT_SPEED"],
            "FLT_ACCELERATION": df_totals["FLT_ACCELERATION"],
        })
    df_kinematics_videos: pd.DataFrame = pd.DataFrame(lst_rows)
    df_merged: pd.DataFrame = pd.merge(df_kinematics_videos, df_osats, on="STR_VIDEO")
    df_merged.to_csv(pth_osats, index=False)

    df_train = df_merged[df_merged["STR_SPLIT"] == "train"]
    df_val = df_merged[df_merged["STR_SPLIT"] == "val"]
    df_test = df_merged[df_merged["STR_SPLIT"] == "test"]
    dct_df_splits = {"train": df_train, "val": df_val, "test": df_test}
    return dct_df_splits


if __name__ == "__main__":
    raise SystemExit(main())
