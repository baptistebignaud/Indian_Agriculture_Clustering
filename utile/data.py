import os

from scipy import rand

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
import sys
import random

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split
from functools import reduce
from operator import __or__ as union, __and__ as interesction
import tensorflow as tf


# set random seeds
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def enc(dicts, x):
    try:
        return dicts[x]
    except:
        return x


class DataSet(pd.DataFrame):
    """
    The aim of this class is to provide a pandas dataframe class in which we could perform pre-processing
    """

    def __init__(
        self,
        data: list,
        columns: list = None,
        numeric_features: List[str] = None,
        textual_features: List[str] = None,
        ohe_features: List[str] = None,
        values_list_to_replace: List[str] = [
            "consider area of aubli-dharwad mahanagara palike as per notification",
            "consider area of hubli-dharwad mahanagara palike as per notification",
            "ngp",
        ],
    ) -> None:
        """
        Constructor of the class
        data : The data of our dataset
        numeric_features: All numeric features one wants inside the dataset
        textual_features: All texual features, which will be embedded, one wants inside the dataset
        ohe_features: All texual features, which will be one-hot encoded, one wants inside the dataset
        values_list_to_replace: Values to delete in column Area Sown
        returns -> None
        """
        super().__init__(data=data, columns=columns)
        self.replace(r"^\s*$", np.nan, regex=True, inplace=True)
        self.numeric_features = numeric_features
        self.textual_features = textual_features
        self.ohe_features = ohe_features
        self.values_list_to_replace = values_list_to_replace

    def set_years_to_keep(self, thresh: float = 0.1, drop=True) -> None:
        """
        Keep only crop yields year in which the sparsity rate is below thresh
        thresh : Threshold
        returns -> None
        """
        years_yield = np.arange(2000, 2019, 1)
        years_to_keep = []
        for year in years_yield:
            empty_rate = len(self[self[f"{year}_Yield"].isna()]) / len(self)
            if empty_rate <= thresh:
                years_to_keep.append(year)
        # Drop rows with NaN values in years one wants to keep
        if drop:
            self.dropna(
                subset=[f"{year}_Yield" for year in years_to_keep],
                inplace=True,
                how="any",
            )
        else:
            values = {}
            for col in [f"{year}_Yield" for year in years_to_keep]:
                values[col] = pd.to_numeric(self[col].dropna()).mean()
            print(values)
            self.fillna(value=values, inplace=True)

        self.reset_index(inplace=True, drop=True)
        # Recast to numeric
        for year in years_to_keep:
            self[f"{year}_Yield"] = pd.to_numeric(
                self[f"{year}_Yield"], downcast="float"
            )
        # Keep track of years to keep
        self.years_crop_yields = [f"{year}_Yield" for year in years_to_keep]

    def pre_process_numeric(self) -> None:
        """
        Clean numeric columns
        returns -> None
        """
        # Clean and recast to numeric, and fillna with mean
        for feature in self.numeric_features:
            if feature == "Area_Sown_Ha":
                self[feature].replace(self.values_list_to_replace, np.nan, inplace=True)
            self[feature] = pd.to_numeric(self[feature], downcast="float")
            self[feature].fillna(self[feature].mean(), inplace=True)
            self[feature] = (self[feature] - self[feature].mean()) / self[feature].std()

    def preprocess_textual(self, subsample=True, thresh_representativty=True) -> None:
        """
        Clean textual columns
        subsample: If we want the number of examples with unknown or others value
        thresh_representativty: Threshold in which we want to put the value to "other" to avoid training on too many different values
        returns -> None
        """
        indices_null_other = []
        indices_non_null = []

        # Encode values that occurs too few times to "other" and fill NaN values to "unknown"
        for feature in self.textual_features:
            self[feature].fillna("unknown", inplace=True)
            self[feature] = self[feature].str.lower()
            # Index of values that occur too few times
            if thresh_representativty:
                count = self[feature].value_counts()
                ind = count[count <= min(0, len(self) / 1000)].index
                other_dicts = {elem: "other" for elem in ind}
                self[feature] = self[feature].apply(lambda x: enc(other_dicts, x))
                df_other = self[self[feature] == "other"]
            df_unknown = self[self[feature] == "unknown"]
            if not subsample:
                continue
            # The aim of this part is to avoid keeping too many examples with either "unknown" or "other" values, so for each features,
            # we keep only the average number of occurences of values of the feature
            if not df_other.empty:
                # For each feature take first n values from indexes of uknown and other values so that the union of both is minimal
                indices_null_other.append(
                    df_other.iloc[
                        0 : min(
                            len(df_other),
                            int(self[feature].value_counts().values.mean()),
                        )
                    ].index
                )
                indices_null_other.append(
                    df_unknown.iloc[
                        0 : min(
                            len(df_unknown),
                            int(self[feature].value_counts().values.mean()),
                        )
                    ].index
                )
            # Take indexes of non other or unknown values
            if not df_unknown.empty:
                indices_non_null.append(
                    self[~self[feature].isin(["other", "unknown"])].index
                )
        if not subsample:
            return
        # Take a subset so that for each feature, we have a correct representation of unkwown and other
        try:
            indx_null_other = reduce(union, (index for index in indices_null_other))
        except:
            pass
        try:
            indx_non_null_other = reduce(
                interesction, (index for index in indices_non_null)
            )
        except:
            pass
        # if indices_null_other:
        #     indx_null_other = indices_null_other[0]
        #     for elem, _ in enumerate(indices_null_other[1:]):
        #         indx_null_other = indx_null_other.union(elem)

        # if indices_non_null:
        #     indx_non_null_other = indices_non_null[0]
        #     for j in range(1, len(indices_non_null)):
        #         indx_non_null_other = indx_non_null_other.intersection(
        #             indices_non_null[j]
        #         )
        try:
            indx_total = indx_null_other.union(indx_non_null_other)
            self.drop(self.index.difference(indx_total), axis=0, inplace=True)
            self.reset_index(inplace=True, drop=True)
        except:
            pass

    def subset_data(self) -> None:
        """
        Keep only the columns one wants keep
        returns -> None
        """
        filtered_list = filter(
            None.__ne__,
            [
                self.numeric_features,
                self.textual_features,
                self.ohe_features,
                self.years_crop_yields,
            ],
        )
        col = [item for sublist in filtered_list for item in sublist]
        self.drop(self.columns.difference(col), axis=1, inplace=True)

    def set_dict_shape(self) -> None:
        """
        Set the shape of each embedding dictionnary
        returns -> None
        """
        dict_shape = {}
        for cf in self.textual_features:
            dict_shape[cf] = len(self[cf].unique())
        self.dict_shape = dict_shape

    def encode_ohe(self) -> None:
        """
        Encode ohe features if needed
        returns -> None
        """
        if not self.ohe_features:
            self.L_OHE = None
            return
        L_OHE = []
        for feature in self.ohe_features:
            self[feature].fillna("None", inplace=True, axis=0)
            y = pd.get_dummies(self[feature], prefix=feature)
            L_OHE.append(y.columns)
            self.drop(feature, axis=1, inplace=True)
            for col in y.columns:
                self[col] = y[col]
        self.L_OHE = L_OHE

    def encode_textual(self) -> None:
        """
        Encode textual features if needed
        returns -> None
        """
        if not self.textual_features:
            return
        vocab = {}

        # textual_feature =
        for feature in self.textual_features:
            # print(feature)
            vocab[feature] = {
                k: v for v, k in enumerate(np.unique(self[feature].apply(str)))
            }
            # print(vocab[feature])
            self[feature] = self[feature].apply(lambda x: vocab[feature][x])
        self.vocab = vocab

    def df_train_test_split(self, random_state=seed, flatten=False):
        """
        flatten: If flatten is true, it means the dataset has been unpivotted and years is a column (crop yields won't be columns then)
        Data train/test split
        returns -> None
        """
        #
        if not flatten:
            label = self.years_crop_yields
            # self.L_OHE = None
        else:
            label = "label"
            self[label] = pd.to_numeric(self[label])
        if self.L_OHE:
            X_train, X_test, y_train, y_test = train_test_split(
                self[
                    self.numeric_features
                    + self.textual_features
                    + self.L_OHE[0].tolist()
                ],
                self[label],
                test_size=0.3,
                random_state=random_state,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self[self.numeric_features + self.textual_features],
                self[label],
                test_size=0.3,
                random_state=seed,
            )
        if self.textual_features:
            input_textual_dict_train = {}
            input_textual_dict_test = {}
            for feature in self.textual_features:
                if isinstance(X_train[feature].iloc[0], np.float32):
                    input_textual_dict_train[feature] = np.asarray(
                        X_train[feature]
                    ).astype(np.float32)
                    input_textual_dict_test[feature] = np.asarray(
                        X_test[feature]
                    ).astype(np.float32)
                else:
                    input_textual_dict_train[feature] = np.asarray(
                        X_train[feature]
                    ).astype(np.int64)
                    input_textual_dict_test[feature] = np.asarray(
                        X_test[feature]
                    ).astype(np.int64)

        if self.numeric_features:
            input_numeric_dict_train = {}
            input_numeric_dict_test = {}
            for feature in self.numeric_features:
                if isinstance(X_train[feature].iloc[0], np.float32):
                    input_numeric_dict_train[feature] = np.asarray(
                        X_train[feature]
                    ).astype(np.float32)
                    input_numeric_dict_test[feature] = np.asarray(
                        X_test[feature]
                    ).astype(np.float32)
                else:
                    input_numeric_dict_train[feature] = np.asarray(
                        X_train[feature]
                    ).astype(np.int64)
                    input_numeric_dict_test[feature] = np.asarray(
                        X_test[feature]
                    ).astype(np.int64)

        if self.ohe_features:
            input_ohe_dict_train = {}
            input_ohe_dict_test = {}
            for feature in self.L_OHE[0].tolist():
                input_ohe_dict_train[feature] = np.asarray(X_train[feature]).astype(
                    np.int64
                )
                input_ohe_dict_test[feature] = np.asarray(X_test[feature]).astype(
                    np.int64
                )
        output = {}
        if self.textual_features:
            output["textual"] = {
                "train": input_textual_dict_train,
                "test": input_textual_dict_test,
            }
        else:
            output["textual"] = {
                "train": None,
                "test": None,
            }
        if self.numeric_features:
            output["numeric"] = {
                "train": input_numeric_dict_train,
                "test": input_numeric_dict_test,
            }
        else:
            output["numeric"] = {
                "train": None,
                "test": None,
            }
        if self.ohe_features:
            output["ohe"] = {
                "train": input_ohe_dict_train,
                "test": input_ohe_dict_test,
            }
        else:
            output["ohe"] = {
                "train": None,
                "test": None,
            }

        output["label"] = {
            "train": y_train,
            "test": y_test,
        }
        return output

    def df_train_test_split_index(self, random_state=seed):
        """
        Train/test split only on index
        """
        self.index_train, self.index_test = train_test_split(
            self.index, random_state=random_state
        )


############# Functions to plot on data vizualisation notebook #############


def plot_text_feature(df: DataSet, feat: str, k: int = 20, logY: bool = False) -> None:
    """
    This function's aim is to plot the repartition of a textual feature of the data
    df : The dataframe which contains the data
    feat: The feature we want to plot
    k : The maximum number of values of the feature we want to see on the plot
    logY: Parameter which set if the scale must be in log
    returns -> None
    """
    # Set seaborn theme
    sns.set_theme()

    # Subplot
    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
    fig.set_figheight(15)
    fig.set_figwidth(25)

    # Keep the k values with more occurencies
    vc = df[feat].value_counts()
    keys = list(vc.keys())[: min(len(vc), k)]

    # Bar chart of the number of occurences wrt. each value (from the top k)
    ax[0].bar(keys, [vc[key] for key in keys], log=logY)
    ax[0].set_xticklabels(labels=keys, rotation=45, ha="right")
    ax[0].set_title(f"Repartition of values among {feat}")

    # Cumulative repartition of the feature
    cumu_sum_list = [list(vc.items())[i][1] for i in range(len(vc.keys()))]
    ax[1].plot(np.arange(len(vc)), np.cumsum(cumu_sum_list / np.sum(cumu_sum_list)))
    ax[1].set_title(f"Cumulative repartition of {feat} wrt. number of values")


def plot_continous_feature(
    df: DataSet, feat: str, k: int = 20, Log: bool = False
) -> None:
    """
    This function's aim is to plot the repartition of a numeric feature of the data
    df : The dataframe which contains the data
    feat: The feature we want to plot
    k : The maximum number of values of the feature we want to see on the plot
    logY: Parameter which set if the scale must be in log
    returns -> None
    """
    # Set seaborn theme
    sns.set_theme()

    # Subplot
    fig, ax = plt.subplots(1, 4)
    fig.set_figheight(15)
    fig.set_figwidth(25)

    # Plot histogramm of feature
    ax[0].set_xscale("log")
    ax[0].hist(df[feat], bins=100, log=Log)
    ax[0].set_title(f"Histogramm of {feat}")

    # Plot repartition function of feature
    sns.kdeplot(df[feat], bw_method=0.1, ax=ax[1])
    ax[1].set_xscale("log")
    ax[1].set_title(f"Repartition funtion of {feat}")

    # Plot cumulative repartition function of feature
    sns.kdeplot(df[feat], bw_method=0.1, cumulative=True, ax=ax[2])
    ax[2].set_xscale("log")
    ax[2].set_title(f"Cumulative repartition funtion of {feat}")

    # Boxplot of feature
    ax[3].boxplot(df[feat])
    ax[3].set_yscale("log")
    ax[3].set_title(f"Boxplot of {feat}")


def plot_link_texual_features(
    df: DataSet, feat: str, k: int = 20, logY: bool = False, norm: str = None
) -> None:
    """
    This function's aim is to plot correlation between crop yields average and a texual feature of the data
    df : The dataframe which contains the data
    feat: The feature we want to plot
    k : The maximum number of values of the feature we want to see on the plot
    logY: Parameter which set if the scale must be in log
    returns -> None
    """
    # Normalize the volume of crop yields wrt. norm feature (if needed)
    if norm:
        df_norm_group_by = df[[feat, norm]].groupby(feat).mean()
        df_plot = df[[feat, "Mean_Years"]]
        df_group_by = df_plot.groupby(feat).mean()
        df_group_by = df_group_by / df_norm_group_by
        df_group_by.sort_values("Mean_Years", ascending=False)

    # Group the data according to feat
    df_plot = df[[feat, "Mean_Years"]]
    df_group_by = (
        df_plot.groupby(feat).mean().sort_values("Mean_Years", ascending=False)
    )

    # Set seaborn theme
    sns.set_theme()

    # Subplot
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(15)
    fig.set_figwidth(25)

    # Keep the k values with higher crop yields
    keys = list(df_group_by.index)[: min(len(df_group_by), k)]

    # Plot bar chart of avg crop yields wrt. top k feat
    ax[0].bar(keys, [df_group_by.loc[key].item() for key in keys], log=logY)
    ax[0].set_ylabel("Average Crop Yields over years")
    ax[0].set_xticklabels(labels=keys, rotation=45, ha="right")
    ax[0].set_title(f"Link between crops yields average and {feat}")

    # Plot boxplot of avg crop yields wrt. feat
    ax[1].boxplot([elem[0] for elem in df_group_by.values])
    ax[1].set_ylabel("Average Crop Yields over years")
    ax[1].set_title(f"Boxplot of average crop yields for {feat}")


def plot_link_numeric_features(df: DataSet, feat: str) -> None:
    """
    This function's aim is to plot correlation between crop yields average and a numeric feature of the data
    df : The dataframe which contains the data
    feat: The feature we want to plot
    returns -> None
    """
    # Group the data according to feat
    df_plot = df[df[feat] >= 0][[feat, "Mean_Years"]]
    df_group_by = df_plot.groupby(feat).mean()

    # Prepare axis values for scatter plot
    x = list(df_group_by.index)
    y = [elem[0] for elem in df_group_by.values]

    # Scatter plot of crops yields avg and feature value
    plt.scatter(x, y)
    plt.legend(f"Scatter plot between crops yields average and {feat}")
    plt.xlabel(f"Value of {feat}")
    plt.ylabel(f"Average value of Crops Yield")
