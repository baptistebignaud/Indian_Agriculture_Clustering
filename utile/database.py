from pathlib import Path
import pandas as pd
import os
import sqlalchemy
import numpy as np
from pandas.api.types import is_numeric_dtype

# Variable to rename crops yield
years = [f"20{i//10}{i%10}" for i in range(0, 19)]

# These are columns to rename so that it could be understood by mysql database
columns_to_rename = {
    "index": "id",
    "Sub-District": "Sub_District",
    "Area Sown (Ha)": "Area_Sown_Ha",
    "Area Insured (Ha)": "Area_Insured_Ha",
    "SI Per Ha (Inr/Ha)": "SI_Per_Ha_Inr_Ha",
    "Sum Insured (Inr)": "Sum_Insured_Inr",
    "Indemnity Level": "Indemnity_Level",
}

# Rename crop yields
for year in years:
    columns_to_rename[f"{year} Yield"] = f"{year}_Yield"


def config_parse() -> dict:
    """
    The aim of this function is to get the configuration to connect to the database thanks to environment variables
    returns -> dictio: The dictionnary with the configuration for the database
    """
    dictio = {}
    value = dict(os.environ).items()
    for val in value:
        dictio[val[0]] = val[1]

    return dictio


def open_dataset(path: str, year: str, name: str) -> pd.DataFrame:
    """
    For a given xlsx file, this function returns the dataframe from this file
    path : The parent folder containing the whole data
    year : The year of the file we want to open
    name : The name of the file we want to open
    returns -> df : The dataframe of the file
    """
    df = pd.read_excel(os.path.join(path, year, name) + ".xlsx")
    return df


def create_table(path: str) -> str:
    """
    This function's aim is to build the statement to create the data table
    path : The parent folder containing the whole data
    returns -> statement : The stamement to use to create the table
    """

    # Get the first dataframe to get the data schema to use in mysql
    d_1 = list(os.listdir(os.path.join(path, "2017")))[0]
    df = pd.read_excel(os.path.join(path, "2017", d_1))
    df.rename(columns=columns_to_rename, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Build the statement in adapting some types of data
    statement = pd.io.sql.get_schema(df.reset_index(), "data_SCOR")
    statement = statement.replace('"', "")
    statement = statement.replace("INTEGER", "REAL ")
    statement = statement.replace("TEXT", "VARCHAR(400) ")
    # statement = statement.replace("Block REAL", "Block VARCHAR(400)")
    # statement = statement.replace("2010_Yield REAL", "2010_Yield VARCHAR(400)")
    statement = statement.replace("REAL", "VARCHAR(400)")
    statement = statement.replace("id INT", "id INT PRIMARY KEY NOT NULL")
    statement += "ENGINE=InnoDB DEFAULT CHARSET=utf8;"
    # print(statement)
    for col in columns_to_rename.keys():
        statement = statement.replace(col, columns_to_rename[col])
    return statement


def create_cnx(dict: dict) -> dict:
    """
    This function creates an engine to use to connect to a mysql server
    dict: the configuration of the database called in config_parse()
    returns -> cnx_engine: Information with the engine and the connexion to the database information
    """
    engine = sqlalchemy.create_engine(
        "mysql+pymysql://{user}:{pw}@{host}/{db}".format(
            host=dict["HOST"], db=dict["DB"], user=dict["USER"], pw=dict["USER_PWD"]
        )
    )

    cnx = engine.raw_connection()
    cnx_engine = {"cnx": cnx, "engine": engine}
    return cnx_engine


def inject_into_db(path: str, connexion: dict) -> None:
    """
    This function's aim is to add all the data into the database
    path : The parent folder containing the whole data
    connexion : The connexion and engine needed to connect to the database
    """
    compt = 0
    Y = ["2017", "2018", "2019"]
    # Values to replace which don't make sense
    values_list_to_replace = [
        "consider Area of Hubli-Dharwad Mahanagara palike as per notification",
        "NGP",
    ]
    nb_file = 0

    # Count the number of file
    for year in Y:
        nb_file += len(os.listdir(os.path.join(path, year)))

    for year in Y:
        d = list(os.listdir(os.path.join(path, year)))

        for name in d:
            try:
                compt += 1
                df = pd.read_excel(os.path.join(path, year, name))
                print(
                    f"Filling database is {(100*compt)//nb_file} % complete", end="\r"
                )
                # print(year, name)
                df.replace(values_list_to_replace, np.nan, inplace=True)
                # df = pd.read_excel(os.path.join(path, year, name))
                df.rename(columns=columns_to_rename, inplace=True)
                for column in df.columns:
                    try:
                        if not is_numeric_dtype(df[column]):
                            df[column] = df[column].str.lower()
                    except:
                        pass
                df.to_sql(
                    con=connexion["engine"],
                    name="data_SCOR",
                    if_exists="append",
                    index=False,
                )
            except Exception as e:
                print(e)
                print("Warning", year, name)
                exit(1)
    print("The import into database is done")


# This function is used to delete duplicates on a database but seems to be very slow.
# There is no parameter in pandas.to_sql to avoid duplicates see https://github.com/pandas-dev/pandas/pull/29636
def delete_duplicates(path):
    d_1 = list(os.listdir(os.path.join(path, "2017")))[0]
    df = pd.read_excel(os.path.join(path, "2017", d_1))
    br = "\n"
    statement = f"""DELETE a
    FROM
        data_SCOR AS a,
        data_SCOR AS b
    WHERE
        a.ID < b.ID {br}"""
    for column in df.columns:

        if column in columns_to_rename.keys():
            statement += f" AND a.{columns_to_rename[column]} <=> b.{columns_to_rename[column]} {br} "
        else:
            statement += f" AND a.{column} <=> b.{column} {br} "
    return statement


########################################################################################################
# This part is dedicated to the import of the VDSA dataset

file_to_table_name = {
    "dt_area_prod_a_web.xlsx": "Crops_Area_Production",
    "dt_cia_a_web.xlsx": "Crop_Wise_Irrigated_Area",
    "dt_hyv_a_web.xlsx": "HYV_Area_Cereal_Crop",
    "dt_landuse_a_web.xlsx": "Land_Use",
    "dt_sia_a_web.xlsx": "Source_Wise_Irrigated_Area",
    "dt_nca_gca_nia_gia_a_web.xlsx": "Cropped_And_Irrigated_Area",
    "dt_fodder_a_web.xlsx": "Fodder_And_Irrigated_Area",
    "dt_fhp_a_web.xlsx": "Farm_Havest_Price",
    "dt_fert_consumption_a_web.xlsx": "Fertilizer_Consumption",
    "dt_fert_prices_a_web.xlsx": "Fertilizer_Prices",
    "dt_june_julyaug_rainfall_a_web.xlsx": "Annual_Monthly_Actual_Rainfall",
    "dt_population_a_web.xlsx": "Population_Census",
    "dt_livestock_a_web.xlsx": "Livestock_Census",
    "dt_agri_implements_a_web.xlsx": "Agriculture_Implements_Census",
    "dt_market_road_a_web.xlsx": "Market_And_Roads",
    "dt_wages_a_web.xlsx": "Wage_Rate",
    "dt_operational_holdings_a_web.xlsx": "Operation_Holding",
    "dt_normal_rainfall_a_web.xlsx": "Annal_Monthly_Normal_Rainfall",
    "dt_lgp_a_web.xlsx": "Length_Growing_Period",
    "dt_soil_type_a_web.xlsx": "Soil_Type",
    "dt_pet_a_web.xlsx": "Annual_Monthly_Normal_Potential_Evapotranspiration",
    "dt_mai_a_web.xlsx": "Annual_Moisture_Available",
    "dt_aesr_a_web.xlsx": "Agroecological_Subregion",
}
# path_vdsa = "../dataset vdsa"

columns_vdsa_to_rename = {
    "JAN": "JANUARY",
    "FEB": "FEBRUARY",
    "SEPT": "SEPTEMBER",
    "OCT": "OCTOBER",
    "DEC": "DECEMBER",
    "LONG": "LON",
    # "ANNUAL ": "YEARLY ",
}


def create_tables_vdsa(file, path_to_vda: str) -> str:
    """
    This function's aim is to build the statement to create the data table
    path : The parent folder containing the whole data
    returns -> statement : The stamement to use to create the table
    """

    # Get the first dataframe to get the data schema to use in mysql
    template_folder = os.path.join(path_to_vda, "Andhra Pradesh")
    # list_files = list(os.listdir(template_folder))
    # for file in list_files:
    table = file_to_table_name[file]
    df = pd.read_excel(os.path.join(template_folder, file))
    # df.rename(columns=columns_to_rename, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Build the statement in adapting some types of data
    statement = pd.io.sql.get_schema(df.reset_index(), table)
    # print(statement)
    # print()
    # print()
    statement = statement.replace('"', "")
    statement = statement.replace("index INTEGER", "id INT")
    statement = statement.replace("INTEGER", "VARCHAR(30)")
    statement = statement.replace("TEXT", "VARCHAR(200)")
    statement = statement.replace(
        "id INT", "id INT PRIMARY KEY NOT NULL AUTO_INCREMENT"
    )
    statement = statement.replace("REAL", "VARCHAR(30)")
    for col in columns_vdsa_to_rename.keys():
        statement = statement.replace(col, columns_vdsa_to_rename[col])

    statement += "ENGINE=InnoDB DEFAULT CHARSET=utf8;"

    # print(statements)
    return statement


def inject_into_db_vdsa(path_vdsa: str, connexion: dict) -> None:
    """
    This function's aim is to add all the data into the database
    path_vdsa : The parent folder containing the whole vdsa dataset
    connexion : The connexion and engine needed to connect to the database
    """

    nb_file = 0
    # Count the number of file
    for file in os.listdir(path_vdsa):
        nb_file += len(os.listdir(os.path.join(path_vdsa, file)))
    compt = 0
    for state in os.listdir(path_vdsa):
        liste_f = list(os.listdir(os.path.join(path_vdsa, state)))

        for file in liste_f:
            try:
                compt += 1
                df = pd.read_excel(os.path.join(path_vdsa, state, file))
                df.fillna(value=np.nan, inplace=True)
                # df.fillna("NULL", inplace=True)
                df.replace("None", "NULL", inplace=True)
                # df["id"] = df.index.tolist()
                print(
                    f"Filling database is {(100*compt)//nb_file} % complete", end="\r"
                )
                # print(year, name)
                # df.replace(values_list_to_replace, np.nan, inplace=True)
                # df = pd.read_excel(os.path.join(path, year, name))
                # df.rename(columns=columns_to_rename, inplace=True)
                for column in df.columns:
                    try:
                        if not is_numeric_dtype(df[column]):
                            df[column] = df[column].str.lower()
                    except:
                        pass
                for column in df.columns:
                    try:
                        df.rename(
                            columns={column: columns_vdsa_to_rename[column]},
                            inplace=True,
                        )
                    except:
                        pass

                df.to_sql(
                    con=connexion["engine"],
                    name=file_to_table_name[file],
                    if_exists="append",
                    index=False,
                )
            except Exception as e:
                print(e)
                # print("Warning", year, name)
                exit(1)
    print("The import into database is done")


# print(create_tables_vdsa(path_to_vda))
