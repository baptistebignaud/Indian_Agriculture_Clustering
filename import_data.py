from logging import exception
from utile.database import (
    config_parse,
    create_table,
    create_cnx,
    inject_into_db,
    delete_duplicates,
    create_tables_vdsa,
    inject_into_db_vdsa,
)
import argparse
import os

list_tables = [
    "Crops_Area_Production",
    "Crop_Wise_Irrigated_Area",
    "HYV_Area_Cereal_Crop",
    "Land_Use",
    "Source_Wise_Irrigated_Area",
    "Cropped_And_Irrigated_Area",
    "Fodder_And_Irrigated_Area",
    "Farm_Havest_Price",
    "Fertilizer_Consumption",
    "Fertilizer_Prices",
    "Annual_Monthly_Actual_Rainfall",
    "Population_Census",
    "Livestock_Census",
    "Agriculture_Implements_Census",
    "Market_And_Roads",
    "Wage_Rate",
    "Operation_Holding",
    "Annal_Monthly_Normal_Rainfall",
    "Length_Growing_Period",
    "Soil_Type",
    "Annual_Monthly_Normal_Potential_Evapotranspiration",
    "Annual_Moisture_Available",
    "Agroecological_Subregion",
]


def main() -> None:
    """
    This function's aim is to fill the database with excel file
    returns -> None
    """
    parser = argparse.ArgumentParser(description="Choose to import database")
    parser.add_argument(
        "--type",
        # "typeImport",
        metavar="t",
        type=str,
        # nargs="+",
        help="what kind of imported you want to do: base is for the dataset provided by SCOR, external is VDSA dataset, both is for both",
        choices=["base", "external", "all"],
        default="base",
    )

    args = parser.parse_args()
    # print(args.accumulate(args.integers))
    path = "./data"
    path_vdsa = "./dataset vdsa"
    # Get the configuration of the data
    parser_config = config_parse()

    # Connect to database
    connexion = create_cnx(parser_config)
    cnx = connexion["cnx"]
    curs = cnx.cursor()
    if args.type in ["base", "all"]:
        # Drop former table if necessary
        print("base")
        try:
            curs.execute("DROP TABLE data_SCOR")
            print("drop")
        except:
            pass

        # Create table for the data
        statement = create_table(path)
        curs.execute(statement)

        # Inject all the data into db
        inject_into_db(path, connexion)
        # Following code is really slow
        # statement = delete_duplicates(path)
        # try:
        #     curs.execute(statement)
        #     print("drop duplicates done")
        # except Exception as e:
        #     print("drop duplicates failed")
        #     print(e)
        # Close connexion
    if args.type in ["external", "all"]:
        print("external")
        for table in list_tables:
            try:
                curs.execute(f"DROP TABLE {table}")
            except:
                pass
        template_folder = os.path.join(path_vdsa, "Andhra Pradesh")
        list_files = list(os.listdir(template_folder))
        # statements = create_tables_vdsa(path_vdsa)
        for i, file in enumerate(list_files):
            # print(i)
            statement = create_tables_vdsa(file, path_vdsa)
            try:

                curs.execute(statement)
                # if i == 12:
                #     print(statement)
                #     print()
            except Exception as e:
                print(file)
                print()
                print(statement)
                print()
                print(e)
                # exit(1)
            # print(statement)

        print("end of creating vdsa tables \n")
        inject_into_db_vdsa(path_vdsa, connexion)
    curs.close()
    cnx.close()


if __name__ == "__main__":
    main()
