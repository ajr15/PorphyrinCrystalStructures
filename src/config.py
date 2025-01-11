COD_SEARCH_API_ENDPOINT = "https://www.crystallography.net/cod/result"
COD_FILE_API_ENDPOINT = "https://www.crystallography.net/cod/$CODID.cif"

import os
if "CRYSTAL_SRC_DIR" not in os.environ:
    os.environ["CRYSTAL_SRC_DIR"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "CRYSTAL_DATA_DIR" not in os.environ:
    os.environ["CRYSTAL_DATA_DIR"] = os.path.join(os.environ["CRYSTAL_SRC_DIR"], "data")
if "CRYSTAL_MAIN_DB" not in os.environ:
    os.environ["CRYSTAL_MAIN_DB"] = os.path.join(os.environ["CRYSTAL_SRC_DIR"], "main.db")

DATA_DIR = os.environ["CRYSTAL_DATA_DIR"]
DISPLACED_STRUCTS_DIR = os.environ["CRYSTAL_SRC_DIR"] + "/displaced_structures"