
# import libraries
import polars as pl
from numpy import intersect1d, min
from multiprocessing import cpu_count, get_context
from os.path import dirname
from os import listdir

# import self-written modules
from bacpy.file_parser_evoware import read_metadata
from bacpy.file_parser_icontrol import parse_file_icontrol



# now building the wrapper function to parse a whole evoware dataset
def parse_dataset_icontrol(mapping=None, n_jobs=-1):
    """
    function to parse a TECAN evoware dataset to obtain a neatly formatted dataframe (long format)
    layout:     str     REQUIRED: path to the layout file:  **plates  <-->  hotel**
    mapping:    str     OPTIONAL: mapping-file:             **strains <-->  wells**
    n_jobs:     int     number of threads: defaults to -1, which means using all cores of the machine 
    """

    # read the inputs
    mapping_tab = read_metadata(mapping)

    # convert ID column to strainID if not present
    if "ID" in mapping_tab.columns:
        mapping_tab = mapping_tab.rename({"ID": "strainID"})

    # drop stuff
    mapping_tab = mapping_tab.filter(pl.col("strainID").is_not_null())
    mapping_tab = mapping_tab[0:min([mapping_tab.shape[0], 96])]
    mapping_tab.columns = [col[0].lower() + col[1:] for col in mapping_tab.columns]

    # detemine threads
    if n_jobs == -1:
        n_jobs = cpu_count()

    # function body
    assay_dir = dirname(mapping)
    print(f"parsing: {mapping}")

    # get the filelist
    file_list = listdir(assay_dir)
    file_list = [file for file in file_list if file.endswith(".xlsx")]
    file_list = [file for file in file_list if not "layout" in file]
    file_list = [file for file in file_list if not "mapping" in file]
    file_list = [file for file in file_list if not file.startswith(".")]
    file_list = [f"{assay_dir}/{file}" for file in file_list if not "mapping" in file]
    file_list = sorted(file_list)


    # using parallel processing to read the files
    with get_context("spawn").Pool(processes=n_jobs) as pool:
        parsed_list = pool.map(parse_file_icontrol, file_list)

    # concat and post processing
    parsed = (pl.concat(parsed_list, how="vertical_relaxed")
                    .with_columns(pl.lit("icontrol").alias("dataset"))
             )

    # if mapping table provided
    parsed = parsed.join(mapping_tab, how="left", on=intersect1d(parsed.columns, mapping_tab.columns), validate="m:1")

    return parsed
