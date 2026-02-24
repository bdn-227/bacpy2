
# import libraries
import polars as pl
from numpy import intersect1d, min
from multiprocessing import cpu_count, get_context
from os.path import dirname
from os import listdir

# import self-written modules
from .file_parser_evoware import read_metadata
from .file_parser_icontrol import parse_file_icontrol



# now building the wrapper function to parse a whole evoware dataset
def parse_dataset_icontrol(mapping: str, 
                           n_jobs: int = -1) -> pl.DataFrame:
    """
    Orchestrates parallel parsing of TECAN iControl Excel files in a directory.

    This wrapper identifies all valid measurement files in the directory associated 
    with the mapping file, processes them in parallel using a multiprocessing pool, 
    and consolidates the results. It also performs metadata cleaning, such as 
    renaming 'ID' to 'strainID', standardizing column casing, and joining 
    experimental metadata onto the measurement data.

    Args:
        mapping: Path to the mapping/metadata file. The parent directory of this 
            file is used as the search path for .xlsx data files.
        n_jobs: Number of CPU cores to use for parallel parsing. Defaults to -1 
            (uses all available cores).

    Returns:
        A long-format Polars DataFrame containing measurements from all 
        discovered files, enriched with metadata from the mapping table.

    Note:
        - Files containing 'layout', 'mapping', or starting with '.' are excluded.
        - The mapping table is limited to a maximum of 96 entries.
        - Parallelism is implemented using the 'spawn' context for cross-platform 
          compatibility.
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
