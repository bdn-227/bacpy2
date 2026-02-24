
# import libraries
import polars as pl
from numpy import intersect1d
from multiprocessing import cpu_count, get_context
from os.path import dirname
from os import listdir

# import self-written modules
from .file_parser_evoware import read_metadata, chunk_list, parse_batch



# now building the wrapper function to parse a whole evoware dataset
def parse_dataset_evoware(layout, mapping=None, n_jobs=-1, manifest_file="parsing_manifest"):
    """
    function to parse a TECAN evoware dataset to obtain a neatly formatted dataframe (long format)
    layout:     str     REQUIRED: path to the layout file:  **plates  <-->  hotel**
    mapping:    str     OPTIONAL: mapping-file:             **strains <-->  wells**
    n_jobs:     int     number of threads: defaults to -1, which means using all cores of the machine 
    """

    # read the inputs
    layout_tab = read_metadata(layout)
    if ("sample" in layout_tab.columns) and ("sampleID" not in layout_tab.columns):
        layout_tab = layout_tab.rename({"sample": "sampleID"})

    # detemine threads
    if n_jobs == -1:
        n_jobs = cpu_count()

    # function body
    assay_dir = dirname(layout)
    print(f"parsing: {layout}")

    # drop stuff
    layout_tab = layout_tab.filter(pl.col("sampleID").is_not_null())

    # get the filelist
    file_list = listdir(assay_dir)
    file_list.remove(layout.split("/")[-1])
    file_list = [file for file in file_list if not file.startswith("._")]
    file_list = [f"{assay_dir}/{file}" for file in file_list if file.endswith(".asc")]
    file_list = sorted(file_list)

    # verify the schema length
    n_plates = layout_tab.shape[0]
    file_extensions = [file.split("_")[-1] for file in file_list]
    scanned_features = list(dict.fromkeys(file_extensions))
    files_per_plate = len(scanned_features)
    expected_extensions = scanned_features * n_plates

    # print ogs
    print(f"measurements per plate:    {files_per_plate}")
    print(f"total number of plates:    {n_plates}")
    print(f"total number of files:     {len(file_extensions)}")
    print(f"expected number of files:  {len(expected_extensions)}")
    if len(file_extensions) != len(expected_extensions):
        ValueError(f"NUMBER OF DETECTED PLATES DOES NOT MATCH NUMBER OF EXPECTED PLATES..\nEXPECTED: {len(expected_extensions)}\nFOUND: {len(file_extensions)}")

    # iterate through file-list and verify scheme
    for idx, e in enumerate(file_extensions):
        if e == expected_extensions[idx]:
            pass
        else:
            raise ValueError(f"found {file_list[idx]} but expected {expected_extensions[idx]}")


    # get lists for dataprocessing
    chunked_file_list = list(chunk_list(file_list, files_per_plate))
    sample_list = list(layout_tab["sampleID"])
    parsing_manifest = pl.DataFrame({"sampleID": sample_list, "files_used": chunked_file_list}).explode("files_used")

    # using parallel processing to read the files
    with get_context("spawn").Pool(processes=n_jobs) as pool:
        parsed = pl.concat(pool.starmap(parse_batch, zip(sample_list, chunked_file_list)), how="vertical_relaxed")

    # concat and post processing
    parsed = (parsed
                    .join(layout_tab, how="left", on=intersect1d(parsed.columns, layout_tab.columns), validate="m:1")
                    .with_columns(pl.lit("evoware").alias("dataset"))
                    .with_columns(pl.col("sampleID").alias("filename"))
                    )

    # if mapping table provided
    if mapping:
        mapping_tab = read_metadata(mapping)
        parsed = parsed.join(mapping_tab, how="left", on=intersect1d(parsed.columns, mapping_tab.columns), validate="m:1")
    else:
        parsed = parsed.with_columns(pl.lit("unknown").alias("strainID"))
    parsing_manifest.write_csv(manifest_file + ".tsv", separator="\t")
    return parsed
