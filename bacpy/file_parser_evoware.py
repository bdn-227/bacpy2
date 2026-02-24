
# load libraries
from os.path import basename
import polars as pl
import numpy as np
from itertools import chain
from re import findall


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_file_evoware(asc_path: str) -> pl.DataFrame:
    """
    Wrapper function to correctly parse ASCII files generated from TECAN-evoware.

    This function handles the specific formatting and metadata structure of 
    Tecan Evoware .asc exports, converting the raw text data into a structured 
    Polars format.

    Args:
        asc_path: The file path to the input ASCII file. Must end with '.asc'.

    Returns:
        A Polars DataFrame containing the parsed data from the file.

    """

    # the the file name
    filename = basename(asc_path)
    
    # perform parsing using the correct function
    if filename.endswith(".asc"):

        # conditional branch
        if ("od" in filename) and ("ex" not in filename):
            parsed_data = get_od_data(asc_path)
        if ("od" not in filename) and ("ex" in filename):
            parsed_data = get_ex_data(asc_path)

        # fix over values
        if parsed_data["response"].dtype == pl.String:
            parsed_data = parsed_data.with_columns(pl.col("response").replace("Overflow", 1e10).cast(float))

        # adjusting some data types here
        parsed_data = (parsed_data
                            .with_columns(pl.col("ex").cast(int))
                            .with_columns(pl.col("em").cast(int))
                            
                            )

        return parsed_data

    # or maybe not lol
    else:
        raise ValueError(f"make sure to provide a valid ascii file ending in .asc")



def get_ex_data(asc_path):
    """
    function to parse an ascii file containing fluorescent spectral data (either a scan or single measurement)
    """

    #asc_path = "../../../../../../../biodata.localized/dep_psl/common/RGO_Tecan/Aude/test1-Carbon/exp2/day4/250705-263_ex508_540.asc"
    
    # perform parsing
    try:
        # read
        ex_tab = pl.read_csv(asc_path, 
                             separator="\t", 
                             truncate_ragged_lines=True, 
                             encoding="iso8859-1", 
                             comment_prefix="S",
                             infer_schema_length=100000)
        

        # define possible wells
        measurement_mode = "Fluorescence Bottom Reading"
        rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        all_wells = np.array(list(chain.from_iterable([[i+str(j) for j in range(1, 13)] for i in rows])))

        # extract the fluorescent responses
        if "[nm]" in ex_tab.columns:
            """
            conditional branch for scan
            """
            value_rows = np.arange(0, ex_tab["[nm]"].is_null().arg_max())
            values = (ex_tab[value_rows]
                                .select(all_wells)
                                .with_columns(ex_tab[value_rows, "[nm]"])
                                .unpivot(index="[nm]")
                                .rename({"[nm]": "em", "variable": "well", "value": "response"}))


            # extract the meta data
            metaData = ex_tab[ex_tab["[nm]"].str.starts_with("Date of measurement").arg_max():,:]

            # get temporal informationh
            startTime = metaData[-1,0].split(",")
            date = startTime[0].replace("Date: ", "")
            start_time = startTime[1].replace(" Time: ", "")
            
            # temperature
            temp = float(findall(r'\d+\.\d+', metaData[14,0])[0])

            # additional information
            label = None
            device = int(metaData[5,0].replace("Instrument serial number: ", ""))
            ex = int(findall(r'\d+', metaData[21,0])[0])
            measurement_type = "scan" if values.shape[0] / np.isin(ex_tab.columns, all_wells).sum() > 1 else "single"
            filename = basename(asc_path)
        elif "Raw data" in ex_tab.columns:
            """
            for single fluorescent measurement
            """
            well_ids = ex_tab["Layout"].drop_nulls()
            values = (ex_tab
                        .filter(pl.col("Layout").is_in(well_ids))
                        .select("Raw data", "Layout")
                        .rename({"Raw data": "response", "Layout": "wellId"})
                        .with_columns(pl.col("response").cast(float))
                        .join(pl.DataFrame({"wellId": well_ids, "well": all_wells[well_ids.str.split("_").list[1].cast(int)-1]}), how="left", on="wellId")
                        .drop("wellId")
                        )
            # extract the meta data
            metaData = ex_tab[ex_tab["Raw data"].str.starts_with("Date of measurement").arg_max():,:]

            # get temporal informationh
            startTime = metaData[-1,0].split(",")
            date = startTime[0].replace("Date: ", "")
            start_time = startTime[1].replace(" Time: ", "")
            
            # temperature
            temp = float(findall(r'\d+\.\d+', metaData[14,0])[0])

            # additional information
            label = None
            device = int(metaData[5,0].replace("Instrument serial number: ", ""))
            measurement_type = "single"
            filename = basename(asc_path)
            ex = int(findall(r'\d+', metaData[3,0])[0])
            em = int(findall(r'\d+', metaData[3,0])[1])
            values = values.with_columns(pl.lit(em).alias("em"))
        else:
            print(ex_tab.columns)
            raise ValueError(f"ask nik to fix unknown fluroescent messurement mode - 12469182")

        # put data together
        values = (values
                        .with_columns(pl.lit(start_time).alias("start_time"))
                        .with_columns(pl.lit(label).alias("label"))
                        .with_columns(pl.lit(ex).alias("ex"))
                        .with_columns(pl.lit(measurement_type).alias("measurement_type"))
                        .with_columns(pl.lit(measurement_mode).alias("measurement_mode"))
                        .with_columns(pl.lit(temp).alias("temp"))
                        .with_columns(pl.lit(filename).alias("filename"))
                        .with_columns(pl.lit(device).alias("device"))
                        .with_columns(pl.lit(date).alias("date"))
                        .with_columns(pl.Series(np.char.strip(values["well"].to_numpy().astype(str), "|".join(np.array(list(range(1,13))).astype(str)))).alias("row"))
                        .with_columns(pl.Series(np.char.strip(values["well"].to_numpy().astype(str), "|".join(rows)).astype(int)).alias("column"))
                        .with_columns(("wv" + pl.col("ex").cast(str) + "." + pl.col("em").cast(str)).alias("ex_em"))
                        .select([
                                 'filename', 
                                 'date', 
                                 'device', 
                                 'measurement_mode', 
                                 'ex', 
                                 'em', 
                                 'temp',
                                 'start_time',
                                 'label', 
                                 'measurement_type', 
                                 'row', 
                                 'column',
                                 'well', 
                                 'ex_em',
                                 'response', 
                                 ])
                        )
        
        # return stuff
        print(f"parsed: {asc_path}")
        return values
    except Exception as E:
        print(f"error parsing: {asc_path}:\n{E}\n")



def get_od_data(asc_path):
    """
    function to parse an ascii file containing optical density data (either a scan or single measurement)
    """

    # perform parsing
    try:
        # read
        od_tab = pl.read_csv(asc_path, 
                            separator="\t", 
                            truncate_ragged_lines=True, 
                            encoding="iso8859-1", 
                            comment_prefix="S",
                            infer_schema_length=10000)

        # define possible wells
        rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        all_wells = np.array(list(chain.from_iterable([[i+str(j) for j in range(1, 13)] for i in rows])))

        # get the mode
        measurement_mode = od_tab[112,0].replace(" ", "")

        # get wells that were measured and corresponding values
        measured_samples = od_tab[:,1].drop_nulls()
        values = (od_tab
                        .filter(pl.col("Layout").is_in(measured_samples))
                        .select("Raw data")
                        .rename({"Raw data": "response"})
                        .with_columns(pl.Series(all_wells).alias("well")))


        # get meta data
        start_time = od_tab[-1,0].split(",")
        date = start_time[0].replace("Date: ", "")
        start_time = start_time[1].replace(" Time: ", "")
        temp = float(od_tab[0,0].replace(" °C", ""))
        label = None
        device=int(od_tab[102,0].replace("Instrument serial number: ", ""))
        em=int(od_tab[100,0].replace("nm", ""))
        ex = em
        measurement_type = "single"
        filename = basename(asc_path)

        # put data together
        values = (values
                        .with_columns(pl.lit(start_time).alias("start_time"))
                        .with_columns(pl.lit(label).alias("label"))
                        .with_columns(pl.lit(ex).alias("ex"))
                        .with_columns(pl.lit(em).alias("em"))
                        .with_columns(pl.lit(measurement_type).alias("measurement_type"))
                        .with_columns(pl.lit(measurement_mode).alias("measurement_mode"))
                        .with_columns(pl.lit(temp).alias("temp"))
                        .with_columns(pl.lit(filename).alias("filename"))
                        .with_columns(pl.lit(device).alias("device"))
                        .with_columns(pl.lit(date).alias("date"))
                        .with_columns(pl.Series(np.char.strip(values["well"].to_numpy().astype(str), "|".join(np.array(list(range(1,13))).astype(str)))).alias("row"))
                        .with_columns(pl.Series(np.char.strip(values["well"].to_numpy().astype(str), "|".join(rows)).astype(int)).alias("column"))
                        .with_columns(("wv" + pl.col("ex").cast(str) + "." + pl.col("em").cast(str)).alias("ex_em"))
                        .select([
                                 'filename', 
                                 'date', 
                                 'device', 
                                 'measurement_mode', 
                                 'ex', 
                                 'em', 
                                 'temp',
                                 'start_time',
                                 'label', 
                                 'measurement_type', 
                                 'row', 
                                 'column',
                                 'well', 
                                 'ex_em',
                                 'response',
                                 ])
                        )
        
        # return stuff
        print(f"parsed: {asc_path}")
        return values
    except Exception as E:
        print(f"error parsing: {asc_path}:\n{E}\n")



def parse_batch(sample, chunked):
    """
    function to parse a batch of files that all belong to one plate
    """

    # parse the data
    parsed_plate = (pl.concat([parse_file_evoware(filename) for filename in chunked], how="vertical_relaxed")
                            .with_columns(pl.lit(sample).alias("sampleID"))
                            )
    
    # add notes to dataframe
    dates = parsed_plate.group_by("date").len()
    date = dates[dates["len"].arg_max(),"date"]
    return parsed_plate.with_columns(pl.lit(date).alias("date"))


def read_metadata(path):
    """
    function to read the metadata belonging to the tecan experiments, this can be 
    a layout-file or a mapping file
    accaptable formats are .tsv .csv .xlsx .odt .ods
    """
    # get the layout file
    if path.endswith("tsv"):
        return pl.read_csv(path, separator="\t").select(pl.exclude(""))
    elif path.endswith("csv"):
        return pl.read_csv(path).select(pl.exclude(""))
    elif path.endswith("xlsx") or path.endswith("odt") or path.endswith("ods"):
        return pl.read_excel(path).select(pl.exclude(""))
    else:
        raise ValueError('File extensing for meta-data file must be one of the following: .tsv .csv .xlsx .odt .ods')


