from bacpy.file_parser_flowjo import parse_file_flowjo, cols
import os
import polars as pl
import time

def parse_dataset_flowjo(mapping):
    data_path = os.path.dirname(mapping)
    if mapping.endswith(".xlsx") or mapping.endswith(".ods"):
        mapping = pl.read_excel(mapping)
    elif mapping.endswith(".tsv"):
        mapping = pl.read_csv(mapping, separator="\t")
    elif mapping.endswith(".csv"):
        mapping = pl.read_csv(mapping)
    events_ls = []
    filelist = mapping["filename"]
    for idx, filename in enumerate(filelist):
        csv_path = f"{data_path}/{filename}"
        if os.path.exists(csv_path):
            progress_pct = round((idx/len(filelist))*100, 2)
            print(f"parsing: {csv_path} - {progress_pct}%")
            events = parse_file_flowjo(csv_path)
            events_ls.append(events)
        else:
            ValueError(f"File does not exist: " + filename)

    start = time.perf_counter()
    events_df = pl.concat(events_ls, how="diagonal_relaxed")
    time_passed_concat = round(time.perf_counter() - start, 2)

    start = time.perf_counter()
    feature_cols = events_df.select(pl.selectors.starts_with(cols)).columns
    events_df = events_df.drop_nulls(feature_cols)
    time_passed_remove = round(time.perf_counter() - start, 2)

    start = time.perf_counter()
    events_df = events_df.join(mapping, how="left", on="filename")
    time_passed_mapping = round(time.perf_counter() - start, 2)


    print(f"CONCATENATING TOOK [s]: {time_passed_concat}")
    print(f"CLEAN-UP [s]: {time_passed_remove}")
    print(f"MAPPING TOOK [s]: {time_passed_mapping}")
    print(f"TOTAL DATASET SIZE: {events_df.shape[0]}")
    return events_df


