import polars as pl
import os
import numpy as np

cols = ['fsc', 'ssc', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 
        'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 
        'uv1', 'uv2', 'uv3', 'uv4', 'uv5', 'uv6', 'uv7', 'uv8', 'uv9', 'uv10', 'uv11', 'uv12', 'uv13', 'uv14', 'uv15', 'uv16', 'uv17', 'uv18', 'uv19', 'uv20', 'uv21', 'uv22', 
        'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 
        'yg1', 'yg2', 'yg3', 'yg4', 'yg5', 'yg6', 'yg7', 'yg8', 'yg9', 
        'imgb1', 'imgb2', 'imgb3', 
        'center_of_mass', 'delta', 'diffusivity', 'eccentricity', 'lightloss', 'long_axis_moment', 'max_intensity', 'radial_moment', 
        'short_axis_moment', 'size', 'total_intensity', 'time']


def parse_file_flowjo(csv_path):
    filename = os.path.basename(csv_path)
    events = pl.read_csv(f"{csv_path}")
    events = (events
                .rename(lambda x: x.replace(" ", "_").lower())
                .select(pl.selectors.starts_with(cols))
                .with_columns(pl.Series(values=np.arange(events.shape[0]), name="event"))
                .with_columns(pl.lit(filename).alias("filename"))
                )
    return events

def get_column_types_flowjo(events_df):
    feature_cols = events_df.select(pl.selectors.starts_with(cols[:-1])).columns
    metadata_cols = [c for c in events_df.columns if c not in feature_cols]
    return feature_cols, metadata_cols
