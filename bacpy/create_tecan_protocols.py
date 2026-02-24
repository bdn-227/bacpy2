

# import modules
import numpy as np
import polars as pl
from typing import Optional, Union, Dict

# import scripts
from .tecan_strings import icontrol_plate_range, icontrol_header_abs, icontrol_fluorescent_scan, icontrol_end
from .spark_strings import spark_header_abs, spark_fluorescence_scan, spark_end



def create_tecan_scripts(
                            features: Optional[Union[np.ndarray, Dict, pl.DataFrame]] = None,
                            device: str = "icontrol",  # options: "icontrol" or "spark"
                            outfile: str = "tecan",
                            n_features: int = -1,
                            abs_nm: int = 720,
                            nm_interval: int = 15,
                            gain: int = 100,
                            zposition: int = 20000,
                            lagtime: int = 0,
                            integrationtime: int = 20,
                            numflashes: int = 3,
                            settletime: int = 20,
                        ) -> None:
    """
    Generates configuration files (.mth or .xml) for Tecan plate readers.

    This utility converts desired spectral features into structured scan protocols. 
    It supports full scans, specific wavelength pairs, or optimized protocols 
    derived from model feature importances.

    Args:
        features: Input determining the scan range.
            - None: Full scan (240-810nm).
            - np.ndarray: List of excitation wavelengths; emission starts at ex+45nm.
            - dict: {ex: start_em} or {ex: [start_em, end_em]}.
            - pl.DataFrame: Feature importance table; automatically creates 
              clustered emission scans for top important wavelengths.
        device: The target software format ('icontrol' or 'spark').
        outfile: Output filename (extension added automatically).
        n_features: Number of top features to use if providing a DataFrame.
        abs_nm: Wavelength for OD/Absorbance measurements (usually 600 or 720nm).
        nm_interval: Step size for emission scans (default 15nm).
        gain: Detector amplification (default 100).
        zposition: Vertical distance of detector from plate (default 20000).
        lagtime: Time delay between flash and measurement (default 0ms).
        integrationtime: Measurement duration per flash (default 20ms).
        numflashes: Number of flashes averaged per well (default 3).
        settletime: Delay after plate movement to reduce liquid vibration (default 20ms).

    Returns:
        None. Writes the script file to the local directory.
    """
    if features is None:
        features = np.arange(240, 810+1, nm_interval)

    if type(features) == pl.DataFrame:
        # 1. sort by importance and subset top n
        feature_importances = features.sort("importance", descending=True)

        # subsetting of needed
        if (n_features > 0) and (n_features<feature_importances.shape[0]):
            feature_importances = feature_importances[:n_features]


        # 2. cluster by excitation wavelength, patch emission if needed
        ex_em_dict = {}
        for em_wv in feature_importances.group_by("ex").agg(pl.col("importance").sum().alias("aggsum")).sort("aggsum", descending=True)["ex"].unique(maintain_order=True):

            # subset
            ex_subset = feature_importances.filter(pl.col("ex") == em_wv)
            em_min = ex_subset["em"].min()
            em_max = ex_subset["em"].max()
            em_range = np.arange(em_min, em_max+1, nm_interval)

            # add nm intervals if missing
            while len(em_range) < 5:
                em_range_min = em_range.min()
                em_range_max = em_range.max()
                em_range =  np.arange(em_range_min-nm_interval, em_range_max+nm_interval+1, nm_interval)
                em_range = em_range[em_range<=840]
                em_range = em_range[em_range>=285]

            # save results in dict
            ex_em_dict[em_wv] = em_range


    elif type(features) == np.ndarray:

        # create the final dict
        ex_em_dict = {}
        for ex in features:
            
            # create the ems
            ems = np.arange(285, 840+1, nm_interval)
            ems = ems[ems >= ex+45]
            ex_em_dict[ex] = ems


    elif type(features) == dict:

        if type(list(features.values())[0]) == list:
            ex_em_dict = {}
            for ex in features:
                
                # create the ems
                ems = np.arange(features[ex][0], features[ex][1]+1, nm_interval)
                ex_em_dict[ex] = ems

        else:
            ex_em_dict = {}
            for ex in features:
                
                # create the ems
                ems = np.arange(285, 840+1, nm_interval)
                ems = ems[ems >= ex+features[ex]]
                ex_em_dict[ex] = ems

    
    # remove empty scans
    ex_em_dict = {key: val for key, val in ex_em_dict.items() if len(val) > 2}
    

    if device == "icontrol":
        write_icontrol(outfile=outfile, 
                       ex_em_dict=ex_em_dict, 
                       numflashes=numflashes,
                       abs_nm=abs_nm,
                       settletime=settletime,
                       nm_interval=nm_interval,
                       gain=gain,
                       zposition=zposition,
                       lagtime=lagtime,
                       integrationtime=integrationtime,
                       )
    
    elif device == "spark":
        write_spark(outfile=outfile, 
                    ex_em_dict=ex_em_dict, 
                    numflashes=numflashes,
                    abs_nm=abs_nm,
                    settletime=settletime,
                    nm_interval=nm_interval,
                    gain=gain,
                    zposition=zposition,
                    lagtime=lagtime,
                    integrationtime=integrationtime,
                    )



def write_icontrol(outfile, 
                   ex_em_dict, 
                   numflashes,
                   abs_nm,
                   settletime,
                   nm_interval,
                   gain,
                   zposition,
                   lagtime,
                   integrationtime):
    """
    helper function to write the iControl 
    scripts to disk
    """

    # now generate the new tecan scripts
    with open(outfile+".mdfx", "w") as text_file:

        # write the header
        text_file.write(icontrol_header_abs(plate = "GRE96ft",
                                            numflashes=numflashes,
                                            abs_nm=abs_nm,
                                            settletime=settletime,
                                            ))

        # counters for ids
        id=10

        # write the fluorescence measurements
        for idx, key in enumerate(ex_em_dict.keys()):

            # add plate range in case we reached 10 measurements
            if idx%9==0:
                string, id = icontrol_plate_range(id)
                text_file.write(string)
            
            # add a scan measurement for each
            string, id = icontrol_fluorescent_scan(ex=key, 
                                                   em_range=ex_em_dict[key], 
                                                   idx=idx, 
                                                   id=id,
                                                   nm_interval=nm_interval,
                                                   gain=gain,
                                                   zposition=zposition,
                                                   lagtime=lagtime,
                                                   integrationtime=integrationtime,
                                                   numflashes=numflashes,
                                                   settletime=settletime,
                                                   )
            text_file.write(string)
        
        # add end of file
        string = icontrol_end(id, 
                              idx,
                              numflashes=numflashes,
                              abs_nm=abs_nm,
                              settletime=settletime)
        text_file.write(string)



def write_spark(outfile, 
                ex_em_dict, 
                numflashes,
                abs_nm,
                settletime,
                nm_interval,
                gain,
                zposition,
                lagtime,
                integrationtime):
    """
    helper function to write spark-icontrol scripts
    """

    # define num measurements
    num_measurements = len(ex_em_dict.keys())

    # now generate the new tecan scripts
    with open(outfile+".xml", "w") as text_file:

        # write the header
        text_file.write(spark_header_abs(wells="all",
                                         numflashes=numflashes,
                                         abs_nm=abs_nm,
                                         settletime=settletime,
                                         num_measurements=num_measurements))
        id=2

        # write the fluorescence measurements
        for idx, key in enumerate(ex_em_dict.keys()):
            
            # add a scan measurement for each
            string, id = spark_fluorescence_scan(id=id,
                                                 idx=idx,
                                                 ex=key,
                                                 em_range=ex_em_dict[key],
                                                 nm_interval=nm_interval,
                                                 gain=gain,
                                                 zposition=zposition,
                                                 lagtime=lagtime,
                                                 integrationtime=integrationtime,
                                                 numflashes=numflashes,
                                                 settletime=settletime,)
            text_file.write(string)


        
        # add end of file
        string = spark_end(id,
                           numflashes=numflashes,
                           abs_nm=abs_nm,
                           settletime=settletime)
        text_file.write(string)

