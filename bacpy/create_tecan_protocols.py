

# import modules
import numpy as np
import polars as pl

# import scripts
from .tecan_strings import icontrol_plate_range, icontrol_header_abs, icontrol_fluorescent_scan, icontrol_end
from .spark_strings import spark_header_abs, spark_fluorescence_scan, spark_end



def create_tecan_scripts(features=None,
                         device="icontrol",  # icontrol or spark
                         outfile="tecan",
                         n_features=-1,
                        
                         # less important parameters
                         abs_nm=720,
                         nm_interval=15,
                         gain=100,
                         zposition=20000,
                         lagtime=0,
                         integrationtime=20,
                         numflashes=3,
                         settletime=20,
                         ):
    """
    function to create the configuration files for tecan plate readers.
    As of know, Tecan iControl and Tecan Spark software is supported
    ----
    Parameter:
        features        (numpy.ndarray | dict | polars.DataFrame): 
                        default = None; creates a full scan with nm_interval
                        list of features that one wishes to be scanned. 
                            numpy.ndarray should only contain excitation wavelengths; the script will produce a full excitation scan with an offset of 45nm
                                i.e. if one of ex-wv is 300, the corresponding emission scan will cover 345 --> 840nm
                            dict contains the desired excitation wave-lengths as keys & the starting emission wavelength as value
                                i.e. a dict with features[500] = 550 will produce a script that uses 500nm as excitation wavelength, where the emission scan reaches from 550 --> 840nm
                                alternatively, the values can be lists, that denote the starting and endling emission wavelength
                                 i.e. features[450] = [550, 600] produces a script that excites with 450 nm and records the emission from 550 --> 600nm
                            polars.DataFrame should be obtained from bacpy.randomForest.get_features_importances(), the function aggregates the feature importances 
                                by excitation wavelength uses those features to create an emission scan that goes from em.min() to em.max(); if there are less than 5 emission
                                wavelengths, the emission spectra will be appended to allow for normalization of the downstream data using z-score transformation | area-under-curve normalization
        device          (str): 
                        default = "icontrol"
                        options are "icontrol" or "spark" xml-type of files
        outfile         (str): 
                        default = "tecan"
                        name for the output file. the respective extension is added automatically
        n_features      (int)
                        default=-1
                        only works with features=pl.DataFrame; determines the number of top-most important features to consider before aggregation of excitation wavelength importances
        abs_nm          (int)
                        default=720
                        determines which wvelength should be used for optical density measurements. These measurements are always performed at the start & end of the measurement
        nm_interval     (int)
                        default=15
                        determines the intervals in nano-meter in step-size of the emission scan. Lower numbers provide more measurements but scanning takes longer
        gain            (int)
                        default=100
                        the gain (amplicification) of the measured response; 100 is the default and we never changed it
        zposition       (int)
                        default=20000
                        zposition, i.e. the distance of the detector from ther 96-well plate during emission measurement. 20000 is the default and we never changed it
        lagtime         (int)
                        default=0
                        time to pass between excitation and fluorescent measurement. I suppose this is important for FRAT-FlIM experiments, so we always set it to 0
        integrationtime (int)
                        default=20
                        Length of the detection of the emission response. Tecan's default is 20ms and we never changed that
        numflashes      (int)
                        default=3
                        number of repeated measurements of the same well using the same ex-em combination. Higher numbers give more accurate values but increase the time of the measurement. Low numbers DO NOT severly impact accuracy
        settletime      (int)
                        default=20
                        time in Milli-seconds (ms) the tecan waits after moving the plate to let cells settle. Tecan's default is 20ms and we never changed that
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

