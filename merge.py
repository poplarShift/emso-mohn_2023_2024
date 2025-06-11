#!/usr/bin/env python3

import xarray as xr
import os
import numpy as np


def merge_datasets(datasets, data_var):
    # Determine the common time grid (e.g., union of all time points)
    all_times = np.concatenate([ds["time"].values for ds in datasets])
    common_time = np.arange(all_times.min(), all_times.max(), np.timedelta64(30, "m"))

    # Interpolate each dataset to the common time grid
    interpolated_datasets = [
        ds[[data_var, "height_above_sea_floor_nominal"]].interp(time=common_time)
        for ds in datasets
        if data_var in ds
    ]

    # Align all datasets on the common time grid
    aligned_datasets = xr.align(*interpolated_datasets, join="outer")

    # Merge all aligned datasets into one
    merged_data = xr.concat(
        aligned_datasets, dim="height_above_sea_floor_nominal"
    ).sortby("height_above_sea_floor_nominal")[data_var]

    return merged_data


if __name__ == "__main__":
    # Define the directory containing the NetCDF files
    directory = "netcdf/"

    # List all NetCDF files in the directory
    netcdf_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".nc")
    ]

    # Load the "temp" variable from all NetCDF files
    datasets = [xr.open_dataset(f) for f in netcdf_files]

    

    for data_var in [
        "temp",
        "turbidity",
        "salinity",
    ]:
        da = merge_datasets(datasets, data_var)
        import matplotlib.pyplot as plt

        da.plot.line(x="time", hue="height_above_sea_floor_nominal")
        plt.yscale('log')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
