#!/usr/bin/env python

from pathlib import Path
import xarray as xr
import os
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import hvplot.pandas
import matplotlib.pyplot as plt


def add_path_end_dummy(df):
    df_path_end_dummy = pd.DataFrame({"time": [df.time.max()], "temp": np.nan})
    return pd.concat((df, df_path_end_dummy))


def plot_temp(datasets):
    df = pd.concat(
        [
            add_path_end_dummy(ds["temp"].to_dataframe().reset_index()).assign(
                fname=fname
            )
            for ds, fname in zip(datasets, netcdf_files)
            if "temp" in ds.data_vars
        ]
    )
    df["fname"] = df["fname"].astype("category")
    df["itime"] = df.time.astype("int64")
    import matplotlib.pyplot as plt
    import holoviews as hv

    hv.extension("matplotlib")
    l = df.hvplot(
        x="time",
        y="temp",
        by="fname",
        datashade=True,
        dynamic=False,
        cmap="glasbey",
        legend="top_left",
        width=800,
        height=400,
        xlabel="Time",
        ylabel="Temperature (deg C)",
        grid=True,
    )
    fig = hv.render(l)
    fig.savefig("figures/temperature.png")


def plot_turbidity(datasets):
    df = pd.concat(
        [
            add_path_end_dummy(ds["turbidity"].to_dataframe().reset_index()).assign(
                fname=fname
            )
            for ds, fname in zip(datasets, netcdf_files)
            if "turbidity" in ds.data_vars
        ]
    )
    df["fname"] = df["fname"].astype("category")
    df["itime"] = df.time.astype("int64")
    import matplotlib.pyplot as plt
    import holoviews as hv

    hv.extension("matplotlib")
    df_subset = df.loc[df.turbidity < 3]
    for name, df_selec in [("full", df), ("subset", df_subset)]:
        l = df_selec.hvplot(
            x="time",
            y="turbidity",
            by="fname",
            legend="top_right",
            width=1200,
            height=500,
            xlabel="Time",
            ylabel="Turbidity (NTU)",
            grid=True,
        )
        fig = hv.render(l)
        fig.savefig(f"figures/turbidity-{name}.png")


def calc_salinity(ds):
    # Convert conductivity to Practical Salinity
    ds["SP"] = gsw.SP_from_C(10 * ds["conductivity"], ds["temp"], ds["pressure"])

    # Calculate Absolute Salinity
    ds["salinity"] = gsw.SA_from_SP(ds["SP"], ds["pressure"], lon=0, lat=70)
    return ds


def plot_salinity(datasets):

    dataset_selection = [
        calc_salinity(ds) for ds in datasets if "conductivity" in ds.data_vars
    ]

    df = pd.concat(
        [
            add_path_end_dummy(ds["salinity"].to_dataframe().reset_index()).assign(
                fname=fname
            )
            for ds, fname in zip(datasets, netcdf_files)
            if "salinity" in ds.data_vars
        ]
    )
    df["fname"] = df["fname"].astype("category")
    df["itime"] = df.time.astype("int64")
    import holoviews as hv

    hv.extension("matplotlib")
    l = df.hvplot(
        x="time",
        y="salinity",
        by="fname",
        datashade=True,
        dynamic=False,
        cmap="glasbey",
        legend="top_left",
        width=800,
        height=400,
        xlabel="Time",
        ylabel="Absolute salinity (g kg-1)",
        grid=True,
    )
    fig = hv.render(l)
    fig.savefig("figures/salinity.png")


def plot_adcp(fname: str):
    ds = xr.load_dataset(fname)
    crit_cor = 80
    crit_amp = 1
    good = (
        (ds["correlation1"] > crit_cor)
        & (ds["correlation2"] > crit_cor)
        & (ds["correlation3"] > crit_cor)
        & (ds["correlation4"] > crit_cor)
        & (ds["amplitude1"] > crit_amp)
        & (ds["amplitude2"] > crit_amp)
        & (ds["amplitude3"] > crit_amp)
        & (ds["amplitude4"] > crit_amp)
    )
    fig, ax = plt.subplots()
    ds["sea_water_speed"].where(good).sel(celldist=good.any("time")).plot.line(
        x="time", ax=ax
    )
    fig.savefig("figures/" + Path(fname).with_suffix(".png").name)


if __name__ == "__main__":
    directory = "netcdf/"

    # List all NetCDF files in the directory
    netcdf_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".nc")
    ]

    # Load the "temp" variable from all NetCDF files
    datasets = [xr.open_dataset(f) for f in netcdf_files]

    plot_temp(datasets)
    plot_turbidity(datasets)
    plot_salinity(datasets)
    plot_adcp("netcdf/ADCP_400_upward.nc")
    plot_adcp("netcdf/ADCP_400_downward.nc")
