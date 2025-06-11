#!/bin/env python3
from pathlib import Path
from collections import namedtuple
import gsw
import pandas as pd, xarray as xr
import numpy as np

datadir = Path("EMSO-Mohn_extracted_data_2023_2024/")
outdir = Path("netcdf")


def convert_aadi():
    df = pd.read_csv(datadir / "Data_AADI_PressureGauge_D_3.csv", parse_dates=["Time"])
    z = 0  # nominal depth
    df = df.rename(
        columns={
            "Time": "time",
            "Pressure(kPa)": "pressure",
            "Temperature(degC)": "temp",
        }
    ).set_index("time")
    df["pressure"] /= 10  # convert kPa to dbar
    # pressure offset, from data on 2 July 2024 (recovery):
    # 02-Jul-2024 19:00:00,31155.934,-0.586
    # 02-Jul-2024 19:30:00,31156.928,-0.585
    # 02-Jul-2024 20:00:00,22492.453,-0.64
    # 02-Jul-2024 20:30:00,11777.275,-0.338
    # 02-Jul-2024 21:00:00,1587.672,4.588
    # 02-Jul-2024 21:30:00,92.842,8.701
    # 02-Jul-2024 22:00:00,93.118,8.756
    # 02-Jul-2024 22:30:00,93.276,9.55
    # 02-Jul-2024 23:00:00,93.155,9.756
    # 02-Jul-2024 23:30:00,93.247,9.871
    df["pressure"] -= 9.32

    ds = df.to_xarray()
    add_depth(ds, z)
    add_metadata(ds)
    add_temp_metadata(ds)
    ds["pressure"].attrs["units"] = "dbar"
    ds["pressure"].attrs["standard_name"] = "sea_water_pressure_at_sea_floor"

    ds.attrs["keywords"] = (
        "EARTH SCIENCE>OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE,",
        "EARTH SCIENCE>OCEANS>OCEAN PRESSURE>WATER PRESSURE",
    )
    cut_mooring_in_water(ds).to_netcdf(outdir / f"AADI_{z}.nc")


def convert_adcp():
    for fname, z, direc in [
        ("data_miu_adcp_103395_downward.csv", 400, "downward"),
        ("data_miu_adcp_103405_upward.csv", 400, "upward"),
    ]:
        df = pd.read_csv(
            datadir / fname,
            index_col=False,
            dtype={"Date  [MMDDYY]": str, "Time  [HHMMSS]  in  UTC  ": str},
        )
        df["time"] = pd.to_datetime(
            df["Date  [MMDDYY]"] + df["Time  [HHMMSS]  in  UTC  "],
            format="%m%d%y%H%M%S",
        )
        df["celldist"] = (
            df["Cell  number  (blanking  0.5  m  cell  size  4  m)"] - 1
        ) * 4 + 0.5

        rename_entry = namedtuple(
            "entry", "oldname newname units standard_name long_name"
        )
        # fmt: off
        rename_list = [
            rename_entry("heading", "heading", "degrees", None, None),
            rename_entry("Pitch  (deg)", "pitch", "degree", "platform_pitch", None),
            rename_entry("Roll  (deg)", "roll", "degree", "platform_roll", None),
            rename_entry("Pressure  (dBar)", "Pressure", "dbar", "sea_water_pressure", None),
            rename_entry("Temperature  (deg  C)", "Temperature", "degC", "sea_water_temperature", None),
            rename_entry("celldist", "celldist", "m", "", "Distance of cell from transducer"),
            rename_entry("Velocity  1  (m/s)  East", "u", "m s-1", "eastward_sea_water_velocity", None),
            rename_entry("Velocity  2  (m/s)  North", "v", "m s-1", "northward_sea_water_velocity", None),
            rename_entry("Velocity  3  (m/s)  Up1", "w1", "m s-1", "upward_sea_water_velocity", None),
            rename_entry("Velocity  4  (m/s)  Up2", "w2", "m s-1", "upward_sea_water_velocity", None),
            rename_entry("Current  speed  (m/s)", "sea_water_speed", "m s-1", "sea_water_speed", None),
            rename_entry("Direction  (deg)", "direction", "degree", "sea_water_velocity_to_direction", None),
            rename_entry("Amplitude  (Beam  1)", "amplitude1", "counts", None, "Backscatter amplitude beam 1"),
            rename_entry("Amplitude  (Beam  2)", "amplitude2", "counts", None, "Backscatter amplitude beam 2"),
            rename_entry("Amplitude  (Beam  3)", "amplitude3", "counts", None, "Backscatter amplitude beam 3"),
            rename_entry("Amplitude  (Beam  4)", "amplitude4", "counts", None, "Backscatter amplitude beam 4"),
            rename_entry("Correlation  (%)  (Beam1)", "correlation1", "%", None, "Correlation beam 1"),
            rename_entry("Correlation  (%)  (Beam2)", "correlation2", "%", None, "Correlation beam 2"),
            rename_entry("Correlation  (%)  (Beam3)", "correlation3", "%", None, "Correlation beam 3"),
            rename_entry("Correlation  (%)  (Beam4)", "correlation4", "%", None, "Correlation beam 4"),
        ]
        # fmt: on
        df = df.set_index(["time", "celldist"])[
            [e.oldname for e in rename_list if e.oldname != "celldist"]
        ].rename(columns={e.oldname: e.newname for e in rename_list})

        ds = df.to_xarray()
        ds["adcp_direction"] = direc
        ds["adcp_direction"].attrs["long_name"] = "Which direction the ADCP is facing"
        for e in rename_list:
            ds[e.newname].attrs["units"] = e.units
            if e.standard_name is not None:
                ds[e.newname].attrs["standard_name"] = e.standard_name
            if e.long_name is not None:
                ds[e.newname].attrs["long_name"] = e.long_name

        add_depth(ds, z)
        add_metadata(ds)
        ds.attrs["instrument"] = "500 kHz ADCP"
        ds.attrs["time_coverage_resolution"] = "P0Y000DT02H00M00S"
        ds.attrs["keywords"] = (
            "EARTH SCIENCE>OCEANS>OCEAN CIRCULATION>OCEAN CURRENTS",
        )
        # collapse celldist dimension in 1D data vars
        for data_var in ["heading", "pitch", "roll", "Pressure", "Temperature"]:
            ds[data_var] = ds[data_var].mean("celldist")

        ds["instrument_depth"] = (["time"], gsw.z_from_p(ds["Pressure"].values, 72.8))
        ds["depth"] = (  # depth downward positive
            ds["instrument_depth"] - ds["celldist"]
            if direc == "downward"
            else ds["instrument_depth"] + ds["celldist"]
        )

        # QC flag
        amplitudes_array = ds.filter_by_attrs(
            long_name=lambda s: "amplitude" in s if s else False
        ).to_array()
        any_amplitude_below_60 = (amplitudes_array < 60).any("variable")
        speed_above_2 = ds.sea_water_speed > 2
        ds["QC"] = speed_above_2.astype(int) + any_amplitude_below_60.astype(int) * 2
        ds["QC"].attrs["long_name"] = "Quality control flag"
        ds["QC"].attrs["flag_values"] = "0,1,2,3"
        ds["QC"].attrs[
            "flag_meanings"
        ] = "good, speed above 2 m/s, amplitude below 60 counts, speed above 2 m/s and amplitude below 60 counts"
        cut_mooring_in_water(ds).to_netcdf(outdir / f"ADCP_{z}_{direc}.nc")


def convert_ctd():
    for fname, z in [
        ("data_CTD_Lander.csv", 0),
        ("data_CTD_misense9357.csv", 450),
        ("data_CTD_misense9358.csv", 500),
        ("data_CTD_MIU.csv", 400),
    ]:
        df = pd.read_csv(datadir / fname)
        df = df.rename(
            columns={
                "Conductivity [S/m]": "conductivity",
                " Conductivity [S/m]": "conductivity",
                "Temperature [C]": "temp",
                " Temperature [C]": "temp",
                "Pressure [dbar]": "pressure",
                " Pressure [dbar]": "pressure",
                " day month year": "day month year",
                " HH:mm:ss": "HH:mm:ss",
            }
        )
        df["time"] = pd.to_datetime(
            df["day month year"] + " " + df["HH:mm:ss"],
            format="%d %b %Y %H:%M:%S",
        )
        df = df.set_index("time")[["temp", "conductivity", "pressure"]]
        ds = df.to_xarray()
        add_temp_metadata(ds)
        ds["pressure"].attrs["units"] = "dbar"
        ds["pressure"].attrs["standard_name"] = "sea_water_pressure"
        ds["conductivity"].attrs["units"] = "S m-1"
        ds["conductivity"].attrs["standard_name"] = "sea_water_electrical_conductivity"
        add_depth(ds, z)
        add_metadata(ds)
        ds.attrs["instrument"] = "Lander"
        ds.attrs["time_coverage_resolution"] = "P0Y000DT00H30M00S"
        ds.attrs["keywords"] = (
            "EARTH SCIENCE>OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE,",
            "EARTH SCIENCE>OCEANS>SALINITY/DENSITY>SALINITY",
        )
        cut_mooring_in_water(ds).to_netcdf(outdir / f"CTD_{z}.nc")


def convert_turbidity():
    for fname, z in [
        ("data_turbidity_Lander.csv", 0),
        ("data_turbidity_misense9355.csv", 350),
        ("data_turbidity_misense9357.csv", 450),
        ("data_turbidity_misense9358.csv", 500),
        ("data_turbidity_MIU.csv", 400),
    ]:
        df = pd.read_csv(datadir / fname)
        df["time"] = pd.to_datetime(df["day  month  year  "] + " " + df["HH:mm:ss.sss"])
        df = df.rename(
            columns={
                "Turbidity [FTU]": "turbidity",
                "Turbidity  [FTU]": "turbidity",
                "Temperature [Deg.C]": "temp",
                "Temperature  [Deg.C]": "temp",
            }
        )[["time", "temp", "turbidity"]]
        ds = df.set_index("time").to_xarray()
        add_temp_metadata(ds)
        add_turbidity_metadata(ds)
        add_depth(ds, z)
        add_metadata(ds)
        ds.attrs["instrument"] = "Lander"
        ds.attrs["time_coverage_resolution"] = "P0Y000DT00H30M00S"
        ds.attrs["keywords"] = (
            "EARTH SCIENCE>OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE,",
            "EARTH SCIENCE>OCEANS>OCEAN OPTICS>TURBIDITY",
        )
        cut_mooring_in_water(ds).to_netcdf(outdir / f"turbidity_{z}.nc")


def convert_sbe39():
    for fname, sno, z in [
        ("SBE39plus-IM09817_2024-07-29.csv", "IM09817", 20),
        ("SBE39plus-IM09818_2024-07-29.csv", "IM09818", 40),
        ("SBE39plus-IM09819_2024-07-29.csv", "IM09819", 60),
        ("SBE39plus-IM09820_2024-07-29.csv", "IM09820", 80),
        ("SBE39plus-IM09821_2024-07-29.csv", "IM09821", 100),
        ("SBE39plus-IM09822_2024-07-29.csv", "IM09822", 270),
        ("SBE39plus-IM09823_2024-07-29.csv", "IM09823", 290),
        ("SBE39plus-IM09824_2024-07-29.csv", "IM09824", 310),
        ("SBE39plus-IM09825_2024-07-29.csv", "IM09825", 330),
        ("SBE39plus-IM09826_2024-07-29.csv", "IM09826", 370),
        ("SBE39plus-IM09827_2024-07-29.csv", "IM09827", 390),
        ("SBE39plus-IM09828_2024-07-29.csv", "IM09828", 420),
        ("SBE39plus-IM09829_2024-07-29.csv", "IM09829", 440),
        ("SBE39plus-IM09830_2024-07-29.csv", "IM09830", 460),
        ("SBE39plus-IM09831_2024-07-29.csv", "IM09831", 480),
    ]:
        df = pd.read_csv(
            datadir / fname,
            parse_dates=["time"],
            date_format="%d-%b-%Y %H:%M:%S",
        ).set_index("time")
        ds = df.to_xarray()
        ds.attrs["instrument"] = "SBE39"
        ds.attrs["time_coverage_resolution"] = "P0Y000DT00H30M00S"
        ds.attrs["keywords"] = (
            "EARTH SCIENCE>OCEANS>OCEAN TEMPERATURE>WATER TEMPERATURE",
        )
        add_temp_metadata(ds)
        add_depth(ds, z)
        add_metadata(ds)
        cut_mooring_in_water(ds).to_netcdf(outdir / f"SBE39_{z}.nc")


def cut_mooring_in_water(ds):
    """These time stamps were determined from AADI pressure readings"""
    return ds.sel(
        time=slice(pd.Timestamp("2023-06-09T15:30"), pd.Timestamp("2024-07-02T19:30"))
    )


def add_temp_metadata(ds):
    ds["temp"].attrs["units"] = "degC"
    ds["temp"].attrs["standard_name"] = "sea_water_temperature"
    ds["temp"].attrs["scale"] = "ITS-90"


def add_turbidity_metadata(ds):
    ds["turbidity"].attrs["units"] = "1e-6"  # FTU=ppm
    ds["turbidity"].attrs["standard_name"] = "sea_water_turbidity"
    ds["turbidity"].attrs["long_name"] = "Sea water turbidity measured in FTU"


def add_depth(ds, height):
    ds["sea_floor_depth"] = 3051  # gsw.z_from_p(3106, 72.756)
    ds["sea_floor_depth"].attrs["standard_name"] = "sea_floor_depth_below_geoid"
    ds["sea_floor_depth"].attrs["units"] = "m"
    ds["height_above_sea_floor_nominal"] = height
    ds["height_above_sea_floor_nominal"].attrs[
        "long_name"
    ] = "Nominal height above sea floor determined from distance along mooring line"
    ds["height_above_sea_floor_nominal"].attrs[
        "standard_name"
    ] = "height_above_sea_floor"
    ds["height_above_sea_floor_nominal"].attrs["units"] = "m"


def add_metadata(ds):
    ds.attrs["title"] = ""  # TODO
    ds.attrs["summary"] = ()  # TODO

    ds.attrs["time_coverage_start"] = (
        pd.Timestamp(ds.time.min().values).tz_localize(tz="UTC").isoformat()
    )
    ds.attrs["time_coverage_end"] = (
        pd.Timestamp(ds.time.max().values).tz_localize(tz="UTC").isoformat()
    )
    ds.attrs["date_created"] = pd.Timestamp.now(tz="UTC").isoformat()
    ds.attrs["geospatial_lon_min"] = np.nan  # TODO
    ds.attrs["geospatial_lon_max"] = np.nan  # TODO
    ds.attrs["geospatial_lat_min"] = np.nan  # TODO
    ds.attrs["geospatial_lat_max"] = np.nan  # TODO
    ds.attrs["geospatial_vertical_min"] = np.nan  # TODO
    ds.attrs["geospatial_vertical_max"] = np.nan  # TODO
    ds.attrs["time_coverage_duration"] = pd.Timedelta(
        (ds.time.max() - ds.time.min()).values
    ).isoformat()
    ds.attrs["version"] = 1.0
    ds.attrs["citation"] = ""  # TODO
    ds.attrs["references"] = ""  # TODO
    ds.attrs["related_url"] = ""  # TODO
    ds.attrs["creator_name"] = "Achim Randelhoff"
    ds.attrs["creator_email"] = "ara@akvaplan.niva.no"
    ds.attrs["creator_institution"] = ""  # TODO
    ds.attrs["creator_url"] = ""
    ds.attrs["institution"] = "IUEM, NORCE, Akvaplan-niva AS"
    ds.attrs["program"] = ""
    ds.attrs["project"] = ""
    ds.attrs["acknowledgment"] = ()
    ds.attrs["contributor_name"] = "Thibaut Barreyre, Beatrice Tomasi, Achim Randelhoff"
    ds.attrs["contributor_role"] = (
        "TB, BT: planning, data collection, processing. AR: processing."
    )
    ds.attrs["featureType"] = "timeSeries"
    ds.attrs["platform"] = "Mooring"
    ds.attrs["source"] = "EMSO-Mohn mooring 2023-2024"
    ds.attrs["Conventions"] = "ACDD-1.3, CF-1.8"
    ds.attrs["license"] = "CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
    ds.attrs["processing_level"] = "Data manually reviewed"
    ds.attrs["data_set_language"] = "eng"
    ds.attrs["data_set_progress"] = "complete"
    ds.attrs["geospatial_bounds_crs"] = "EPSG:4326"
    ds.attrs["geospatial_lat_units"] = "degrees_north"
    ds.attrs["geospatial_lon_units"] = "degrees_east"
    ds.attrs["geospatial_vertical_positive"] = "down"
    ds.attrs["geospatial_vertical_units"] = "m"
    ds.attrs["operational_status"] = "Scientific"


if __name__ == "__main__":
    convert_aadi()
    convert_adcp()
    convert_ctd()
    convert_sbe39()
    convert_turbidity()
