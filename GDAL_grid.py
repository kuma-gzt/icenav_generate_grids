# run with
# python3 GDAL_grid.py --config config.json

import os
import time
import argparse
import json
import csv
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    return cfg


class GDAL_Processing:
    def __init__(self, cfg):
        self.cfg = cfg
        self.centroids = cfg["centroids"]
        self.icedata_dir = cfg["icedata_dir"]
        self.output_dir = cfg["output_dir"]

    def process(self):
        start = time.time()

        self.process_geotiffs()

        end = time.time()
        print(f"Elapsed time: {end - start:.2f} seconds")

    def process_geotiffs(self):
        centroids = self.get_centroids()
        arr_c = np.array(centroids)
        x, y = arr_c.T

        lonlat = self.get_lonlat()
        arr_l = np.array(lonlat, dtype=float)
        lon, lat = arr_l.T

        grouped = self.group_files()

        for lst in grouped:
            output_csv = os.path.join(
                self.output_dir, os.path.basename(lst[0])[3:11] + ".csv"
            )

            fast_ = []
            fyi_ = []
            myi_ = []
            tc_max_ = []
            tc_mid_ = []
            tc_min_ = []
            thi_ = []

            pix_coords = self.get_pixelcoord(lst[0])

            for item in lst:
                print(item)
                ds = gdal.Open(item, gdal.GA_ReadOnly)
                if ds is None:
                    raise RuntimeError(f"Could not open raster: {item}")

                band = ds.GetRasterBand(1)
                values = []

                for x_px, y_px in pix_coords:
                    values.append(band.ReadAsArray(x_px, y_px, 1, 1)[0, 0])

                if "fast" in item:
                    fast_ = values
                elif "fyi" in item:
                    fyi_ = values
                elif "myi" in item:
                    myi_ = values
                elif "tc_max" in item:
                    tc_max_ = values
                elif "tc_mid" in item:
                    tc_mid_ = values
                elif "tc_min" in item:
                    tc_min_ = values
                elif "thi" in item:
                    thi_ = values

                ds = None

            fast = np.array(fast_)
            fyi = np.array(fyi_)
            myi = np.array(myi_)
            tc_max = np.array(tc_max_)
            tc_mid = np.array(tc_mid_)
            tc_min = np.array(tc_min_)
            thi = np.array(thi_)

            arr_2d = np.column_stack(
                [x, y, lon, lat, fast, fyi, myi, tc_max, tc_mid, tc_min, thi]
            )
            np.savetxt(
                output_csv,
                arr_2d,
                fmt="%d",
                delimiter=",",
                header="x, y, lon_e6, lat_e6, fast, fyi, myi, tc_max, tc_mid, tc_min, thi",
                comments="",
            )

    def get_centroids(self):
        centroids = []
        with open(self.centroids, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = int(row["X"])
                y = int(row["Y"])
                centroids.append((x, y))
        return centroids

    def get_lonlat(self):
        lonlat = []
        with open(self.centroids, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lon = int(round(float(row["lon"]), 6) * 1000000)
                lat = int(round(float(row["lat"]), 6) * 1000000)
                lonlat.append((lon, lat))

        # returns lon and lat as integers so the final product
        # uses only integers. To get the actual values divide
        # by 1000000
        return lonlat

    def group_files(self):
        files = sorted([f for f in os.listdir(self.icedata_dir) if f.endswith(".tif")])
        file_list = []
        file_names = []

        for file in files:
            name = file[3:11]
            if name not in file_names:
                file_names.append(name)

        for item in file_names:
            tmp = []
            for file in files:
                if item in file:
                    tmp.append(os.path.join(self.icedata_dir, file))
            file_list.append(tmp)

        return file_list

    def get_pixelcoord(self, band):
        centroids = self.get_centroids()

        ds = gdal.Open(band, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Could not open raster: {band}")

        # Get transform and inverse transform
        gt = ds.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(gt)

        lst = []

        # Convert world coordinates to pixel coordinates
        for x, y in centroids:
            px, py = gdal.ApplyGeoTransform(inv_gt, x, y)
            px, py = int(px), int(py)

            # Check if within raster bounds
            if px >= ds.RasterXSize or py >= ds.RasterYSize:
                print(
                    f"Skipping point {px},{py}: outside raster extent ({ds.RasterXSize},{ds.RasterYSize})"
                )
                continue

            lst.append([px, py])

        ds = None
        return lst


def main():
    cfg = load_config()
    gdal = GDAL_Processing(cfg)
    gdal.process()


if __name__ == "__main__":
    main()
