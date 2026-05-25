import json
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LON = 50.3167
LAT = 29.2500
BUFFER_M = 3000

CRS = 'EPSG:32639'
SCALE = 10

DATE_WINDOWS = {
    'pre_0506': ('2026-05-01', '2026-05-05'),
    'around_0506': ('2026-05-06', '2026-05-08'),
    'around_0525': ('2026-05-23', '2026-05-27'),
}

OUTPUT_DIR = Path('outputs/sentinel1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def initialize_ee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def get_water_mask():
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence')
    return gsw.gt(0)


def to_linear(db_img):
    return ee.Image.constant(10).pow(db_img.divide(10))


def compute_oil_mask(start, end, aoi, water_mask):
    collection = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.eq('resolution_meters', 10))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    )

    image = collection.median()

    vv = image.select('VV').focal_mean(30, 'circle', 'meters')
    vh = image.select('VH').focal_mean(30, 'circle', 'meters')

    vv_lin = to_linear(vv)
    vh_lin = to_linear(vh)
    ratio = vv_lin.divide(vh_lin).rename('VVVH_ratio')

    masked = vv.addBands(vh).addBands(ratio).updateMask(water_mask)
    stats = masked.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', True),
        geometry=aoi,
        scale=SCALE,
        maxPixels=1e9,
    )

    vv_mean = ee.Number(stats.get('VV_mean'))
    vv_std = ee.Number(stats.get('VV_stdDev'))
    vh_mean = ee.Number(stats.get('VH_mean'))
    vh_std = ee.Number(stats.get('VH_stdDev'))
    ratio_mean = ee.Number(stats.get('VVVH_ratio_mean'))
    ratio_std = ee.Number(stats.get('VVVH_ratio_stdDev'))

    vv_thresh = vv_mean.subtract(vv_std.multiply(1.5))
    vh_thresh = vh_mean.subtract(vh_std.multiply(1.5))
    ratio_thresh = ratio_mean.subtract(ratio_std.multiply(1.0))

    oil = (
        vv.lt(vv_thresh)
        .And(vh.lt(vh_thresh))
        .And(ratio.lt(ratio_thresh))
    )

    oil = oil.updateMask(water_mask).selfMask().rename('oil_mask')
    pixel_count = oil.connectedPixelCount(100, True)
    oil = oil.updateMask(pixel_count.gte(8))

    return vv.rename('VV'), vh.rename('VH'), ratio.rename('VVVH_ratio'), oil


def export_image(image, filename, region):
    geemap.ee_export_image(
        image,
        filename=str(filename),
        scale=SCALE,
        crs=CRS,
        region=region,
        file_per_band=False,
    )


def add_scale_bar(ax, transform, length_km=1.0, location=(0.08, 0.07)):
    pixel_size = abs(transform.a)
    length_m = length_km * 1000
    length_px = length_m / pixel_size

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_start = x_min + (x_max - x_min) * location[0]
    y_start = y_min + (y_max - y_min) * location[1]
    x_end = x_start + length_px * pixel_size

    ax.plot([x_start, x_end], [y_start, y_start], color='black', linewidth=3)
    ax.plot([x_start, x_start], [y_start, y_start + length_px * 0.02 * pixel_size], color='black', linewidth=3)
    ax.plot([x_end, x_end], [y_start, y_start + length_px * 0.02 * pixel_size], color='black', linewidth=3)
    ax.text((x_start + x_end) / 2, y_start + length_px * 0.04 * pixel_size, f'{length_km:.0f} km',
            ha='center', va='bottom', fontsize=10, color='black')


def add_north_arrow(ax, location=(0.92, 0.1), size=0.07):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x = x_min + (x_max - x_min) * location[0]
    y = y_min + (y_max - y_min) * location[1]

    ax.annotate(
        'N',
        xy=(x, y + (y_max - y_min) * size),
        xytext=(x, y),
        arrowprops=dict(facecolor='black', width=3, headwidth=10),
        ha='center', va='center', fontsize=12, color='black',
    )


def render_map(base_path, mask_path, output_png, output_tif, title):
    with rasterio.open(base_path) as src:
        base = src.read(1).astype('float32')
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        extent = rasterio.plot.plotting_extent(src)
        nodata = src.nodata

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    if nodata is not None:
        base[base == nodata] = np.nan
    vmin, vmax = np.nanpercentile(base, [2, 98])

    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(base, cmap='gray', vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')

    oil = np.ma.masked_where(mask == 0, mask)
    ax.imshow(oil, cmap='Reds', alpha=0.6, extent=extent, interpolation='nearest')

    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    legend_elements = [Patch(facecolor='red', edgecolor='red', label='Oil spill (detected)')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True)

    add_scale_bar(ax, transform, length_km=1.0)
    add_north_arrow(ax)

    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)

    image = plt.imread(output_png)
    if image.shape[2] == 4:
        image = (image[:, :, :3] * 255).astype('uint8')
    else:
        image = (image * 255).astype('uint8')

    with rasterio.open(
        output_tif,
        'w',
        driver='GTiff',
        height=image.shape[0],
        width=image.shape[1],
        count=3,
        dtype='uint8',
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(image[:, :, 0], 1)
        dst.write(image[:, :, 1], 2)
        dst.write(image[:, :, 2], 3)


def main():
    initialize_ee()

    aoi = ee.Geometry.Point([LON, LAT]).buffer(BUFFER_M)
    region = aoi.bounds()
    water_mask = get_water_mask()

    stats_out = {}

    for label, (start, end) in DATE_WINDOWS.items():
        vv, vh, ratio, oil = compute_oil_mask(start, end, aoi, water_mask)

        area = oil.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=SCALE,
            maxPixels=1e9,
        ).get('oil_mask')

        area_value = ee.Algorithms.If(
            area,
            ee.Number(area).divide(1e6),
            0,
        ).getInfo()

        stats_out[label] = {
            'start_date': start,
            'end_date': end,
            'area_km2': area_value,
        }

        base_path = OUTPUT_DIR / f'{label}_vv.tif'
        mask_path = OUTPUT_DIR / f'{label}_oil_mask.tif'
        map_png = OUTPUT_DIR / f'{label}_map.png'
        map_tif = OUTPUT_DIR / f'{label}_map.tif'

        export_image(vv, base_path, region)
        export_image(oil, mask_path, region)

        title = f'Sentinel-1 Oil Spill Detection ({start} to {end})'
        render_map(base_path, mask_path, map_png, map_tif, title)

    stats_file = OUTPUT_DIR / 'oil_stats.json'
    stats_file.write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
