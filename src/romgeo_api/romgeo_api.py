import romgeo as rg
import numpy as np
import numba as nb
#import math
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import simplejson as json
import ezdxf

import shutil
import uuid
import zipfile
import psutil
import secrets

#import romgeo.cuda
#import numba.cuda

from config import DEF_MULTILIST, INFO_TEXT, PRJ_CONTENT, TMP_ROOT, ZBOX_RO_ETRS, ZBOX_RO_ST70, SHP_PRJ_CONTENT
from extras import _is_inside_bounds, dd4_or_dms4, dd_or_dms, _parse_line_etrs

from fastapi import FastAPI, Query, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response


max_grid = None
grids_dir = os.path.join(os.path.dirname(__file__), "grids")
if os.path.isdir(grids_dir):
    grid_files = [f for f in os.listdir(grids_dir) if f.lower().endswith(".spg")]
    if grid_files:
        max_grid = sorted(grid_files)[-1]
LATEST_GRID = max_grid


app = FastAPI(
    root_path="/api/v1",
    title="RomGEO API",

    swagger_ui_parameters={"defaultModelsExpandDepth": -1,
                           "tryItOutEnabled":True}
)



def _do_transform(N, E, H, st70_Y, st70_X, st_H, grid="latest"):

    if grid == "latest":
        grid = LATEST_GRID

    t = rg.transformations.Transform(filename=f"{os.path.dirname(__file__)}/grids/{grid}")

    # print(N.dtype, N.shape, N.flags['C_CONTIGUOUS'])
    # print(E.dtype, E.shape, E.flags['C_CONTIGUOUS'])
    # print(H.dtype, H.shape, H.flags['C_CONTIGUOUS'])
    # print(st70_Y.dtype, st70_Y.shape, st70_Y.flags['C_CONTIGUOUS'])
    # print(st70_X.dtype, st70_X.shape, st70_X.flags['C_CONTIGUOUS'])
    # print(st_H.dtype, st_H.shape, st_H.flags['C_CONTIGUOUS'])

    #t.grid_shifts['grid'] = t.grid_shifts['grid'].astype(np.float64, copy=False)    
    #t.geoid_heights['grid'] = np.ascontiguousarray(t.geoid_heights['grid'], dtype=np.float32)

    t.etrs_to_st70(N,E,H, st70_Y,st70_X,st_H)


    return t



@app.get("/health")
def health_check():
    load = psutil.cpu_percent(interval=1)  # Get CPU load
    if load > 80:  # Overload threshold
        return Response(status_code=503)  # Make Traefik failover
    return {"status": "ok"}

@app.get("/transformText/{text}" )
def Convert_Text(text:str,
                 grid:str    =Query("latest",description="RO Grid version number, latest for latest available grid"),
                 srs:str     =Query("4326",  description="Source EPSG Code, only EPSG:4326 (ETRS89) is currently supported"),
                 crs:str     =Query("3844",  description="Destination EPSG Code, only EPSG:3844 (Stere70) is currently supported"),
                 astext:bool =Query(False,   description="Output as text only")):


    N,E,H, pct, comment = _parse_line_etrs(text)

    N = np.array([N],    dtype=np.float64) #45
    E = np.array([E],    dtype=np.float64) #25
    H = np.array([H],    dtype=np.float64) #-36

    st70_Y = np.full_like(E, 0.0)
    st70_X = np.full_like(N, 0.0)
    st_H   = np.full_like(H, 0.0)

    t = _do_transform(N, E, H, st70_Y, st70_X, st_H, grid)

    pct = 'noname' if pct == '' else pct

    ret = f"{pct}, {N[0]:.6f}, {E[0]:.6f}, {H[0]:.6f}, {st70_X[0]:.3f}, {st70_Y[0]:.3f}, {st_H[0]:.4f}"

    t1 = _is_inside_bounds(N[0],E[0],"etrs")
    t2 = _is_inside_bounds(st70_Y[0],st70_X[0],"st70")
    t3 = (ZBOX_RO_ETRS[0] <= H[0] < ZBOX_RO_ETRS[1])
    t4 = (ZBOX_RO_ST70[0] <= st_H[0] < ZBOX_RO_ST70[1])

    print([t1, t2, t3, t4])

    if not all([t1, t2, t3, t4]):
        print(ret)
        ret = "Input error or Out of bounds."

    if astext:
        print(ret)
        return f"{ret}"
    else:
        return {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs}

@app.get("/transformCoord/{lat}/{lon}/{he}/")
def Convert_LatLon(lat:str, 
                   lon:str, 
                   he:str , 
                   grid:str=Query("latest",description="RO Grid version number, latest for latest available grid"),  
                   srs:str =Query("4326",  description="Source EPSG Code, only EPSG:4326 (ETRS89) is currently supported"),  
                   crs:str =Query("3844",  description="Destination EPSG Code, only EPSG:3844 (Stere70) is currently supported"),
                   astext:bool =Query(False,  description="Output as text only")):

    lat = dd_or_dms(lat)
    lon = dd_or_dms(lon)
    he =  float(he)

    N = np.array([lat],    dtype=float) #45
    E = np.array([lon],    dtype=float) #25
    H = np.array([he],     dtype=float) #-36

    st70_Y = np.full_like(E, 0.0)
    st70_X = np.full_like(N, 0.0)
    st_H   = np.full_like(H, 0.0)

    t = _do_transform(N, E, H, st70_Y, st70_X, st_H)
    
    ret = f"{N[0]:.6f}, {E[0]:.6f}, {H[0]:.6f}, {st70_X[0]:.3f}, {st70_Y[0]:.3f}, {st_H[0]:.4f}"

    t1 = _is_inside_bounds(N[0],E[0],"etrs")
    t2 = _is_inside_bounds(st70_Y[0],st70_X[0],"st70")
    t3 = (ZBOX_RO_ETRS[0] <= H[0] < ZBOX_RO_ETRS[1])
    t4 = (ZBOX_RO_ST70[0] <= st_H[0] < ZBOX_RO_ST70[1])

    print([t1, t2, t3, t4])

    if not all([t1, t2, t3, t4]):
        print(ret)
        ret = "Input error or Out of bounds."

    if astext:
        print(ret)
        return f"{ret}"
    else:
        print( {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs})
        return {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs}

@app.get("/transformMultiText/")
def Convert_MultiText(multiText:list[str]=Query(DEF_MULTILIST, description="list of texts to convert, see /transformText/ for formatting",), 
                      grid:str =Query("latest",description="RO Grid version number, latest for latest available grid"),
                      srs:str  =Query("4326",  description="Source EPSG Code, only EPSG:4326 (ETRS89) is currently supported"),
                      crs:str  =Query("3844",  description="Destination EPSG Code, only EPSG:3844 (Stere70) is currently supported")):

    dms4table = [dd4_or_dms4(line) for line in multiText]

    print(f"dms4table= {dms4table}")

    N,E,H,pct = zip(*dms4table)

    N = np.array(N,    dtype=float) #45
    E = np.array(E,    dtype=float) #25
    H = np.array(H,    dtype=float) #-36

    st70_Y = np.full_like(E, 0.0)
    st70_X = np.full_like(N, 0.0)
    st_H   = np.full_like(H, 0.0)

    t = _do_transform(N, E, H, st70_Y, st70_X, st_H)

    ret = list(zip(pct, N,E,H, st70_X,st70_Y, st_H))

    import simplejson as json
    ret = json.loads(json.dumps(ret, ignore_nan=True))

    print( {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs})
    return {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs}

@app.post("/transformMultiText/")
def Convert_MultiText(multiText:list[str]=DEF_MULTILIST, 
                      grid:str =Query("latest",description="RO Grid version number, latest for latest available grid"),
                      srs:str  =Query("4326",  description="Source EPSG Code, only EPSG:4326 (ETRS89) is currently supported"),
                      crs:str  =Query("3844",  description="Destination EPSG Code, only EPSG:3844 (Stere70) is currently supported")):

    dms4table = [dd4_or_dms4(line) for line in multiText]

    print(f"dms4table= {dms4table}")

    N,E,H,pct = zip(*dms4table)

    N = np.array(N,    dtype=float) #45
    E = np.array(E,    dtype=float) #25
    H = np.array(H,    dtype=float) #-36

    st70_Y = np.full_like(E, 0.0)
    st70_X = np.full_like(N, 0.0)
    st_H   = np.full_like(H, 0.0)

    t = _do_transform(N, E, H, st70_Y, st70_X, st_H)

    ret = list(zip(pct, N,E,H, st70_X,st70_Y, st_H))

    import simplejson as json
    ret = json.loads(json.dumps(ret, ignore_nan=True))

    print( {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs})
    return {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs}

@app.post("/transformMultiTextToShapefile/")
def convert_multitext_to_shapefile(multiText: list[str] = DEF_MULTILIST,
                                   grid: str = Query("latest", description="RO Grid version number, latest for latest available grid"),
                                   srs: str = Query("4326", description="Source EPSG Code, only EPSG:4326 (ETRS89) is currently supported"),
                                   crs: str = Query("3844", description="Destination EPSG Code, only EPSG:3844 (Stereo70) is currently supported"),
                                   swap_xy: bool = Query(False, description="Swap X and Y coordinates if needed")):
    """
    Convert coordinates and output a 3D Shapefile, filtering out invalid points.
    Returns the Shapefile as a downloadable ZIP file.
    """

    # Ensure temp root exists
    os.makedirs(TMP_ROOT, exist_ok=True)

    # Generate a random folder name
    temp_folder = os.path.join(TMP_ROOT, str(uuid.uuid4()))
    os.makedirs(temp_folder, exist_ok=True)

    # Define the Shapefile name based on the folder name
    shapefile_name = f"{secrets.token_hex(5)}_stereo70"
    shapefile_path = os.path.join(temp_folder, shapefile_name + ".shp")
    prj_path = os.path.join(temp_folder, shapefile_name + ".prj")

    # Perform coordinate transformation
    dms4table = [dd4_or_dms4(line) for line in multiText]

    N, E, H, pct = zip(*dms4table)

    N = np.array(N, dtype=float)  # Latitude (ETRS89)
    E = np.array(E, dtype=float)  # Longitude (ETRS89)
    H = np.array(H, dtype=float)  # Height (Ellipsoidal)

    st70_Y = np.full_like(E, 0.0)
    st70_X = np.full_like(N, 0.0)
    st_H   = np.full_like(H, 0.0)

    _do_transform(N, E, H, st70_Y, st70_X, st_H)

    # Prepare transformed data for GeoDataFrame
    columns = ["Name", "Latitude", "Longitude", "Height_Ellipsoidal", "st70_X", "st70_Y", "H_mn"]
    transformed_data = list(zip(pct, N, E, H, st70_Y, st70_X, st_H))
    df = pd.DataFrame(transformed_data, columns=columns)

    # Filter out invalid points (using `is_inside_bounds` + H_mn range check)
    df = df[df.apply(lambda row: _is_inside_bounds(row["st70_X"], row["st70_Y"], "st70") and 
                                ZBOX_RO_ST70[0] <= row["H_mn"] <= ZBOX_RO_ST70[1], axis=1)]

    if df.empty:
        shutil.rmtree(temp_folder, ignore_errors=True)
        return {"error": "No valid points found after filtering. No shapefile was created."}

    # Create geometry (full precision for GIS processing)
    if swap_xy:
        df["geometry"] = df.apply(lambda row: Point(row["st70_Y"], row["st70_X"], row["H_mn"]), axis=1)
        print("Swapping st70_X and st70_Y (Using Y as East, X as North).")
    else:
        df["geometry"] = df.apply(lambda row: Point(row["st70_X"], row["st70_Y"], row["H_mn"]), axis=1)
        print("Using standard EPSG:3844 coordinate order (X = East, Y = North).")

    # Convert to GeoDataFrame with EPSG:3844
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:3844")

    # Round st70_X, st70_Y, and H_mn to 3 decimals for table attributes (while keeping geometry at full precision)
    gdf["st70_X"] = gdf["st70_X"].round(3)
    gdf["st70_Y"] = gdf["st70_Y"].round(3)
    gdf["H_mn"] = gdf["H_mn"].round(3)

    # Save Shapefile
    gdf.to_file(shapefile_path, driver="ESRI Shapefile")

    print(f"3D Shapefile saved to: {shapefile_path}")

    # Save .prj file with vertical CRS
    with open(prj_path, "w") as prj_file:
        prj_file.write(SHP_PRJ_CONTENT)

    # Create ZIP archive in the TMP_ROOT (outside temp folder)
    zip_filename = os.path.join(TMP_ROOT, shapefile_name + ".zip")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, file)
            zipf.write(file_path, arcname=file)

    print(f"Shapefile zipped to: {zip_filename}")

    # Delete temp folder (only keeping the ZIP file)
    shutil.rmtree(temp_folder, ignore_errors=True)

    # Return ZIP file as response
    return FileResponse(zip_filename, media_type="application/zip", filename=os.path.basename(zip_filename))

@app.post("/transformMultiTextToDxffile/")
def convert_multitext_to_dxf(multiText: list[str] = DEF_MULTILIST,
                             grid: str = Query("latest", description="RO Grid version number, latest for latest available grid"),
                             srs: str = Query("4326", description="Source EPSG Code, only EPSG:4326 (ETRS89) is currently supported"),
                             crs: str = Query("3844", description="Destination EPSG Code, only EPSG:3844 (Stereo70) is currently supported"),
                             swap_xy: bool = Query(True, description="Swap X and Y coordinates if needed")):
    """
    Convert coordinates and output a DXF file, ensuring it is in EPSG:3844.
    Returns the DXF file as a downloadable ZIP file, along with .prj and .txt info files.
    """

    os.makedirs(TMP_ROOT, exist_ok=True)

    temp_folder = os.path.join(TMP_ROOT, str(uuid.uuid4()))
    os.makedirs(temp_folder, exist_ok=True)

    base_filename = f"{secrets.token_hex(5)}_stereo70"
    dxf_path = os.path.join(temp_folder, base_filename + ".dxf")
    prj_path = os.path.join(temp_folder, base_filename + ".prj")
    info_txt_path = os.path.join(temp_folder, "info.txt")

    dms4table = [dd4_or_dms4(line) for line in multiText]

    N, E, H, pct = zip(*dms4table)

    N = np.array(N, dtype=float)
    E = np.array(E, dtype=float)
    H = np.array(H, dtype=float)

    st70_Y = np.full_like(E, 0.0)
    st70_X = np.full_like(N, 0.0)
    st_H   = np.full_like(H, 0.0)

    _do_transform(N, E, H, st70_Y, st70_X, st_H)

    columns = ["Name", "Latitude", "Longitude", "Height_Ellipsoidal", "st70_X", "st70_Y", "H_mn"]
    transformed_data = list(zip(pct, N, E, H, st70_Y, st70_X, st_H))
    df = pd.DataFrame(transformed_data, columns=columns)

    df = df[df.apply(lambda row: _is_inside_bounds(row["st70_X"], row["st70_Y"], "st70") and 
                                ZBOX_RO_ST70[0] <= row["H_mn"] <= ZBOX_RO_ST70[1], axis=1)]

    if df.empty:
        shutil.rmtree(temp_folder, ignore_errors=True)
        return {"error": "No valid points found after filtering. No DXF file was created."}

    df["Name"] = df["Name"].fillna(df.index.to_series().apply(lambda x: f"Point {x+1}"))

    doc = ezdxf.new()
    msp = doc.modelspace()

    for idx, row in df.iterrows():
        x, y, h = (row["st70_X"], row["st70_Y"], row["H_mn"]) if swap_xy else (row["st70_Y"], row["st70_X"], row["H_mn"])
        label = row["Name"] if row["Name"] else f"Point {idx+1}"
        msp.add_point((x, y, h), dxfattribs={"layer": "Stereo70_EPSG3844"})
        msp.add_text(label, dxfattribs={"insert": (x + 5, y + 5, h), "layer": "Labels"})

    doc.saveas(dxf_path)

    with open(prj_path, "w") as prj_file:
        prj_file.write(PRJ_CONTENT)

    with open(info_txt_path, "w") as info_file:
        info_file.write(INFO_TEXT)

    zip_filename = os.path.join(TMP_ROOT, base_filename + ".zip")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(dxf_path, arcname=base_filename + ".dxf")
        zipf.write(prj_path, arcname=base_filename + ".prj")
        zipf.write(info_txt_path, arcname="info.txt")

    shutil.rmtree(temp_folder, ignore_errors=True)

    return FileResponse(zip_filename, media_type="application/zip", filename=os.path.basename(zip_filename))

app.mount("/", StaticFiles(directory=f"{os.path.dirname(__file__)}/static",html = True), name="static")

if __name__ == "__main__":

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    ## debugging purposes only, run with uvicorn
    # Example usage:
    # Convert_Text("45.123456, 25.123456, 36.123456", "latest", "4326", "3844", True)

    Convert_Text("45.123456, 25.123456, 36.123456", "latest", "4326", "3844", True)
    Convert_LatLon("45.123456", "25.123456", "36.123456", "latest", "4326", "3844", True)
    Convert_MultiText(["45.123456, 25.123456, 36.123456", "45.654321, 25.654321, 36.654321"], "latest", "4326", "3844")
    convert_multitext_to_shapefile(["45.123456, 25.123456, 36.123456", "45.654321, 25.654321, -36.654321"], "latest", "4326", "3844")
    convert_multitext_to_dxf(["45.123456, 25.123456, 36.123456", "45.654321, 25.654321, -36.654321"], "latest", "4326", "3844")