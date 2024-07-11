import romgeo as rg
import romgeo.cuda
import numpy as np
import numba as nb
import numba.cuda
import math
import os

from .extras import is_inside_bounds, dd4_or_dms4, dd_or_dms

ZBOX_RO_ETRS = [-100, 2600]
ZBOX_RO_ST70 = [ -50, 2600]

DESC_TEXT_FORMAT = "Should contain: PointName(optional), lat, lon in any format (DD or DMS) and height as float"

DEF_MULTILIST = ["DEMO1 44°34\'31.54821\" 22°39\'02.48758\" 198.848",
                 "DEMO2 N44g34m31.54821s 22 39 02.48758 E 198.848",
                 "DEMO3 44.84821 22.48758 198.848m"]

from fastapi import FastAPI, Query, Depends
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="RomGEO API",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1,
                           "tryItOutEnabled":True}
)

@app.get("/transformText/{text}" )
def Convert_Text(text:str,
                 grid:str    =Query("latest",description="RO Grid version number, latest for latest available grid"),
                 srs:str     =Query("4326",  description="Source EPSG Code, only EPSG:4326 (ETRS89) is currently supported"),
                 crs:str     =Query("3844",  description="Destination EPSG Code, only EPSG:3844 (Stere70) is currently supported"),
                 astext:bool =Query(False,   description="Output as text only")):

    N,E,H, pct = dd4_or_dms4(text)

    N = np.array([N],    dtype=float) #45
    E = np.array([E],    dtype=float) #25
    H = np.array([H],    dtype=float) #-36

    X = np.full_like(N, 0.0)
    Y = np.full_like(E, 0.0)
    Z = np.full_like(H, 0.0)

    t = rg.transformations.Transform() #if grid in ['latest'] else rg.transformations.TransDatRO(filename = f"{grid}")    
    t.etrs_to_st70(N,E,H, X,Y,Z)
    
    pct = 'noname' if pct == '' else pct

    ret = f"{pct}, {N[0]:.6f}, {E[0]:.6f}, {H[0]:.6f}, {X[0]:.3f}, {Y[0]:.3f}, {Z[0]:.4f}"

    t1 = is_inside_bounds(N[0],E[0],"etrs")
    t2 = is_inside_bounds(X[0],Y[0],"st70")
    t3 = (ZBOX_RO_ETRS[0] <= H[0] < ZBOX_RO_ETRS[1])
    t4 = (ZBOX_RO_ST70[0] <= Z[0] < ZBOX_RO_ST70[1])

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

    X = np.full_like(N, 0.0)
    Y = np.full_like(E, 0.0)
    Z = np.full_like(H, 0.0)

    t = rg.transformations.Transform() #if grid in ['latest'] else rg.transformations.TransDatRO(filename = f"{grid}")    
    t.etrs_to_st70(N,E,H, X,Y,Z)
    
    ret = f"{N[0]:.6f}, {E[0]:.6f}, {H[0]:.6f}, {X[0]:.3f}, {Y[0]:.3f}, {Z[0]:.4f}"

    t1 = is_inside_bounds(N[0],E[0],"etrs")
    t2 = is_inside_bounds(X[0],Y[0],"st70")
    t3 = (ZBOX_RO_ETRS[0] <= H[0] < ZBOX_RO_ETRS[1])
    t4 = (ZBOX_RO_ST70[0] <= Z[0] < ZBOX_RO_ST70[1])

    print([t1, t2, t3, t4])

    if not all([t1, t2, t3, t4]):
        print(ret)
        ret = "Input error or Out of bounds."

    if astext:
        print(ret)
        return f"{ret}"
    else:
        return {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs}

@app.get("/transformMultiText/")
def Convert_MultiText(multiText:list[str]=Query(DEF_MULTILIST,description="list of texts to convert, see /transformText/ for formatting",), 
                      grid:str =Query("latest",description="RO Grid version number, latest for latest available grid"),
                      srs:str  =Query("4326",  description="Source EPSG Code, only EPSG:4326 (ETRS89) is currently supported"),
                      crs:str  =Query("3844",  description="Destination EPSG Code, only EPSG:3844 (Stere70) is currently supported")):

    dms4table = [dd4_or_dms4(line) for line in multiText]

    print(f"dms4table= {dms4table}")

    N,E,H,pct = zip(*dms4table)

    N = np.array(N,    dtype=float) #45
    E = np.array(E,    dtype=float) #25
    H = np.array(H,    dtype=float) #-36

    X = np.full_like(N, 0.0)
    Y = np.full_like(E, 0.0)
    Z = np.full_like(H, 0.0)

    t = rg.transformations.Transform() #if grid in ['latest'] else rg.transformations.TransDatRO(filename = f"{grid}")      
    t.etrs_to_st70(N,E,H, X,Y,Z)

    ret = list(zip(pct, N,E,H, X,Y,Z))

    import simplejson as json
    ret = json.loads(json.dumps(ret, ignore_nan=True))

    print({"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs})
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

    X = np.full_like(N, 0.0)
    Y = np.full_like(E, 0.0)
    Z = np.full_like(H, 0.0)

    t = rg.transformations.Transform() #if grid in ['latest'] else rg.transformations.TransDatRO(filename = f"{grid}")      
    t.etrs_to_st70(N,E,H, X,Y,Z)

    ret = list(zip(pct, N,E,H, X,Y,Z))

    import simplejson as json
    ret = json.loads(json.dumps(ret, ignore_nan=True))

    print({"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs})
    return {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs}


app.mount("/", StaticFiles(directory=f"{os.path.dirname(__file__)}/static",html = True), name="static")


