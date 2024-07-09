import romgeo as rg
import romgeo.cuda
import numpy as np
import numba as nb
import numba.cuda
import math
import os

from .extras import is_inside_bounds, dd4_or_dms4

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.get("/transformText/{text}")
def Convert_Text(text:str, grid:str="latest", srs:str="4326", crs:str="3844"):

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
    t3 = (-300 <= H[0] < 2500)
    t4 = (-200 <= Z[0] < 2600)

    print([t1, t2, t3, t4])

    if not all([t1, t2, t3, t4]):
        print(ret)
        ret = "Input error or Out Of Bounds."

    return {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs}


@app.get("/transformMultiText/")
def Convert_MultiText(multiText:list[str], grid:str = "latest", srs:str="EPSG:4326", crs:str="EPSG:3844"):
    
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
    return {"result": ret, "grid":grid, "grid_version":t.grid_version, "srs": srs, "crs": crs}




app.mount("/", StaticFiles(directory=f"{os.path.dirname(__file__)}/static",html = True), name="static")


