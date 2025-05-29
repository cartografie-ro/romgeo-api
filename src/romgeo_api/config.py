DESC_TEXT_FORMAT = "Should contain: PointName(optional), lat, lon in any format (DD or DMS) and height as float"

DEF_MULTILIST = ["DEMO1 44°34\'31.54821\" 22°39\'02.48758\" 198.848",
                 "DEMO2 N44g34m31.54821s 22 39 02.48758 E 198.848",
                 "DEMO3 44.84821 22.48758 198.848m"]

INFO_TEXT = """The DXF file is in projected coordinates, EPSG:3844 (Stereo70) with heights referenced to Black Sea 1975.
Fisierul DXF este in coordonate EPSG:3844 (Stereo70) cu inaltimi referite la Marea Neagra 1975 (sistem local romanesc).

https://epsg.io/3844
"""

PRJ_CONTENT = """PROJCS["Pulkovo 1942(58) / Stereo70",
GEOGCS["Pulkovo 1942(58)",
    DATUM["Pulkovo_1942_58",
        SPHEROID["Krasovsky 1940",6378245,298.3,
            AUTHORITY["EPSG","7024"]],
        AUTHORITY["EPSG","6170"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4170"]],
PROJECTION["Oblique_Stereographic"],
PARAMETER["latitude_of_origin",46],
PARAMETER["central_meridian",25],
PARAMETER["scale_factor",0.99975],
PARAMETER["false_easting",500000],
PARAMETER["false_northing",500000],
UNIT["metre",1,
    AUTHORITY["EPSG","9001"]],
AUTHORITY["EPSG","3844"]],
VERT_CS["Black Sea 1975 height",
    VERT_DATUM["Black Sea 1975",2005,
        AUTHORITY["CUSTOM","BlackSea_1975"]],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Gravity-related height",UP]]]"""

SHP_PRJ_CONTENT="""PROJCS["Pulkovo_1942_Adj_58_Stereo_70",GEOGCS["GCS_Pulkovo_1942_Adj_1958",DATUM["D_Pulkovo_1942_Adj_1958",SPHEROID["Krasovsky_1940",6378245.0,298.3]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Double_Stereographic"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",500000.0],PARAMETER["Central_Meridian",25.0],PARAMETER["Scale_Factor",0.99975],PARAMETER["Latitude_Of_Origin",46.0],UNIT["Meter",1.0]],VERTCS["BlackSea_1975",VDATUM["BlackSea_1975"],PARAMETER["Vertical_Shift",0.0],PARAMETER["Direction",1.0],UNIT["Meter",1.0]]"""

TMP_ROOT = "/tmp/api-shapefiles"

ZBOX_RO_ETRS = [-100, 2600]
ZBOX_RO_ST70 = [ -50, 2600]



