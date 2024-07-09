from typing import Literal
import re
import pandas as pd
import numpy as np
import logging

# LATEST

PREGEX_DMS = r"([NEne]?)(\d+)(\D+)(\d+)(\D+)([\d.]+)(\D)*"
PREGEX_DMS3= r"((?P<name>([\w\-\_\s\S])*)(?P<s0>[\s,;\t]))*(?P<lat>([NEne]?)(?P<lat_d>[4][345678]+)(\D+)(?P<lat_m>\d+)(\D+)(?P<lat_s>[\d]{2}([.][\d]+)*)(\D)*)(?P<s1>[\s,;\t])(?P<lon>([NEne]?)(?P<lon_d>[23][\d]+)(\D+)(?P<lon_m>\d+)(\D+)(?P<lon_s>[\d]{2}([.][\d]+)*)(\D)*)(?P<s2>[\s,;\t])(?P<height>[\d.]+)"
PREGEX_DMS4= r"((?P<name>([\w\-\_\s\S])*)(?P<s0>[\s,;\t]))*(?P<lat>(([NEne]?)(?P<lat_d>[4][345678]+)(\D+)(?P<lat_m>\d+)(\D+)(?P<lat_s>[\d]{2}([.][\d]+)*)|(?P<lat_dd>[4][345678]\.[\d]*))(\D)*)(?P<s1>[\s,;\t])(?P<lon>(([NEne]?)(?P<lon_d>[23][\d]+)(\D+)(?P<lon_m>\d+)(\D+)(?P<lon_s>[\d]{2}([.][\d]+)*)|(?P<lon_dd>[23][\d]\.[\d]*))(\D)*)(?P<s2>[\s,;\t])(?P<height>[\d.]+)"

NA_VALUES = ['','-',' ','.',',',
             '-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A','N/A', 'NA', '#NA', 'NULL', 'NaN', '-NaN', 'nan', '-nan']

BBOX_RO = [20.26, 43.44, 31.41, 48.27] 

# format types for printing
__FORMAT_TYPES = Literal["tuple", "dms", "text", "formated"]

# Replace list for special characters
__FIELD_SPECIALS = {"TAB": "\t",
                    "SPACE": " ",}

def print_debug(text, **kwargs):
    if logging.DEBUG >= logging.root.level:
        print(text, **kwargs)
    logging.info(text)

def print_info(text, **kwargs):
    print(text, **kwargs)
    logging.info(text)

def print_warn(text, **kwargs):
    print(text, **kwargs)
    logging.warn(text)

def print_err(text, **kwargs):
    print(text, **kwargs)
    logging.error(text)

def print_crit(text, **kwargs):
    print(text, **kwargs)
    logging.critical(text)


def is_DMS(val):
    """is the value in degree minutes seconds

    Args:
        val (any): any vaslue to test

    Returns:
        bool: true if valid DMS
    """    
    if val:
        mo = re.search(PREGEX_DMS,val)
        if mo:
            return True
        else:
            return False
    else:
        return False

def is_DMS3(val):
    """is the value in degree minutes seconds (3 sets)
    Args:
        val (any): any vaslue to test
    Returns:
        bool: true if valid DMS
    """    
    if val:
        mo = re.search(PREGEX_DMS,val)
        if mo:
            return True
        else:
            return False
    else:
        return False



def _predicate(L:list, predicate)->float:
    """Return the procentage of values that match function

    Args:
        L (list): input list
        predicate (function): function that returns bool

    Returns:
        float: output percentage
    """    
    return (sum(1.0 for v in L if predicate(v)) / len(L)) * 100

def is_inside_bounds(lat:float, lon:float)->bool:
    """Checks if coordinates are inside BBOX (Romania)
    Args:
        lat (float): Latitude in decimal degrees
        lon (float): Longitude in decimal degreees

    Returns:
        bool: True if inside, False if outside BBox
    """
    
    return (BBOX_RO[1] <= lat <= BBOX_RO[3]) and (BBOX_RO[0] <= lon <= BBOX_RO[2])

def dd_or_dms(x):
    """Converts parameter to decimal dgrees

    Args:
        x (string, float, whatever): value to be converted

    Raises:
        Exception: "Bad Value" conversion cannot be done

    Returns:
        float: Decimal degrees
    """
    import re
    try:
        return float(x)
    except:
        try:
            x = re.search(PREGEX_DMS,x).groups()
            return float(x[1]) + float(x[3])/60 + float(x[5])/3600
        except:
            return np.nan
            #raise Exception("Bad Value") 

def dd3_or_dms3(x):
    """Converts parameter to decimal dgrees
    Args:
        x (string, float, whatever): value to be converted
    Raises:
        Exception: "Bad Value" conversion cannot be done
    Returns:
        float: Decimal degrees
    """
    import re
    try:
        x = re.search(PREGEX_DMS3,x).groupdict()

        la = float(x['lat_d'] ) + float(x['lat_m']) /60 + float(x['lat_s']) /3600
        lo = float(x['lon_d']) + float(x['lon_m'])/60 + float(x['lon_s'])/3600
        he = float(x['height'])
        
        return la, lo, he
    except:
        return np.nan, np.nan, np.nan
        #raise Exception("Bad Value") 

def dd4_or_dms4(x) -> tuple[float, float, float, str]:
    """Converts parameter to decimal dgrees
    Args:
        x (string, float, whatever): value to be converted
    Raises:
        Exception: "Bad Value" conversion cannot be done
    Returns:
        float: Decimal degrees
    """
    import re
    try:
        x = re.search(PREGEX_DMS4,x).groupdict()

        if None != x['name']:
            pointName = x['name']
        else:
            pointName = ''

        if None not in [ x['lat_d'], x['lat_m'], x['lat_s'] ]:
            la = float(x['lat_d'] ) + float(x['lat_m']) /60 + float(x['lat_s']) /3600
        elif ['lat_dd'] != None:
            la = float(x['lat_dd'])
        else:
            la = np.nan

        if None not in [ x['lon_d'],x['lon_m'], x['lon_s'] ]:
            lo = float(x['lon_d'] ) + float(x['lon_m']) /60 + float(x['lon_s']) /3600
        elif ['lon_dd'] != None:
            lo = float(x['lon_dd'])
        else:
            lo = np.nan

        if ['height'] != None:
            he = float(x['height'])
        else:
            he = np.nan
        
        return la, lo, he, pointName
    except:
        return np.nan, np.nan, np.nan, "error"
        #raise Exception("Bad Value") 

def val_to_float(x):
    """Converts parameter to decimal dgrees

    Args:
        x (string, float, whatever): value to be converted

    Raises:
        Exception: "Bad Value" conversion cannot be done

    Returns:
        float: Decimal degrees
    """
    try:
        return float(x)
    except:
        return np.nan

def islat(v) -> bool:
    v = dd_or_dms(v)
    if v:
        return (BBOX_RO[1] <= v <= BBOX_RO[3])
    else:
        return False

def islon(v) -> bool:
    v = dd_or_dms(v)
    if v:
        return (BBOX_RO[0] <= v <= BBOX_RO[2])
    else:
        return False

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def infer_latlon(arrVal, trust=70):
    prdLat = _predicate(arrVal, islat)
    prdLon = _predicate(arrVal, islon)

    if prdLat>=trust:
        return "lat"
    elif prdLon>=trust:
        return "lon"
    else:
        return None      

def dd2dms(dd:float, format:__FORMAT_TYPES="tuple"):
    """Converts Decimal degrees to DD*MM'SS.ss", or to tuple (d,m,s) format

    Args:
        dd (float): _description_
        format (__FORMAT_TYPES, optional): _description_. Defaults to "tuple".

    Returns:
        string or tuple (d,m,s)
    """

    is_positive = dd >= 0
    dd = abs(dd)
    m,s = divmod(dd*3600,60)
    d,m = divmod(m,60)
    d = d if is_positive else -d
    if 'tuple' == format :
        return (d,m,s) 
    else:
        return f"{d:.0f}\N{DEGREE SIGN}{m:.0f}\N{Apostrophe}{s:.5f}\N{Quotation mark}"

def _field_delimiter(d:str):
    """Returns Unicode character from 
    named special definition

    Args:
        d (str): name of character in unicode standard format

    Returns:
        str: Unicode Character
    """
    if d.upper() in __FIELD_SPECIALS.keys():
        d = __FIELD_SPECIALS[d.upper()]
    
    return d

def sanitize_column(column_data):
    # remove whitespace
    c = column_data.str.replace(r'\s+', np.nan, regex=True)

    # remove wierd values
    for rr in NA_VALUES:
        c = c.replace(rr, np.nan)

    ## keep rows intact? or not?
    #c = c.dropna()
    return c
    
def sanitize_df_for_values(df,subset):
    df = df.dropna(axis='index', subset=subset, how='any')
    return df

def get_valid_subset(df):
    df = df.dropna(axis='index',  how='any')

def sanitize_df(df):

    #primary data cleanup by column
    for c in df.columns:
        c = sanitize_column(c)

    # drop empty rows
    df = df.dropna(axis='index', how='all')

def infer_type_from_data(column_data):
    """
    Infer the type of a column based on its content.

    Parameters:
    - column_data: pandas Series representing a column

    Returns:
    - column_type: string representing the inferred type of the column
    """

    types_list = ['datetime64', 'int64', 'float64', 'str']
    new_type='not_set'

    if column_data.dtype in types_list:
        return column_data.dtype
    else:
        for t in types_list:
            try:
                column_data.astype(dtype=t)
                new_type = t
                if new_type != "not_set":
                    break
            except:
                pass

    if new_type == 'str':
        try:
            df_filtered = column_data.apply(is_DMS)
            if df_filtered.all():
                new_type = 'DMS'
        except:
            pass

    return new_type

def infer_filetype(args):
    import magic
    return magic.from_file(args.filename, mime = True)

def infer_header(args): # not used
    
    ftype = infer_filetype(args)
    if ftype == "text/plain":
        df = pd.read_csv(args.filename, sep=args.sep,  comment="#")
    elif ftype=="application/vnd.ms-excel" :
        df = pd.read_excel(args.filename)
    else:
        raise Exception('Unknown file type')
    
    df = df.replace(r'^\s*$', np.nan, regex=True)

    df.convert_dtypes().dtypes

    for i, col in enumerate(df.columns):
        # guess lat or lon
        latlon_type = infer_latlon(df[col], 70)

        if latlon_type:
            print_info (f'{col} is {latlon_type}')
        else:
            if _predicate(df[col], is_float) >= 70:
                column_type = 'float64'
                print_info (f'{col} is {column_type}')
            else:
                column_type = 'str'
                print_info (f'{col} is {column_type}')        

def infer_columns(df):

    df = df.replace(r'^\s*$', np.nan, regex=True)
    df.convert_dtypes()

    c_lat, c_lon, c_h = None, None, None

    for col in df.columns:
        # guess lat or lon
        latlon_type = infer_latlon(df[col])

        if latlon_type:
            if latlon_type == 'lat':
                c_lat = col
            if latlon_type == 'lon':
                c_lon = col  

        if all([c_lon, c_lat, _predicate(df[col], is_float) >= 70 ]):
            c_h = col
            break

    return [c_lat, c_lon, c_h]