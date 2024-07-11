# Romgeo API

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/cartografie-ro/romgeo-api/blob/main/README.md)
[![ro](https://img.shields.io/badge/lang-ro-green.svg)](https://github.com/cartografie-ro/romgeo-api/blob/main/README.ro.md)


A simple FastAPI implementation of romgeo.

## Requirements
 - romgeo
 - fastapi
 - simplejson
 - numba
 - numpy
 - pandas

## Installation

<code>pip install romgeo_api</code>

## Run the API:

You can start with the following command:

<code>fastapi romgeo_api:app --port=8881 --host=0.0.0.0</code>

Then open your browser at <http://localhost:8881>

>You can change the port and IP on which the API will be available.

## API Endpoints

### Transform a Single Text-Based Coordinate

Endpoint: /transformText/{text}

**Parameters:**
- text: The coordinate text to be transformed.
- grid (optional): RO Grid version number. Defaults to 'latest'.
- srs (optional): Source EPSG Code. Only EPSG:4326 (ETRS89) is supported. Defaults to '4326'.
- crs (optional): Destination EPSG Code. Only EPSG:3844 (Stereo70) is supported. Defaults to '3844'.
- astext (optional): Output as text only. Defaults to 'False'.

Example URL: 

http\://localhost:8881/transformText/DEMO1%2044°34m31.54821s%2022°39m02.48758s%20198.848

Example curl:

<code>curl -X GET 'http\://localhost:8881/transformText/DEMO1%2044°34m31.54821s%2022°39m02.48758s%20198.848'</code>

### Transform a Single Coordinate

Endpoint: /transformCoord/{lat}/{lon}/{he}/

**Parameters:**
- lat: Latitude of the coordinate.
- lon: Longitude of the coordinate.
- he: Height of the coordinate.
- grid (optional): RO Grid version number. Defaults to 'latest'.
- srs (optional): Source EPSG Code. Only EPSG:4326 (ETRS89) is supported. Defaults to '4326'.
- crs (optional): Destination EPSG Code. Only EPSG:3844 (Stereo70) is supported. Defaults to '3844'.
- astext (optional): Output as text only. Defaults to 'False'.

Example URL: 

http\://localhost:8881/transformCoord/45.7489/25.2087/100

Example curl:

<code>curl -X GET 'http\://localhost:8881/transformCoord/45.7489/25.2087/100'</code>

### Transform Multiple Text-Based Coordinates (GET)

Endpoint: /transformMultiText/

**Parameters:**
- multiText (required): List of coordinate texts to be transformed.
- grid (optional): RO Grid version number. Defaults to 'latest'.
- srs (optional): Source EPSG Code. Only EPSG:4326 (ETRS89) is supported. Defaults to '4326'.
- crs (optional): Destination EPSG Code. Only EPSG:3844 (Stereo70) is supported. Defaults to '3844'.

Example URL: 

http\://localhost:8881/transformMultiText/?multiText=DEMO1%2044°34m31.54821s%2022°39m02.48758s%20198.848&multiText=DEMO2%2044.84821%2022.48758%20198.848

Example curl:

<code>curl -X GET 'http\://localhost:8881/transformMultiText/?multiText=DEMO1%2044°34m31.54821s%2022°39m02.48758s%20198.848&multiText=DEMO2%2044.84821%2022.48758%20198.848'</code>

### Transform Multiple Text-Based Coordinates (POST)

Endpoint: /transformMultiText/

**Parameters:**
- multiText (required): List of coordinate texts to be transformed.
- grid (optional): RO Grid version number. Defaults to 'latest'.
- srs (optional): Source EPSG Code. Only EPSG:4326 (ETRS89) is supported. Defaults to '4326'.
- crs (optional): Destination EPSG Code. Only EPSG:3844 (Stereo70) is supported. Defaults to '3844'.


Example curl:

<code>curl -X POST 'http\://localhost:8881/transformMultiText/' -H 'Content-Type: application/json' -d '{"multiText": ["DEMO1 44°34m31.54821s 22°39m02.48758s 198.848", "DEMO2 44.84821 22.48758 198.848"]}'</code>

## How to use the API from ROMGEO in Excel

- Create a new Excel document.
- Add values like 'Point_1 N45°25m35.123456s E25°15m45.123456s 123.4567' into a single column, let's say column A.
- In B1 type this formula:
   <code>=WEBSERVICE('http\://api.romgeo.ro/v1/transformText/' & A1 & '&astext=true')</code>
- Press enter
- and drag-fill the formula down. You will get 'lat, lon, elipsoidal_height, X_northing, Y_easting, Z_blacksea'.
- Convert column B to text, and split it with comma as separator.

## Documentation is available here

<https://github.com/cartografie-ro/romgeo-api/docs>
