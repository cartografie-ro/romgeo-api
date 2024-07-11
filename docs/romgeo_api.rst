RomGEO API Documentation
========================

Overview
--------

The RomGEO API provides endpoints for transforming geographical coordinates between different reference systems, specifically designed for Romanian spatial data. This API supports transformations from ETRS89 to Stereo70 and UTM projections.

Base URL
--------

The base URL for the API is `/`.

Endpoints
---------

### `/transformText/{text}`

Transform a single text-based coordinate.

**Parameters:**

- `text` (str): The coordinate text to be transformed.
- `grid` (str, optional): RO Grid version number. Defaults to "latest".
- `srs` (str, optional): Source EPSG Code. Only EPSG:4326 (ETRS89) is supported. Defaults to "4326".
- `crs` (str, optional): Destination EPSG Code. Only EPSG:3844 (Stereo70) is supported. Defaults to "3844".
- `astext` (bool, optional): Output as text only. Defaults to `False`.

**Returns:**

- JSON object containing the transformed coordinates and additional information.

### `/transformCoord/{lat}/{lon}/{he}/`

Transform a single coordinate specified by latitude, longitude, and height.

**Parameters:**

- `lat` (str): Latitude of the coordinate.
- `lon` (str): Longitude of the coordinate.
- `he` (str): Height of the coordinate.
- `grid` (str, optional): RO Grid version number. Defaults to "latest".
- `srs` (str, optional): Source EPSG Code. Only EPSG:4326 (ETRS89) is supported. Defaults to "4326".
- `crs` (str, optional): Destination EPSG Code. Only EPSG:3844 (Stereo70) is supported. Defaults to "3844".
- `astext` (bool, optional): Output as text only. Defaults to `False`.

**Returns:**

- JSON object containing the transformed coordinates and additional information.

### `/transformMultiText/` (GET)

Transform multiple text-based coordinates specified in a list.

**Parameters:**

- `multiText` (list[str], optional): List of coordinate texts to be transformed. Defaults to a predefined list.
- `grid` (str, optional): RO Grid version number. Defaults to "latest".
- `srs` (str, optional): Source EPSG Code. Only EPSG:4326 (ETRS89) is supported. Defaults to "4326".
- `crs` (str, optional): Destination EPSG Code. Only EPSG:3844 (Stereo70) is supported. Defaults to "3844".

**Returns:**

- JSON object containing the transformed coordinates and additional information.

### `/transformMultiText/` (POST)

Transform multiple text-based coordinates specified in a list via a POST request.

**Parameters:**

- `multiText` (list[str], optional): List of coordinate texts to be transformed. Defaults to a predefined list.
- `grid` (str, optional): RO Grid version number. Defaults to "latest".
- `srs` (str, optional): Source EPSG Code. Only EPSG:4326 (ETRS89) is supported. Defaults to "4326".
- `crs` (str, optional): Destination EPSG Code. Only EPSG:3844 (Stereo70) is supported. Defaults to "3844".

**Returns:**

- JSON object containing the transformed coordinates and additional information.

Notes
-----

- Ensure that the EPSG codes used are supported: EPSG:4326 for the source (ETRS89) and EPSG:3844 for the destination (Stereo70).
- The `grid` parameter can be set to "latest" to use the most recent grid data available.
- The `astext` parameter, when set to `True`, returns the result as a plain text string instead of a JSON object.
