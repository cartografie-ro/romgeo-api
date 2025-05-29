# Romgeo API

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/cartografie-ro/romgeo-api/blob/main/README.md)
[![ro](https://img.shields.io/badge/lang-ro-green.svg)](https://github.com/cartografie-ro/romgeo-api/blob/main/README.ro.md)

O implementare simplă FastAPI a romgeo.

## Servicii online disponibile:
- Servicii Web API: [api.romgeo.ro](https://api.romgeo.ro/api/v1/demo.html#ro)
- Ghid pentru serviciile API: [api.romgeo.ro/api/v1/docs](https://api.romgeo.ro/api/v1/docs)

## Cerințe
 - romgeo
 - fastapi
 - simplejson
 - numba
 - numpy
 - pandas

## Instalare

<code>pip install romgeo_api</code>

## Pornirea API-ului:

Puteți începe cu următoarea comandă:

<code>fastapi romgeo_api:app --port=8881 --host=0.0.0.0</code>

Apoi deschideți browser-ul la <http://localhost:8881>

>Puteți schimba portul și IP-ul pe care va fi disponibil API-ul.

## Endpoint-uri API

### Transformă o singură coordonată text-based

Endpoint: /transformText/{text}

**Parametri:**
- text: Textul coordonatei care trebuie transformat.
- grid (opțional): Versiunea rețelei RO. Implicit este 'latest'.
- srs (opțional): Codul EPSG sursă. Doar EPSG:4326 (ETRS89) este suportat. Implicit este '4326'.
- crs (opțional): Codul EPSG destinație. Doar EPSG:3844 (Stereo70) este suportat. Implicit este '3844'.
- astext (opțional): Output ca text doar. Implicit este 'False'.

Exemplu URL: 

http\://localhost:8881/transformText/DEMO1%2044°34m31.54821s%2022°39m02.48758s%20198.848

Exemplu curl:

<code>curl -X GET 'http\://localhost:8881/transformText/DEMO1%2044°34m31.54821s%2022°39m02.48758s%20198.848'</code>

### Transformă o singură coordonată

Endpoint: /transformCoord/{lat}/{lon}/{he}/

**Parametri:**
- lat: Latitudinea coordonatei.
- lon: Longitudinea coordonatei.
- he: Înălțimea coordonatei.
- grid (opțional): Versiunea rețelei RO. Implicit este 'latest'.
- srs (opțional): Codul EPSG sursă. Doar EPSG:4326 (ETRS89) este suportat. Implicit este '4326'.
- crs (opțional): Codul EPSG destinație. Doar EPSG:3844 (Stereo70) este suportat. Implicit este '3844'.
- astext (opțional): Output ca text doar. Implicit este 'False'.

Exemplu URL: 

http\://localhost:8881/transformCoord/45.7489/25.2087/100

Exemplu curl:

<code>curl -X GET 'http\://localhost:8881/transformCoord/45.7489/25.2087/100'</code>

### Transformă coordonate multiple text-based (GET)

Endpoint: /transformMultiText/

**Parametri:**
- multiText (obligatoriu): Lista de texte coordonate care trebuie transformate.
- grid (opțional): Versiunea rețelei RO. Implicit este 'latest'.
- srs (opțional): Codul EPSG sursă. Doar EPSG:4326 (ETRS89) este suportat. Implicit este '4326'.
- crs (opțional): Codul EPSG destinație. Doar EPSG:3844 (Stereo70) este suportat. Implicit este '3844'.

Exemplu URL: 

http\://localhost:8881/transformMultiText/?multiText=DEMO1%2044°34m31.54821s%2022°39m02.48758s%20198.848&multiText=DEMO2%2044.84821%2022.48758%20198.848

Exemplu curl:

<code>curl -X GET 'http\://localhost:8881/transformMultiText/?multiText=DEMO1%2044°34m31.54821s%2022°39m02.48758s%20198.848&multiText=DEMO2%2044.84821%2022.48758%20198.848'</code>

### Transformă coordonate multiple text-based (POST)

Endpoint: /transformMultiText/

**Parametri:**
- multiText (obligatoriu): Lista de texte coordonate care trebuie transformate.
- grid (opțional): Versiunea rețelei RO. Implicit este 'latest'.
- srs (opțional): Codul EPSG sursă. Doar EPSG:4326 (ETRS89) este suportat. Implicit este '4326'.
- crs (opțional): Codul EPSG destinație. Doar EPSG:3844 (Stereo70) este suportat. Implicit este '3844'.

Exemplu curl:

<code>curl -X POST 'http\://localhost:8881/transformMultiText/' -H 'Content-Type: application/json' -d '{"multiText": ["DEMO1 44°34m31.54821s 22°39m02.48758s 198.848", "DEMO2 44.84821 22.48758 198.848"]}'</code>

## Cum se utilizează API-ul ROMGEO în Excel

1. Creați un document Excel nou.
2. Adăugați valori precum 'Point_1 N45°25m35.123456s E25°15m45.123456s 123.4567' într-o singură coloană, să zicem coloana A.
3. În celula B1 introduceți formula:
   <code>=WEBSERVICE('http\://api.romgeo.ro/v1/transformText/' & A1 & '&astext=true')</code>
4. Apăsați enter și trageți formula în jos. Veți obține 'lat, lon, elipsoidal_height, X_northing, Y_easting, Z_blacksea'.
5. Convertiți coloana B în text și împărțiți-o cu separatorul virgulă.

## Documentația este disponibilă aici

<https://github.com/cartografie-ro/romgeo-api/docs>
