# Terraforming maps 

Altitude, Pressure, Temperature, Precipitations, Climate zones, Natural colors, Coast distance, Population density and radiation terraforming maps for Mars, Venus, Earth, Mercury and Moon. You can make images, mosaics of some types of maps and videos.

### How to use:
1) Copy files and unzip ziped files
2) Examples of usage in Terraforming_maps_introduction.ipynb
3) If you want to use original geotiffs, uncomment row 11 in MapConstructor.py

Youtube playlist with terraformed maps videos is here: https://www.youtube.com/playlist?list=PL4GHmzDz7hcVJEolJfmmpxtbw1mYLY5Mk

Example of generated maps mosaic for Mars with 30% sea share:

![alt text](https://github.com/ilyenkov/terraforming_maps/blob/main/Mars_30_percent_mosaic.jpg?raw=true)

### How it works:

First you should calc water levels depending of water volume. They are precalced in params_dict.json for 3 different resolutions (FHD, 4K and 8K). Altitude scale by default is normed to surface gravitation (so Altitude maps of Moon, Mercury and Mars with lower surface gravitation are greener than they should be if we use the same colors as on Earth or Venus altitude maps).

Simplified climate model has three inputs: solar irradiance (W/m2), sea-level pressure (Pa) and land share (%). Atmosphere (sea-level pressure) creates greenhouse effect, i.e. increases average temperature and decrerases temperature range. Sea-level temperature depends on solar irradiance (including greenhouse effect) and latitude. Surface temperature and pressure depends on altitude and temperature lapse rates (which are Earth's temperature lapse rates adjusted to other surface gravity). Average level of precipitations depends on average temperature (higher temperature stimulates higher evaporation) and land share (on Earth water better evaporates from ocean). Precipitations depend on surface pressure, latitude and remoteness from seashore. And finally, climate depends on temperature and precipitations. Natural colors are colors of according climate zone on Earth. There is also purple colormap in case you want to see, how planet can look from space if it has retinal-based photosyntesis. Population density depends on climate zone and coast distance.

This simplified climate model doesn't take in account ocean currents (so, for example, for Earth it gives slightly more wet climate for Sahara desert and slightly cooler climate for Europe), rotation speed of a planet (and so wind speed), it's orbital eccentricity and albedo (should be higher for calculations with lower solar irradiance and so create feedback loop decreasing temperature even more). But nevertheless, I assume these maps can help to imagine how will terraformed planets can look like.

Surface radiation (only for Mars) depends on atmosphere pressure. For Earth and Mars ice polar caps can be melted (default) or not.

### Original GeoTIFFs: 
https://astrogeology.usgs.gov/search/map/Mars/Topography/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2
https://astrogeology.usgs.gov/search/map/Venus/Magellan/RadarProperties/Venus_Magellan_Topography_Global_4641m_v02
https://www.eea.europa.eu/data-and-maps/data/world-digital-elevation-model-etopo5
https://astrogeology.usgs.gov/search/map/Mercury/Topography/MESSENGER/Mercury_Messenger_USGS_DEM_Global_665m_v2
https://astrogeology.usgs.gov/search/map/Moon/LRO/LOLA/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014
