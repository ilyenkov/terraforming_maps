# Terraforming maps 

## Altitude, Pressure, Temperature, Precipitations, Climate zones, Natural colors, Coast distance, Population density and radiation terraforming maps for Mars, Venus, Earth, Mercury and Moon. You can make images, mosaics of some types of maps and videos.

### How to use:
1) Copy files and unzip ziped files
2) Examples of usage in Terraforming_maps_introduction.ipynb
3) If you want to use original geotiffs, uncomment row 11 in MapConstructor.py

Youtube playlist with terraformed maps videos is here: https://www.youtube.com/playlist?list=PL4GHmzDz7hcVJEolJfmmpxtbw1mYLY5Mk

Example of generated maps mosaic for Mars with 30% sea share:

![alt text](https://github.com/ilyenkov/terraforming_maps/blob/main/Mars_30_percent_mosaic.jpg?raw=true)

### How it works:
Simplified climate model has three inputs: solar irradiance (W/m2), sea-level pressure (Pa) and land share (%). Atmosphere (sea-level pressure) creates greenhouse effect, i.e. increases average temperature and decrerases temperature range. Sea-level temperature depends on solar irradiance (including greenhouse effect) and latitude. Surface temperature and pressure depends on altitude and temperature lapse rates (which are Earth's temperature lapse rates adjusted to other surface gravity). Average level of precipitations depends on average temperature (higher temperature stimulates higher evaporation) and land share (on Earth water better evaporates from ocean). Precipitations depend on surface pressure, latitude and remoteness from seashore. And finally, climate depends on temperature and precipitations. Natural colors are colors of according climate zone on Earth. Population density depends on climate zone and coast distance.
This simplified climate model doesn't take in account ocean currents (so, for example, for Earth it gives slightly more wet climate for Sahara desert and slightly cooler climate for Europe), rotation speed of a planet (and so wind speed), it's orbital eccentricity and albedo (should be higher for calculations with lower solar irradiance and so create feedback loop decreasing temperature even more). But nevertheless, I assume these maps can help to imagine how will terraformed planets can look like.
Surface radiation (only for Mars) depends on atmosphere pressure. For Earth and Mars ice polar caps can be melted (default) or not.
