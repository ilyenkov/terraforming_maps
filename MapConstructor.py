import numpy as np
import pandas as pd
#import cartopy.crs as ccrs
import json, os, matplotlib 
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
from matplotlib.colors import LinearSegmentedColormap as lsc
from matplotlib.colors import ListedColormap
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics.pairwise import haversine_distances
#from osgeo import gdal   ##################### раскомментить это
from math import radians, atan2, sin, cos, sqrt
import time
import math
import cv2 as cv
from mpl_toolkits.basemap import Basemap
from joblib import Parallel, delayed
import gc
from matplotlib.figure import Figure
import pickle
from sys import getsizeof
import io
import copy
import lightgbm as lgb
import pickle
import multiprocessing
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter
plt.switch_backend('Agg')
np.seterr(all='ignore')
import warnings
warnings.filterwarnings("ignore")

def getRotMatrix(rot_x, rot_y, rot_z, unit='degree'):
    if(unit == 'degree'):
        rot_x = np.deg2rad(rot_x)
        rot_y = np.deg2rad(rot_y)
        rot_z = np.deg2rad(rot_z)

    RotMat_X = np.array([[1, 0, 0],
                         [0, math.cos(rot_x), -math.sin(rot_x)],
                         [0, math.sin(rot_x), math.cos(rot_x)]])
    RotMat_Y = np.array([[math.cos(rot_y), 0, math.sin(rot_y)],
                         [0, 1, 0], 
                         [-math.sin(rot_y), 0, math.cos(rot_y)]])
    RotMat_Z = np.array([[math.cos(rot_z), -math.sin(rot_z), 0],
                         [math.sin(rot_z), math.cos(rot_z), 0],
                         [0, 0, 1]])
    return np.matmul(np.matmul(RotMat_X, RotMat_Y), RotMat_Z)

def Pixel2LonLat(equirect):
    # LongLat - shape = (N, 2N, (Long, Lat)) 
    Lon = np.array([2*(x/equirect.shape[1]-0.5)*np.pi for x in range(equirect.shape[1])])
    Lat = np.array([(0.5-y/equirect.shape[0])*np.pi for y in range(equirect.shape[0])])
        
    Lon = np.tile(Lon, (equirect.shape[0], 1))
    Lat = np.tile(Lat.reshape(equirect.shape[0], 1), (equirect.shape[1]))

    return np.dstack((Lon, Lat))

def LonLat2Sphere(LonLat):
    x = np.cos(LonLat[:, :, 1])*np.cos(LonLat[:, :, 0])
    y = np.cos(LonLat[:, :, 1])*np.sin(LonLat[:, :, 0])
    z = np.sin(LonLat[:, :, 1])

    return np.dstack((x, y, z))

def Sphere2LonLat(xyz):
    Lon = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])
    Lat = np.pi/2 - np.arccos(xyz[:, :, 2])

    return np.dstack((Lon, Lat))

def LonLat2Pixel(LonLat):
    width = LonLat.shape[1]
    height = LonLat.shape[0]
    j = (width*(LonLat[:, :, 0]/(2*np.pi)+0.5))%width
    i = (height*(0.5-(LonLat[:, :, 1]/np.pi)))%height

    return np.dstack((i, j)).astype('int')

def isEquirect(height, width):
    if(height*2 != width):
        print("Warning: Source Image is not an Equirectangular Image...")
        print("height is %d, width is %d" %(height, width))
        return False
    return True

def Equirect_Rotate(src_img, rot_x, rot_y, rot_z, isInverse = False, unit = 'degree'):
    height = src_img.shape[0]
    width = src_img.shape[1]
    if(not isEquirect(height, width)):
        print("End program...")
        return
    
    Rot_Matrix = getRotMatrix(rot_x, rot_y, rot_z, unit)
    if(isInverse):
        Rot_Matrix = np.transpose(Rot_Matrix)
    
    out_img = np.zeros_like(src_img)

    # mapping equirect coordinate into LonLat coordinate system
    out_LonLat = Pixel2LonLat(out_img)

    # mapping LonLat coordinate into xyz(sphere) coordinate system
    out_xyz = LonLat2Sphere(out_LonLat)

    src_xyz = np.zeros_like(out_xyz)
    Rt = np.transpose(Rot_Matrix)
    src_xyz = np.matmul( out_xyz, Rt)

    # mapping xyz(sphere) coordinate into LonLat Coordinate system
    src_LonLat = Sphere2LonLat(src_xyz)

    # mapping LonLat coordinate into equirect coordinate system
    src_Pixel = LonLat2Pixel(src_LonLat)
    
    out_img = src_img[src_Pixel[:,:,0], src_Pixel[:,:,1]]
    
    return out_img

def calc_pixel_square(dm, x, y, r):
    pixel_v = math.pi*r / dm[0]
    pixel_h = 2*math.pi*r / dm[1] * math.cos(abs(y-dm[0]/2)/(dm[0]/2)*math.pi/2)
    sq = pixel_v*pixel_h
    return sq

def calc_pixel_diag(dm, x, y, r):
    pixel_v = math.pi*r / dm[0]
    pixel_h = 2*math.pi*r / dm[1] * math.cos(abs(y-dm[0]/2)/(dm[0]/2)*math.pi/2)
    return (pixel_v**2+pixel_h**2)**0.5

with open('neigh.pickle', 'rb') as f:
    neigh = pickle.load(f)
    
with open('pop_den_dict.pickle', 'rb') as f:
    pop_den_dict = pickle.load(f)
    
with open('params_dict.json') as json_file:
    params_dict = json.load(json_file)        
    
with open('colors_dict.json') as json_file:
    colors_dict = json.load(json_file)
    
cm_europe_4 = ListedColormap(colors_dict['europe_4_cl'])
cm_delta = ListedColormap(colors_dict['delta_cl'])
cm_des2 = ListedColormap(colors_dict['des2_cl'])
cm_wiki = ListedColormap(colors_dict['wiki_cl'])
cm_rain = ListedColormap(colors_dict['rain_cl'])
cmap_windy_temp = lsc.from_list('cm_windy', colors_dict['cm_windy_temp'], gamma=1.0)
cmap_windy_pressure = lsc.from_list('cm_windy', colors_dict['cm_windy_pressure'], gamma=1.0)
cmap_windy_precipation = lsc.from_list('cm_windy', colors_dict['cm_windy_precipation'], gamma=1.0)
cm_knutux = ListedColormap(colors_dict['knutux_cl'])
cmap_climate = colors_dict['cmap_climate']
cmap_natural_colors_most_pop_8 = colors_dict['cmap_natural_colors_most_pop_8']
colors_list_purple = colors_dict['cmap_natural_colors_most_pop_8_purple']

pressure_table = pd.DataFrame({'b': [0, 1, 2, 3, 4, 5, 6], 'static_pressure': [5*101325, 22632, 5475, 868, 111, 67, 4],
                               'lapse_rate': [-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002],
                               #'temperature': [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
                              })
# https://en.wikipedia.org/wiki/Barometric_formula

radiation_pressure_levels = [0,2026,4053,6079,8106,10132,12159,14185,16212,18238,20265,22291,24318,26344,28371,30397,32424,
                             34450,36477,38503,40530,42556,44583,46609,48636,50662,52689,54715,56742,58768,60795,62821,64848,
                             66874,68901,70927,72954,74980,77007,79033,81060,83086,85113,87139,89166,91192,93219,95245,97272,
                             99298,101325]

radiation_levels = [233.500538,110.176163,118.840350,126.201488,127.843988,124.706813,117.299138,106.762500,94.394475,
                    81.391350,68.763263,57.087825,46.679850,37.700850,30.098813,23.784221,18.622939,14.469878,11.149564,
                    8.534978,6.491434,4.907243,3.687960,2.756663,2.050579,1.518847,1.119227,0.821332,0.600525,0.437179,
                    0.317194,0.229416,0.165189,0.118649,0.085002,0.060663,0.043159,0.030630,0.021683,0.015311,0.010785,
                    0.007578,0.005312,0.003715,0.002593,0.001805,0.001254,0.000870,0.000602,0.000416,0.000287]

#constants
sigma = 5.67*10**(-8)
R_star = 8.3144598 #  universal gas constant:  J/(mol·K)
M = 0.0289644      # molar mass of Earth's air: 0.0289644 kg/mol

albedo = 0.31 
g_0_earth = 9.807        # Earth gravitational acceleration: m/s2

def pws(t):
    return np.exp(77.3450+0.0057*(t+273.15)-7235/(t+273.15))/((t+273.15)**8.2)

figsize = (19.53, 18.55/2-1.7) 

###############################################################################################################
###############################################################################################################

class MapMaker:
    
    def __init__(self, planet='Mars', quality='FHD', recalc_water_levels=False, norm_highland_level=True, melt_ice_caps=True):
        
        self.planet = planet
        self.quality = quality
        
        if not melt_ice_caps:
            self.params = params_dict[self.planet]
        elif self.planet == 'Mars':
            self.params = params_dict[self.planet+'_melted_caps']
        elif self.planet == 'Earth':
            self.params = params_dict[self.planet+'_melted_caps']    
        else:
            print('!!! Melting of ice caps is possible only for Mars now !!!')
            self.params = params_dict[self.planet]
        
        self.r = self.params['r']
        self.full_square = 4*math.pi*self.r**2
        
        self.g_0 = self.params['g_0']
        self.pressure_table = pressure_table.copy(deep=True)
        self.pressure_table['lapse_rate'] = self.pressure_table['lapse_rate'] * self.g_0 / g_0_earth
        #https://en.wikipedia.org/wiki/Lapse_rate
        
        self.filename = self.params['filename']
        self.filename_pickle = self.params['filename_pickle']
        
        if norm_highland_level:
            self.highland=self.params['highland_level_normed']
        else:
            self.highland=self.params['highland_level']
        
        if self.quality == '8K':
            self.size = (6750, 3375)
            self.dpi = 600
        elif self.quality =='4K':
            self.size = (3374, 1687)
            self.dpi = 300
        elif self.quality == 'FHD':
            self.size = (1686, 843)
            self.dpi = 150
        else:
            print('!!! Not supported quality !!!')
        
        self.read_file(melt_ice_caps=melt_ice_caps)
        
        if recalc_water_levels:
            self.calc_water_levels()
        else:
            self.water_volumes = self.params[self.quality]['water_volumes']
            self.sea_levels = self.params[self.quality]['sea_levels']
            self.sea_shares = self.params[self.quality]['sea_shares']
        
        
    def read_file(self, melt_ice_caps):
        
        if os.path.exists(self.filename_pickle):
            with open(self.filename_pickle, 'rb') as f:
                self.heights = pickle.load(f)
            self.min_orig = self.params['min_orig']
            self.max_orig = self.params['max_orig']
            
            if self.planet =='Mars':
                with open('mars_ice.pickle', 'rb') as f:
                    self.ice_caps = pickle.load(f)  
            elif self.planet =='Earth':
                with open('earth_ice.pickle', 'rb') as f:
                    self.ice_caps = pickle.load(f)
            else:
                print('!!! No available polar ice caps file for '+self.planet+'. Polar ice caps equals to zero.')     
                self.ice_caps = np.zeros(self.heights.shape)
            
            if melt_ice_caps:
                self.heights = self.heights - self.ice_caps
            
        elif os.path.exists(self.filename):
            gdal_data = gdal.Open(self.filename)
            self.heights = gdal_data.ReadAsArray().astype(np.int16)
            del gdal_data
            gc.collect()
            self.min_orig = self.heights[self.heights!=-32768].min()
            self.max_orig = self.heights[self.heights!=-32768].max()
            print('!!! Polar ice caps equals to zero. This option is possible only with precalculated heights files.')
            self.ice_caps = np.zeros(self.heights.shape)
            
            #interpolation
            if self.planet=='Venus':
                self.heights = self.heights.astype(float)
                self.heights[self.heights==-32768] = np.nan
                self.heights = pd.DataFrame(self.heights).interpolate(method='linear', limit_direction='both', axis=1).values*0.5+\
                                pd.DataFrame(self.heights).interpolate(method='linear', limit_direction='both', axis=0).values*0.5
                self.heights = np.vstack((pd.DataFrame(self.heights[:2048]).interpolate(method='backfill', limit_direction='backward', axis=1).values, 
                                          pd.DataFrame(self.heights[2048:]).interpolate(method='ffill', limit_direction='forward', axis=1).values))
            elif self.planet=='Mars':
                heights_2_last_col = self.heights[:,-2:].astype(float)
                heights_2_last_col[heights_2_last_col==-32768] = np.nan
                heights_2_last_col = pd.DataFrame(heights_2_last_col).interpolate(method='ffill', limit_direction='forward', axis=0).values
                self.heights[:,-2:] = heights_2_last_col.astype(int)
            else:
                self.heights[self.heights==-32768] = self.heights[self.heights!=-32768].mean()
        else:
            print('!!! No available file for '+self.planet+'. Please, download it.')        
            
        #resizing
        self.heights = cv.resize(self.heights, dsize=self.size, interpolation=cv.INTER_AREA)
        self.ice_caps = cv.resize(self.ice_caps, dsize=self.size, interpolation=cv.INTER_AREA)
        
        self.min_new = self.heights[np.isnan(self.heights)==False].min()
        self.max_new = self.heights[np.isnan(self.heights)==False].max()
        self.max_delta = self.max_new - self.min_new
        
        self.heights = self.heights-self.min_new
        
        #areas of each pixel
        self.s = [calc_pixel_square(self.heights.shape, 0, i, self.r) for i in range(self.heights.shape[0])]
        self.s = np.array([self.s]*self.heights.shape[1]).T
        
        #lats_lons
        self.lats = -np.array([[(i-self.heights.shape[0]/2)/(self.heights.shape[0]/2)*90 for i in range(self.heights.shape[0])]]*self.heights.shape[1]).T
        self.lons = np.array([[i/(self.heights.shape[1]-1)*360 for i in range(self.heights.shape[1])]]*self.heights.shape[0])
        
        #pixel_diagonal
        self.arcs_diag = [calc_pixel_diag(self.heights.shape, 0, i, self.r) for i in range(self.heights.shape[0])]
        self.arcs_diag = np.array([self.arcs_diag]*self.heights.shape[1]).T
        
        #switching hemispheres
        self.heights = np.hstack([self.heights[:, int(self.heights.shape[1]/2):],
                                  self.heights[:,:int(self.heights.shape[1]/2)]])
        
        azimuth = 315
        altitude = 45
        x, y = np.gradient(self.heights*0.025)
        azimuthrad = azimuth * np.pi / 180.0
        altituderad = altitude * np.pi / 180.0
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos((azimuthrad - np.pi / 2.0) - aspect)
        self.hillshade = 255 * (shaded + 1) / 2
        del x, y, azimuthrad, altituderad, slope, aspect, shaded
        gc.collect()
        
        self.hs_dict={}
        self.shade_dict={}
        
        self.above_part = None
        self.remoteness = None
        self.temperatures = None
        self.pressures = None
        self.precipitations = None
        self.climate = None
        self.population_density = None
        self.radiation = None
        
    def upscale_frame(self, frame, size, resize=False):
        if resize:
            if frame.shape[1]/frame.shape[0] > size[0]/size[1]:            
                frame = cv.resize(frame, dsize=(size[0], int(frame.shape[0]*size[0]/frame.shape[1])), interpolation=cv.INTER_CUBIC)
            else:            
                frame = cv.resize(frame, dsize=(int(frame.shape[1]*size[1]/frame.shape[0]), size[1]), interpolation=cv.INTER_CUBIC)
        new_frame = np.ones((size[1],size[0],3), dtype=np.uint8)*255
        left = (size[1] - frame.shape[0])//2
        right = size[1] - frame.shape[0] - left
        top = (size[0] - frame.shape[1])//2
        bottom = size[0] - frame.shape[1] - top

        new_frame[left:size[1]-right, top:size[0]-bottom,:] = frame

        return new_frame
        
    def calc_water_levels(self):
        
        self.water_volumes = [i*self.params['water_recalc']['step'] for i in range(0, self.params['water_recalc']['num_steps'], 1)]
        self.sea_levels = [0 for i in range(0, self.params['water_recalc']['num_steps'], 1)]
        self.sea_shares = [0 for i in range(0, self.params['water_recalc']['num_steps'], 1)]

        levels_volume = []
        total_weight = 0
        level = 1
        
        for l in range(self.params['water_recalc']['meters_range']):

            v = np.sum((self.heights<=l*level).astype(int)*s)*level # m**3

            levels_volume.append(v)
            total_weight = total_weight + v*1000 # kg
            print(l*level, round(v), round(total_weight), round(v/level/self.full_square,4)) if l*level%200==0 else next

            for i, w in enumerate(self.water_volumes):
                if total_weight>=w:
                    if self.sea_levels[i]==0:
                        self.sea_levels[i]=level*l
                        self.sea_shares[i]=round(v/level/self.full_square*100,1)
                        
    def calc_remoteness(self, i, verbose=False):
        
        start_time=time.time()
        self.above_part = copy.deepcopy(self.heights)
        self.above_part = self.above_part-self.sea_levels[i]
        self.above_part[self.above_part==0] = 0.1
        self.above_part[self.above_part<0] = 0
        
        if self.quality == '8K':
            kernel_size=32
        elif self.quality =='4K':
            kernel_size=16
        elif self.quality == 'FHD':
            kernel_size=8
        else:
            print('!!! Not supported quality !!!')
        
        sea_part = (self.above_part==0).astype(int)        
        kernel = np.ones((kernel_size, kernel_size), np.float32)/(kernel_size**2)
        water_near = cv.filter2D(sea_part.astype(np.float32), -1, kernel)
        #water_near = (water_near>=0.9999).astype(int)
        land_part = (water_near<0.5).astype(int)
        self.large_land = maximum_filter(water_near, footprint=np.ones([kernel_size*3, kernel_size*3]))
        #self.large_land[self.large_land>0.999] = 1
        self.above_part[(self.above_part==0)&(land_part==1)]=1
        #self.above_part[(self.above_part==0)&(land_part==1)&(self.large_land>0.999)]=1
        
        land_part = (self.above_part>0).astype(int)
        land_near = cv.filter2D(land_part.astype(np.float32), -1, kernel)
        self.large_land = maximum_filter(land_near, footprint=np.ones([kernel_size*2, kernel_size*2]))
        self.large_land[self.large_land>0.999] = 1

        remoteness1 = ndimage.distance_transform_edt(self.above_part, return_indices=False)
        remoteness1 = remoteness1*self.arcs_diag

        #rot_height = Equirect_Rotate(height, 0, 0, 180, isInverse = False, unit = 'degree')
        remoteness2 = ndimage.distance_transform_edt(np.hstack((self.above_part[:,self.above_part.shape[1]//2:], 
                                                                self.above_part[:,:self.above_part.shape[1]//2])), 
                                                     return_indices=False)
        remoteness2 = remoteness2*self.arcs_diag
        #remoteness2 = Equirect_Rotate(remoteness2, 0, 0, -180, isInverse = False, unit = 'degree')
        remoteness2 = np.hstack((remoteness2[:, self.above_part.shape[1]//2:], remoteness2[:, :self.above_part.shape[1]//2]))

        rot_height = Equirect_Rotate(self.above_part, 90, 0, 0, isInverse = False, unit = 'degree')
        remoteness3 = ndimage.distance_transform_edt(rot_height, return_indices=False)
        remoteness3 = remoteness3*self.arcs_diag
        #remoteness3 = Equirect_Rotate(remoteness3, -90, 0, 0, isInverse = False, unit = 'degree')

        #rot_height = Equirect_Rotate(height, -90, 0, 0, isInverse = False, unit = 'degree')
        remoteness4 = ndimage.distance_transform_edt(np.hstack((rot_height[:, self.above_part.shape[1]//2:], 
                                                                rot_height[:, :self.above_part.shape[1]//2])), 
                                                     return_indices=False)
        remoteness4 = remoteness4*self.arcs_diag
        remoteness4 = np.hstack((remoteness4[:, self.above_part.shape[1]//2:], remoteness4[:, :self.above_part.shape[1]//2]))

        remoteness4 = np.minimum(remoteness3, remoteness4)    
        remoteness4 = Equirect_Rotate(remoteness4, -90, 0, 0, isInverse = False, unit = 'degree')

        self.remoteness = np.min(np.array([remoteness1, remoteness2, remoteness4]), axis=0)
        
        if i==0:
            self.remoteness = np.ones(self.remoteness.shape) * self.r * math.pi
        
        self.average_remoteness = (self.remoteness*self.s)[self.remoteness>0].sum() / self.s[self.remoteness>0].sum() / 1000
        self.max_remoteness = self.remoteness.max() / 1000
        if verbose:
            print('Coast distance calcs', round(time.time()-start_time,1), 'seconds')
                        
    def calc_temperature_pressure(self, i, solar_irradiance=1361, sea_level_pressure=101325, verbose=False):
        
        start_time=time.time()
        #if self.above_part == None:
        #self.above_part = copy.deepcopy(self.heights)
        #self.above_part = self.above_part-self.sea_levels[i]
        #self.above_part[self.above_part<0] = 0
        
        fs = 0.802222*(sea_level_pressure/101325)**0.19601
        self.temp_avg = ( solar_irradiance * (1-albedo) / (4*sigma*(1-fs/2)) ) ** (1/4)
        self.temp_delta = -((sea_level_pressure/101325)-4.6259) / 0.0625
        #if verbose:
        #    print('Average temperature', self.temp_avg, 'K, temperature range', self.temp_delta, 'K')
            
        #temperatures
        #sea_level:
        self.temperatures = self.temp_avg-self.temp_delta*0.8+self.temp_delta*np.sin((1-np.abs(self.lats)/90)*(np.pi/2)) + \
                                4*np.sin(np.abs(self.lats)/90*np.pi*2) 
        self.temperatures[self.heights<self.sea_levels[i]] = np.nan

        #pressures and temperature height correction
        for b, p in enumerate(self.pressure_table.static_pressure):
            if b <= sea_level_pressure:
                break

        h_b = 0
        P_b = sea_level_pressure
        T_b = self.temp_avg
        self.pressures = np.ones(self.heights.shape)*P_b
        self.pressures[self.heights<self.sea_levels[i]] = np.nan

        for b_i in range(b, 7):
            L_b = self.pressure_table.loc[b_i, 'lapse_rate']
            if b_i < 6:
                P_b_next = self.pressure_table.loc[b_i+1, 'static_pressure']
            else:
                P_b_next = 0
            #print(b_i, L_b, P_b_next)
            if L_b != 0:
                self.pressures[self.above_part>=h_b] = self.pressures[self.above_part>=h_b]*(T_b / (T_b + L_b * (self.above_part[self.above_part>=h_b] - h_b)))**(self.g_0*M/(R_star*L_b))   
            else:
                self.pressures[self.above_part>=h_b] = self.pressures[self.above_part>=h_b]*np.exp(-self.g_0*M*(self.above_part[self.above_part>=h_b] - h_b)/(R_star*T_b))

            self.temperatures[self.above_part>=h_b] = self.temperatures[self.above_part>=h_b] + L_b * self.above_part[self.above_part>=h_b]
            #print('temp pixels', b_i,'|', len(temperatures[~np.isnan(temperatures)]))

            if ((self.pressures[~np.isnan(self.pressures)]<P_b_next*1.01)&(self.pressures[~np.isnan(self.pressures)]>P_b_next*0.99)).sum()>0:        
                h_b = self.above_part[(np.abs(self.pressures-P_b_next)<0.01*P_b_next)].mean()
                T_b = T_b + h_b*L_b
                self.pressures[self.pressures<P_b_next] = P_b_next
                self.temperatures[self.pressures<P_b_next] = self.temperatures[self.pressures<P_b_next] - L_b * (self.above_part[self.pressures<P_b_next]-h_b)
                #print('temp pixels', b_i,' corr |', len(temperatures[~np.isnan(temperatures)]))
                #print(h_b, T_b)
            else:
                break
        
        #translation to Celsius degrees
        self.temperatures = self.temperatures - 273.15
        #mask = np.isnan(self.temperatures)
        #self.temperatures[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), self.temperatures[~mask])
        #self.temperatures[self.heights<self.sea_levels[i]]=np.nan
        if verbose:
            print('Temperature and pressure calcs', round(time.time()-start_time,1), 'seconds')
            
        self.min_pressure = self.pressures[~np.isnan(self.pressures)].min()
        self.max_pressure = self.pressures[~np.isnan(self.pressures)].max()
        self.min_temperature = self.temperatures[~np.isnan(self.temperatures)].min()
        self.max_temperature = self.temperatures[~np.isnan(self.temperatures)].max()
        
    def calc_precipitations(self, i, verbose=False):
        
        start_time=time.time()
        land_share = 1-self.sea_shares[i]/100
        evaporation_of_1sqm_of_water = 1305/1000
        evaporation_of_1sqm_of_land = 534/1000
        total_precipitations_new = evaporation_of_1sqm_of_water*self.full_square*(1-land_share) + evaporation_of_1sqm_of_land*self.full_square*land_share
        total_precipation_land_new = evaporation_of_1sqm_of_land*self.full_square*land_share + 0.08*evaporation_of_1sqm_of_water*self.full_square*(1-land_share)
        precipitations_of_1sqm_of_land = total_precipation_land_new / (self.full_square*land_share)
        total_precipation_water_new = (1-0.08)*evaporation_of_1sqm_of_water*self.full_square*(1-land_share)
        #precipitations_of_1sqm_of_water = total_precipation_water_new / (full_square*(1-land_share))
        #print(round(precipitations_of_1sqm_of_water*1000), round(precipitations_of_1sqm_of_land*1000))    
        #average_precipation=precipitations_of_1sqm_of_land*(-20/(temp_avg-273.15-36.5))*1000
            #average_precipation=precipitations_of_1sqm_of_land*((temp_avg-273.15)/100+0.8)*1000
            #average_precipation = 0 if average_precipation<0 else average_precipation
        self.average_precipation = precipitations_of_1sqm_of_land*1000*pws(self.temp_avg-273.15)/pws(15)
        #https://www.engineeringtoolbox.com/water-vapor-saturation-pressure-air-d_689.html


        rem_coef = 1.5/(self.remoteness/1e6+1.5)
        lat_coef = np.zeros(self.lats.shape)
        lat_coef[np.abs(self.lats)<20] = 3-2.2/20*np.abs(self.lats)[np.abs(self.lats)<20]
        lat_coef[(np.abs(self.lats)>=20)&(np.abs(self.lats)<60)] = 0.8
        lat_coef[np.abs(self.lats)>=60] = 0.8-0.8/30*(np.abs(self.lats)[np.abs(self.lats)>=60]-60)
        max_pressure = self.pressures[~np.isnan(self.pressures)].max()
        min_pressure = self.pressures[~np.isnan(self.pressures)].min()/1000
        pressure_coef = 0.2/(1.18-(self.pressures-min_pressure)/(max_pressure-min_pressure))

        self.precipitations = rem_coef * lat_coef * pressure_coef
        self.precipitations = self.precipitations / ((self.precipitations*self.s)[~np.isnan(self.precipitations)].sum()/self.s[~np.isnan(self.precipitations)].sum())
        self.precipitations = self.average_precipation * self.precipitations
        if verbose:
            print('Precipitations calcs', round(time.time()-start_time,1), 'seconds')
        
    def calc_climate(self, climate_prediction_type, verbose=False):
        
        start_time=time.time()
        self.df = pd.DataFrame()
        self.df['temperature'] = self.temperatures[~np.isnan(self.temperatures)]
        self.df['precipitations'] = self.precipitations[~np.isnan(self.precipitations)]
        self.df['lat'] = -self.lats[~np.isnan(self.temperatures)]
        self.df['long'] = self.lons[~np.isnan(self.temperatures)]
        self.df['s'] = self.s[~np.isnan(self.temperatures)]
        self.df['remoteness'] = self.remoteness[~np.isnan(self.temperatures)]
        self.df['large_land'] = self.large_land[~np.isnan(self.temperatures)]
        self.df['height'] = self.above_part[~np.isnan(self.temperatures)]*self.g_0/g_0_earth

        if climate_prediction_type == 'neigh':
            self.df['climate'] = neigh.predict(self.df[['temperature','precipitations']])
            self.df.loc[self.df[self.df.temperature>30].index,'climate']=8 # venusian hammam
            self.climate = pd.crosstab(self.df.lat, self.df.long, self.df.climate, aggfunc=np.mean, dropna=False)#.values
            temp_df = pd.DataFrame(index=-self.lats[:,0], columns=self.lons[0,:])
            temp_df.loc[self.climate.index, self.climate.columns] = self.climate
            self.climate = temp_df.sort_index(ascending=False).fillna(np.nan).values
            self.climate = np.flipud(self.climate)
        elif climate_prediction_type == 'neigh_detailed':
            cols = ['temperature','precipitations', 'height']#, 'remoteness']
            scaler_dict['precipitations'] = scaler_dict['precip']
            X = np.zeros(self.df[cols].shape)
            for i, col in enumerate(cols):
                X[:,i] = (self.df[col]-scaler_dict[col]['mean']) / scaler_dict[col]['std'] * scaler_dict[col]['coef']
            self.df['climate'] = neigh_detailed.predict(X)
            self.df.loc[self.df[self.df.temperature>30].index,'climate']=31 # venusian hammam
            self.climate = pd.crosstab(self.df.lat, self.df.long, self.df.climate, aggfunc=np.mean, dropna=False)#.values
            temp_df = pd.DataFrame(index=-self.lats[:,0], columns=self.lons[0,:])
            temp_df.loc[self.climate.index, self.climate.columns] = self.climate
            self.climate = temp_df.sort_index(ascending=False).fillna(np.nan).values
            self.climate = np.flipud(self.climate)
            #self.climate[(self.temperatures>30)&(self.precipitations>2500)] = 1
        else:
            self.climate = np.zeros(self.temperatures.shape)
            self.climate[(self.precipitations <= 400/48*(self.temperatures + 18))] = 3 # desert - 3
            self.climate[(self.precipitations > 400/48*(self.temperatures + 18))&(self.precipitations <= 800/51*(self.temperatures + 21))] = 4 # steppe - 4
            self.climate[(self.precipitations > 800/51*(self.temperatures + 21))&(self.temperatures <= -13)] = 7 # polar - 7
            self.climate[(self.precipitations > 800/51*(self.temperatures + 21))&(self.temperatures > -13)&(self.temperatures <= 10)] = 6 # cold - 6
            self.climate[(self.precipitations > 800/51*(self.temperatures + 21))&(self.temperatures > 10)&(self.temperatures <= 22)] = 5 # temperate - 5
            self.climate[(self.precipitations > 800/51*(self.temperatures + 21))&(self.temperatures > 22)&(self.precipitations <= 1700)] = 2 # savanna - 2
            self.climate[(self.precipitations > 800/51*(self.temperatures + 21))&(self.temperatures > 22)&(self.precipitations > 1700)] = 1 # rainforest - 1
            self.climate[self.temperatures>30] = 8 # venusian hammam
            self.climate[self.climate==0] = np.nan
            self.df['climate'] = self.climate[~np.isnan(self.temperatures)]
        
        self.climate[np.isnan(self.climate)] = 0

        self.climate_zones_shares = [self.df[self.df.climate==i]['s'].sum() / self.df['s'].sum() for i in range(0, 9)]
        self.climate_zones_areas = [self.df[self.df.climate==i]['s'].sum() for i in range(0, 9)]
            
        self.climate_zones_areas[0] = self.full_square - sum(self.climate_zones_areas)
        self.climate_zones_shares[0] = self.climate_zones_areas[0]/(self.full_square-self.climate_zones_areas[0])        
        
        r_earth = 6371000
        full_square_earth = 4*math.pi*r_earth**2
        earth_climate_zones_areas = [full_square_earth * 0.71]+[i * full_square_earth * 0.29 for i in [0.078880, 0.114450, 
                                                                                                       0.204329, 0.116920, 
                                                                                                       0.108708, 0.231699, 
                                                                                                       0.145013, 0]]
        self.climate_zones_share_of_earths = ["{:5.1f}".format(self.climate_zones_areas[i]/earth_climate_zones_areas[i]*100)+" % of Earth's zone" for i in range(len(self.climate_zones_areas))]
        self.climate_zones_share_of_earths[-1] = ''
        self.climate_zones_areas = [' '+str(round(i/1e12,1)) for i in self.climate_zones_areas]
        self.climate_zones_shares = ["{:5.1f}".format(i*100) for i in self.climate_zones_shares]
        self.labels_text = ['Seas & Oceans', 'Tropical Rainforest', 'Tropical Savannah', 'Arid (Desert)', 
                                 'Semi-arid (Steppe)', 'Temperate','Continental Cold',
                                 'Tundra & Polar', 'Venusian hammam']
        self.labels_text = [self.labels_text[i]+'\n'+self.climate_zones_areas[i]+' mln sq km'+'\n'+self.climate_zones_shares[i]+' % of '+self.planet+' land'+'\n'+self.climate_zones_share_of_earths[i]  for i in range(len(self.labels_text))]
        if verbose:
            print('Climate zones calcs', round(time.time()-start_time,1), 'seconds')
        
    def calc_pop_den(self, verbose=False):
        
        start_time=time.time()
        self.df['remoteness_rounded'] = self.df['remoteness']//35183.0*35183.0
        self.df['pop_den_pred'] = 0

        for i in range(1, 8):
            self.df.loc[(self.df.climate==i)&(self.df.large_land==1), 
                        'pop_den_pred'] = self.df.loc[(self.df.climate==i)&(self.df.large_land==1), 
                                                      'remoteness_rounded'].map(pop_den_dict[i])    

        self.df['pop_pred'] = self.df['pop_den_pred'] * self.df['s']
        self.total_population = round(self.df['pop_pred'].sum()/1e6)

        self.population_density = pd.crosstab(self.df.lat, self.df.long, self.df['pop_den_pred'].fillna(0), aggfunc=np.mean, dropna=False)#.values
        temp_df = pd.DataFrame(index=-self.lats[:,0], columns=self.lons[0,:])
        temp_df.loc[self.population_density.index, self.population_density.columns] = self.population_density
        self.population_density = temp_df.sort_index(ascending=False).fillna(np.nan).values
        self.population_density = np.flipud(self.population_density)
        if verbose:
            print('Population density calcs', round(time.time()-start_time,1), 'seconds')
    
    def calc_radiation(self, verbose=False):
        
        start_time=time.time()
        self.radiation = np.empty(self.pressures.shape)
        self.radiation[:] = np.nan
        for r, pres in enumerate(radiation_pressure_levels):            
            if r>0:
                pres_m1 = radiation_pressure_levels[r-1]
                rad = radiation_levels[r]
                rad_m1 = radiation_levels[r-1]
                self.radiation[(self.pressures>=pres_m1)&(self.pressures<=pres)] = rad_m1 + (rad - rad_m1)*(self.pressures[(self.pressures>=pres_m1)&(self.pressures<=pres)]-pres_m1)/(pres-pres_m1)
                
        self.radiation[(np.isnan(self.radiation))&(~np.isnan(self.pressures))] = radiation_levels[-1]
        
        self.min_radiation = self.radiation[~np.isnan(self.radiation)].min()
        self.max_radiation = self.radiation[~np.isnan(self.radiation)].max()
        self.mean_radiation = self.radiation[~np.isnan(self.radiation)].mean()
        
        if verbose:
            print('Radiation calcs', round(time.time()-start_time,1), 'seconds')
        
        
    def make_maps(self, i=None, sea_share=None, maps_list=['altitude', 'temperature', 'pressure', 'precipitations', 
                                                         'climate', 'pop_den', 'radiation', 'remoteness'], 
                  solar_irradiance=1361, sea_level_pressure=101325, purple=False,
                  climate_prediction_type='notneigh', read_file=False, verbose=True, title='long', caption=True,
                  projection='eck4', angle_lon=0, angle_lat=0, view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        if i is None and sea_share is not None:
            i = np.abs(np.array(self.sea_shares) - sea_share).argmin()
        elif i is None and sea_share is None:
            print('!!! Please specify sea share or i, but not both at the same time!')
        elif i is not None and sea_share is not None:
            print('!!! Please specify sea share or i, but not both at the same time! Default sea share = 50 %')
            sea_share=50
            i = np.abs(np.array(self.sea_shares) - sea_share).argmin()
        
        #calculations  
        if 'remoteness' in maps_list or 'pop_den' in maps_list or 'precipitations' in maps_list or 'climate' in maps_list or 'natural_colors' in maps_list or 'temperature' in maps_list or 'pressure' in maps_list or 'radiation' in maps_list or 'title' in maps_list:
            self.calc_remoteness(i, verbose=verbose)
            
        if 'temperature' in maps_list or 'pressure' in maps_list or 'climate' in maps_list or 'natural_colors' in maps_list or 'radiation' in maps_list or 'pop_den' in maps_list or 'precipitations' in maps_list or 'title' in maps_list:
            self.calc_temperature_pressure(i, solar_irradiance=solar_irradiance, sea_level_pressure=sea_level_pressure, 
                                           verbose=verbose)
            
        if 'precipitations' in maps_list or 'climate' in maps_list or 'natural_colors' in maps_list or 'pop_den' in maps_list or 'title' in maps_list:
            self.calc_precipitations(i, verbose=verbose)
            
        if 'climate' in maps_list or 'pop_den' in maps_list or 'title' in maps_list or 'natural_colors' in maps_list:
            self.calc_climate(climate_prediction_type=climate_prediction_type, verbose=verbose)
            
        if 'pop_den' in maps_list or 'title' in maps_list:
            self.calc_pop_den(verbose=verbose)
            
        if ('radiation' in maps_list or 'title' in maps_list) and self.planet=='Mars':
            self.calc_radiation(verbose=verbose)
        
        
        maps_dict = {}
        #maps
        
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        
        if len(maps_list)>0:
            if (projection+'_'+view+'_'+str(caption)+'_'+facecolor not in self.hs_dict) or projection=='ortho':
                self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor] = self.make_hillshade_map(i=i, 
                                                                                                      projection=projection, 
                                                                                        title_text='', verbose=verbose, 
                                                                                        angle_lon=angle_lon, 
                                                                                        angle_lat=angle_lat, 
                                                                                        read_file=True, caption=caption, 
                                                                                        facecolor=facecolor, view=view)
            if (projection+'_'+view+'_'+str(caption)+'_'+facecolor not in self.shade_dict) or projection=='ortho':
                self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor] = self.make_shade_map(i=i, 
                                                                                                      projection=projection, 
                                                                                        title_text='', verbose=verbose, 
                                                                                        angle_lon=angle_lon, 
                                                                                        angle_lat=angle_lat, 
                                                                                        read_file=True, caption=caption, 
                                                                                        facecolor=facecolor, view=view)
        
        if 'altitude' in maps_list:
            img_altitude = self.make_altitude_map(i, read_file=read_file, verbose=verbose, title=title,
                                                 projection=projection, angle_lon=angle_lon, angle_lat=angle_lat, 
                                                 show_hillshade=show_hillshade, show_shade=show_shade, caption=caption, view=view)
            maps_dict['altitude'] = img_altitude
        
        if 'temperature' in maps_list:
            img_temperature = self.make_temperature_map(i, solar_irradiance=solar_irradiance, verbose=verbose, title=title, 
                                                        sea_level_pressure=sea_level_pressure, read_file=read_file,
                                                        projection=projection, angle_lon=angle_lon, angle_lat=angle_lat, 
                                                        show_hillshade=show_hillshade, show_shade=show_shade, caption=caption, view=view)
            maps_dict['temperature'] = img_temperature
            
        if 'remoteness' in maps_list:            
            img_remoteness = self.make_remoteness_map(i, read_file=read_file, verbose=verbose, title=title,
                                                      projection=projection, angle_lon=angle_lon, angle_lat=angle_lat, 
                                                     show_hillshade=show_hillshade, show_shade=show_shade, caption=caption, view=view)
            maps_dict['remoteness'] = img_remoteness
            
        if 'pressure' in maps_list:
            img_pressure = self.make_pressure_map(i, solar_irradiance=solar_irradiance, verbose=verbose, title=title, 
                                                  sea_level_pressure=sea_level_pressure, read_file=read_file,
                                                  projection=projection, angle_lon=angle_lon, angle_lat=angle_lat, 
                                                  show_hillshade=show_hillshade, show_shade=show_shade, caption=caption, view=view)
            maps_dict['pressure'] = img_pressure
            
        if 'precipitations' in maps_list:
            img_precipitations = self.make_precipitations_map(i, solar_irradiance=solar_irradiance, verbose=verbose, title=title, 
                                                              sea_level_pressure=sea_level_pressure, read_file=read_file,
                                                              projection=projection, angle_lon=angle_lon, angle_lat=angle_lat, 
                                                              show_hillshade=show_hillshade, show_shade=show_shade, caption=caption, view=view)
            maps_dict['precipitations'] = img_precipitations
            
        if 'climate' in maps_list:
            img_climate = self.make_climate_map(i, solar_irradiance=solar_irradiance, verbose=verbose, title=title, 
                                                sea_level_pressure=sea_level_pressure, read_file=read_file,
                                                projection=projection, angle_lon=angle_lon, angle_lat=angle_lat, 
                                                show_hillshade=show_hillshade, show_shade=show_shade, caption=caption, view=view)
            maps_dict['climate'] = img_climate
            
        if 'natural_colors' in maps_list:
            img_natural_colors = self.make_natural_colors_map(i, solar_irradiance=solar_irradiance, verbose=verbose, title=title, 
                                                              sea_level_pressure=sea_level_pressure, read_file=read_file,
                                                              projection=projection, angle_lon=angle_lon, angle_lat=angle_lat,
                                                              view=view, show_hillshade=show_hillshade, show_shade=show_shade, 
                                                              caption=caption, purple=purple)
            maps_dict['natural_colors'] = img_natural_colors
            
        if 'pop_den' in maps_list:
            img_population_density = self.make_pop_den_map(i, solar_irradiance=solar_irradiance, verbose=verbose, title=title, 
                                                           sea_level_pressure=sea_level_pressure, read_file=read_file,
                                                           projection=projection, angle_lon=angle_lon, angle_lat=angle_lat, 
                                                           show_hillshade=show_hillshade, show_shade=show_shade, caption=caption, view=view)
            maps_dict['pop_den'] = img_population_density
            
        if 'radiation' in maps_list and self.planet=='Mars':
            img_radiation = self.make_radiation_map(i, solar_irradiance=solar_irradiance, verbose=verbose, title=title, 
                                                    sea_level_pressure=sea_level_pressure, read_file=read_file,
                                                    projection=projection, angle_lon=angle_lon, angle_lat=angle_lat, 
                                                    show_hillshade=show_hillshade, show_shade=show_shade, caption=caption, view=view)  
            maps_dict['radiation'] = img_radiation
                   
            
        return maps_dict
        
        
    def make_maps_mosaic(self, i=None, sea_share=None, maps_matrix = [['altitude', 'climate', 'title'], 
                                                                    ['pressure', 'temperature', 'pop_den'], 
                                                                    ['remoteness', 'precipitations','radiation']], 
                         solar_irradiance=1361, sea_level_pressure=101325, purple=False,
                         climate_prediction_type='notneigh', read_file=False, verbose=True, caption=False,
                         projection='eck4', angle_lon=0, angle_lat=0, title='short', view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        if i is None and sea_share is not None:
            i = np.abs(np.array(self.sea_shares) - sea_share).argmin()
        elif i is None and sea_share is None:
            print('!!! Please specify sea share or i, but not both at the same time!')
        elif i is not None and sea_share is not None:
            print('!!! Please specify sea share or i, but not both at the same time! Default sea share = 50 %')
            sea_share=50
            i = np.abs(np.array(self.sea_shares) - sea_share).argmin()
            
            
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        maps_list = pd.DataFrame(maps_matrix).values.reshape(-1).tolist()
        mosaic_shape = pd.DataFrame(maps_matrix).values.shape
        maps_matrix = pd.DataFrame(maps_matrix).values
        
        
        maps_dict = self.make_maps(i=i, maps_list=maps_list, solar_irradiance=solar_irradiance, 
                                   sea_level_pressure=sea_level_pressure, climate_prediction_type=climate_prediction_type, 
                                   read_file=True, verbose=verbose, title=title, purple=purple,
                                   projection=projection, angle_lon=angle_lon, angle_lat=angle_lat, view=view, 
                                   show_hillshade=show_hillshade, show_shade=show_shade, caption=caption)
        
        
        
        #frames_sizes = [maps_dict[i].shape for i in maps_dict if i!='climate' and maps_dict[i] is not None]
        frames_sizes = [maps_dict[i].shape for i in maps_dict if maps_dict[i] is not None]
        max_width = max([i[1] for i in frames_sizes])
        max_height = max([i[0] for i in frames_sizes])
        widths = [max([maps_dict[maps_matrix[i,j]].shape[1] for i in range(maps_matrix.shape[0]) \
                       if (maps_matrix[i,j]!='title' and maps_matrix[i,j]!=None and maps_matrix[i,j]!='radiation' and self.planet!='Mars') or \
                       (maps_matrix[i,j]!='title' and maps_matrix[i,j]!=None and self.planet=='Mars')]) for j in range(maps_matrix.shape[1])]
        heights = [max([maps_dict[maps_matrix[i,j]].shape[0] for j in range(maps_matrix.shape[1]) \
                       if (maps_matrix[i,j]!='title' and maps_matrix[i,j]!=None and maps_matrix[i,j]!='radiation' and self.planet!='Mars') or \
                       (maps_matrix[i,j]!='title' and maps_matrix[i,j]!=None and self.planet=='Mars')]) for i in range(maps_matrix.shape[0])]
        #print(max_width, max_height)  
        #print('widths', widths)
        #print('heights', heights)
        if 'title' in maps_list:
            if widths[np.argwhere(maps_matrix=='title')[0][1]] < 2000 * self.dpi/150:            
                widths[np.argwhere(maps_matrix=='title')[0][1]] = int(2000 * self.dpi/150)
            title_size = (heights[np.argwhere(maps_matrix=='title')[0][0]], widths[np.argwhere(maps_matrix=='title')[0][1]])
           
            maps_dict['title'] = self.title_frame(maps_list=maps_list, maps_dict=maps_dict, i=i, w=w, ss=ss, sl=sl, 
                                                  sea_level_pressure=sea_level_pressure, solar_irradiance=solar_irradiance,
                                                  size = title_size)
        
        #mosaic = np.ones((max_width*mosaic_shape[0], max_height*mosaic_shape[1], 3))*255
        mosaic = np.ones((np.sum(heights), np.sum(widths), 3))*255
        #print(mosaic.shape)
        
        for k in range(mosaic_shape[0]):
            for j in range(mosaic_shape[1]):
                if maps_matrix[k, j] is not None and maps_matrix[k,j] in maps_dict:
                    frame = maps_dict[maps_matrix[k, j]]
                    if frame is not None:
                        mosaic[int(np.sum(heights[:k+1])-frame.shape[0]) : int(np.sum(heights[:k+1])),
                               int(np.sum(widths[:j])) : int(np.sum(widths[:j])+frame.shape[1]), :] = frame
                
        if read_file==True:
            return mosaic
        else:
            directory = 'out_'+self.planet.lower()+'_maps'
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            fn = directory+'/'+self.planet+'_maps_mosaic_n_'+'%04d'%i+'_angle_'+str(int(angle_lon))+\
                    '_water_volume_'+str(w)+'_sea_level_'+str(sl)+'_sea_share_'+str(ss)+'_insol_'+\
                    str(round(solar_irradiance))+'_pressure_'+str(round(sea_level_pressure))+'_quality_'+\
                    str(self.quality)+'_'+projection+'_title_'+title+'.jpg'
            cv.imwrite(fn, mosaic)
        
        
    def make_videos(self, maps_list=['altitude', 'temperature', 'pressure', 'precipitations', 
                                     'climate', 'pop_den', 'radiation', 'remoteness'], 
                    i_list = [], sea_level_pressure_list=[], solar_irradiance_list=[], purple=False,
                    angle_lon_list = [], angle_lat_list = [], gaps=20, verbose=True, fps=60, projection='eck4', title='long', 
                    view='lines_toponyms', show_hillshade=True, show_shade=False, caption=True):
        
        self.num_cores = multiprocessing.cpu_count()
        
        if i_list == []:
            i_list = [i for i in range(len(self.water_volumes))]
        if angle_lon_list==[] and projection in ['eck4', 'cyl']:
            angle_lon_list=[0]*len(i_list)
        elif angle_lon_list==[] and projection=='ortho':
            angle_lon_list = [-360*i/((len(i_list)-1)/1) for i in range(len(i_list))]
        if angle_lat_list==[]:
            angle_lat_list=[0]*len(i_list)
        if sea_level_pressure_list==[]:
            sea_level_pressure_list = [101325]*len(i_list)
            print("Sea level pressures list is not specified. Default sea level pressure equals to Earth's - 101 325 Pa")
        if solar_irradiance_list==[]:
            solar_irradiance_list = [1361]*len(i_list)
            print("Solar irradiances list is not specified. Default solar irradiance equals to Earth's - 1 361 W/m2")
        if self.planet!='Mars' and 'radiation' in maps_list:
            maps_list.remove('radiation')
        
        directory = 'out_'+self.planet.lower()+'_maps_videos'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        out_dict = {}
        size_dict = {}
        out_file_dict = {}
        for m in maps_list:
            out_dict[m] = None
            fn = directory+'/terraforming_'+self.planet+'_'+m+'_map_'+self.quality+'_'+str(fps)+'fps_'+projection+'_title_'+title
            if os.path.exists(fn+'.mp4'):
                suf=0
                while os.path.exists(fn+'_'+str(suf)+'.mp4'):
                    suf = suf+1
                out_file_dict[m] = fn+'_'+str(suf)+'.mp4'
            else:
                out_file_dict[m] = fn+'.mp4'
            
        j_range = math.ceil(len(i_list)/(self.num_cores-1))
        for j in range(j_range):
            l = j*(self.num_cores-1)
            u = (j+1)*(self.num_cores-1) if (j+1)*(self.num_cores-1)<len(i_list) else len(i_list)
            if verbose:
                print(time.ctime(),' | chunk', j, '| from', l, 'to', u)
            img_dicts = Parallel(n_jobs=self.num_cores-1)(delayed(self.make_maps)(i=i_list[k], 
                                                                                  sea_level_pressure=sea_level_pressure_list[k],
                                                                                  solar_irradiance=solar_irradiance_list[k],
                                                                                  angle_lon = angle_lon_list[k],
                                                                                  angle_lat = angle_lat_list[k],
                                                                                  verbose=False, title='long', 
                                                                                  projection=projection,
                                                                                  maps_list=maps_list,
                                                                                  read_file=True, view=view,
                                                                                  show_hillshade=show_hillshade, show_shade=show_shade,
                                                                                  caption=caption, purple=purple) for k in range(l, u, 1))
            
            for m in maps_list:
                if out_dict[m] is None:
                    size_dict[m] = (img_dicts[0][m].shape[1], img_dicts[0][m].shape[0])
                    print(m, ' size ',size_dict[m])
                    out_dict[m] = cv.VideoWriter(out_file_dict[m], cv.VideoWriter_fourcc(*'mp4v'), fps, 
                                                 size_dict[m], isColor=True)
            if l==0:
                for m in maps_list:
                    if img_dicts[0][m][0,0,0]<100:
                        image = np.ones((size_dict[m][1], size_dict[m][0], 3), dtype='uint8')*0
                    else:
                        image = np.ones((size_dict[m][1], size_dict[m][0], 3), dtype='uint8')*255
                    for g in range(gaps):
                        out_dict[m].write(cv.addWeighted(image, (gaps-g)/gaps, 
                                                         img_dicts[0][m][:size_dict[m][1],:size_dict[m][0],:], g/gaps, 0))
            for img_dict in img_dicts:
                for m in maps_list:
                    out_dict[m].write(self.upscale_frame(img_dict[m][:size_dict[m][1],:size_dict[m][0],:], size_dict[m]))
            
            if u==len(i_list):
                for m in maps_list:
                    if img_dict[m][0,0,0]<100:
                        image = np.ones((size_dict[m][1], size_dict[m][0], 3), dtype='uint8')*0
                    else:
                        image = np.ones((size_dict[m][1], size_dict[m][0], 3), dtype='uint8')*255
                    for g in range(gaps):
                        out_dict[m].write(cv.addWeighted(self.upscale_frame(img_dict[m][:size_dict[m][1],:size_dict[m][0],:], 
                                                                            size_dict[m]), 
                                                         (gaps-g)/gaps, image, g/gaps, 0))
            
            del img_dict, img_dicts
            gc.collect()
        
        for m in maps_list:
            out_dict[m].release()
        cv.destroyAllWindows()
        
    def make_videos_mosaic(self, i_list = [], maps_matrix=[['altitude', 'climate', 'title'], 
                                                           ['pressure', 'temperature', 'pop_den'], 
                                                           ['remoteness', 'precipitations','radiation']], 
                          sea_level_pressure_list=[], solar_irradiance_list=[], purple=False,
                          angle_lon_list = [], angle_lat_list = [], gaps=20, verbose=True, fps=60, projection='eck4', 
                          title='short', view='lines_toponyms', show_hillshade=True, show_shade=False, caption=False):
        
        self.num_cores = multiprocessing.cpu_count()
        
        maps_matrix_shape = pd.DataFrame(maps_matrix).values.shape
        if maps_matrix_shape[0]==2 and self.quality=='4K':
            self.dpi=275
        elif self.quality == '8K':
            print('Video mosaic can be made only for 4K (max 2*2) and FHD (max 3*3)')
            return None
        
        #if 'title' in pd.DataFrame(maps_matrix).values.reshape(-1).tolist():
        #    title='short'
        #else:
        #    title='long'
        
        
        if i_list == []:
            i_list = [i for i in range(len(self.water_volumes))]
        if angle_lon_list==[] and projection in ['eck4', 'cyl']:
            angle_lon_list=[0]*len(i_list)
        elif angle_lon_list==[] and projection=='ortho':
            angle_lon_list = [-360*i/((len(i_list)-1)/1) for i in range(len(i_list))]
        if angle_lat_list==[]:
            angle_lat_list=[0]*len(i_list)
        if sea_level_pressure_list==[]:
            sea_level_pressure_list = [101325]*len(i_list)
            print("Sea level pressures list is not specified. Default sea level pressure equals to Earth's - 101 325 Pa")
        if solar_irradiance_list==[]:
            solar_irradiance_list = [1361]*len(i_list)
            print("Solar irradiances list is not specified. Default solar irradiance equals to Earth's - 1 361 W/m2")
        
        directory = 'out_'+self.planet.lower()+'_maps_videos'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        out = None
        fn = directory+'/terraforming_'+self.planet+'_mosaic_'+'_map_'+self.quality+'_'+str(fps)+'fps_'+projection+'_title_'+title
        if os.path.exists(fn+'.mp4'):
            suf=0
            while os.path.exists(fn+'_'+str(suf)+'.mp4'):
                suf = suf+1
            out_file = fn+'_'+str(suf)+'.mp4'
        else:
            out_file = fn+'.mp4'
            
        j_range = math.ceil(len(i_list)/(self.num_cores-1))
        for j in range(j_range):
            l = j*(self.num_cores-1)
            u = (j+1)*(self.num_cores-1) if (j+1)*(self.num_cores-1)<len(i_list) else len(i_list)
            if verbose:
                print(time.ctime(),' | chunk', j, '| from', l, 'to', u)
            img_list = Parallel(n_jobs=self.num_cores-1)(delayed(self.make_maps_mosaic)(i=i_list[k], 
                                                                                        sea_level_pressure=sea_level_pressure_list[k],
                                                                                        solar_irradiance=solar_irradiance_list[k],
                                                                                        angle_lon = angle_lon_list[k],
                                                                                        angle_lat = angle_lat_list[k],
                                                                                        verbose=False, title=title, 
                                                                                        projection=projection,
                                                                                        maps_matrix=maps_matrix,
                                                                                        read_file=True, view=view,
                                                                                        show_hillshade=show_hillshade, show_shade=show_shade,
                                                                                        caption=caption, purple=purple) for k in range(l, u, 1))
            
            if out is None:
                size = (img_list[0].shape[1], img_list[0].shape[0])
                print('size ',size)
                out = cv.VideoWriter(out_file, cv.VideoWriter_fourcc(*'mp4v'), fps, size, isColor=True)
            if l==0:                
                if img_list[0][0,0,0]<100:
                    image = np.ones((size[1], size[0], 3), dtype='uint8')*0
                else:
                    image = np.ones((size[1], size[0], 3), dtype='uint8')*255

                for g in range(gaps):
                    out.write(cv.addWeighted(image, (gaps-g)/gaps, 
                                      img_list[0][:size[1],:size[0],:].astype(np.uint8), g/gaps, 0))
            for img in img_list:
                out.write(self.upscale_frame(img[:size[1],:size[0],:].astype(np.uint8), size))
            
            if u==len(i_list):
                if img[0,0,0]<100:
                    image = np.ones((size[1], size[0], 3), dtype='uint8')*0
                else:
                    image = np.ones((size[1], size[0], 3), dtype='uint8')*255
                for g in range(gaps):
                    out.write(cv.addWeighted(self.upscale_frame(img[:size[1],:size[0],:].astype(np.uint8),size), 
                                             (gaps-g)/gaps, image, g/gaps, 0))
            
            del img_list
            gc.collect()
        
        out.release()
        cv.destroyAllWindows()
        
    def add_toponyms_eck4(self, w, font_color):
        
        if self.planet=='Mars':
            
            plt.annotate('Olympus \nMons', xy=(0.1, 0.63), xycoords='axes fraction', color=font_color)
            plt.annotate('Tharsis \nMontes', xy=(0.16, 0.53), xycoords='axes fraction', color=font_color)
            plt.annotate('Alba \nMons', xy=(0.19, 0.76), xycoords='axes fraction', color=font_color)
            plt.annotate('Elysium \nMons', xy=(0.86, 0.67), xycoords='axes fraction', color=font_color)

            plt.annotate('Hellas \nPlanitia', xy=(0.64, 0.18), xycoords='axes fraction', color=font_color) if w <1*1e18 else plt.annotate('Hellas \nMare', xy=(0.64, 0.18), xycoords='axes fraction', color=font_color)
            plt.annotate('Argyre \nPlanitia', xy=(0.37, 0.12), xycoords='axes fraction', color=font_color) if w <62*1e18 else plt.annotate('Argyre \nMare', xy=(0.37, 0.12), xycoords='axes fraction', color=font_color)

            plt.annotate('Arcadia \nPlanitia', xy=(0.09, 0.8), xycoords='axes fraction', color=font_color) if w <16*1e18 else plt.annotate('Arcadia \nMare', xy=(0.09, 0.8), xycoords='axes fraction', color=font_color)
            plt.annotate('Amazonis \nPlanitia', xy=(0.02, 0.63), xycoords='axes fraction', color=font_color) if w <20*1e18 else plt.annotate('Amazonis \nMare', xy=(0.02, 0.63), xycoords='axes fraction', color=font_color)
            plt.annotate('Acidalia \nPlanitia', xy=(0.45, 0.83), xycoords='axes fraction', color=font_color) if w <6*1e18 else plt.annotate('Acidalia \nMare', xy=(0.45, 0.83), xycoords='axes fraction', color=font_color)
            plt.annotate('Chryse \nPlanitia', xy=(0.38, 0.7), xycoords='axes fraction', color=font_color) if w <34*1e18 else plt.annotate('Chryse \nMare', xy=(0.38, 0.7), xycoords='axes fraction', color=font_color)
            plt.annotate('Utopia \nPlanitia', xy=(0.78, 0.78), xycoords='axes fraction', color=font_color) if w <8*1e18 else plt.annotate('Utopia \nMare', xy=(0.78, 0.78), xycoords='axes fraction', color=font_color)
            plt.annotate('Isidis \nPlanitia', xy=(0.72, 0.6), xycoords='axes fraction', color=font_color) if w <25*1e18 else plt.annotate('Isidis \nMare', xy=(0.72, 0.6), xycoords='axes fraction', color=font_color)

            plt.annotate('Arabia \nTerra', xy=(0.5, 0.6), xycoords='axes fraction', color=font_color) if w <168*1e18 else plt.annotate('Arabia \nMare', xy=(0.5, 0.6), xycoords='axes fraction', color=font_color)
            plt.annotate('Noachis \nTerra', xy=(0.53, 0.18), xycoords='axes fraction', color=font_color)
            plt.annotate('Promethei \nTerra', xy=(0.72, 0.06), xycoords='axes fraction', color=font_color)# if w <300*1e18 else plt.annotate('Promethei \n Islands', xy=(0.8, 0.13), xycoords='axes fraction', color=font_color)
            plt.annotate('Aonia \nTerra', xy=(0.27, 0.10), xycoords='axes fraction', color=font_color)

            plt.annotate('Terra \nCimmeria', xy=(0.81, 0.06), xycoords='axes fraction', color=font_color)
            plt.annotate('Tyrrhena \nTerra', xy=(0.72, 0.40), xycoords='axes fraction', color=font_color)
            plt.annotate('Terra \nSabaea', xy=(0.6, 0.40), xycoords='axes fraction', color=font_color)

            plt.annotate('Terra \nSirenum', xy=(0.12, 0.13), xycoords='axes fraction', color=font_color)
            plt.annotate('Solis \nPlanum', xy=(0.25, 0.30), xycoords='axes fraction', color=font_color)
            plt.annotate('Valles Marineris', xy=(0.26, 0.4), xycoords='axes fraction', color=font_color) if w <24*1e18 else plt.annotate('Marineris Channels', xy=(0.26, 0.4), xycoords='axes fraction', color=font_color)

            plt.annotate('Vastitas Borealis', xy=(0.45, 0.94), xycoords='axes fraction', color=font_color) if w <10*1e18 else plt.annotate('Borealis Oceanus', xy=(0.45, 0.94), xycoords='axes fraction', color=font_color)
            plt.annotate('Planum Australe', xy=(0.45, 0.02), xycoords='axes fraction', color=font_color)
            
        elif self.planet=='Venus':
            
            plt.annotate('Aphrodite Terra', xy=(0.75, 0.4), xycoords='axes fraction', color=font_color)
            plt.annotate('Lada \n Terra', xy=(0.48, 0.05), xycoords='axes fraction', color=font_color)
            plt.annotate('Ishtar Terra', xy=(0.42, 0.9), xycoords='axes fraction', color=font_color) 
            plt.annotate('Lakshmi \n Planum', xy=(0.4, 0.94), xycoords='axes fraction', color=font_color)

            plt.annotate('Alpha \n Regio', xy=(0.5, 0.3), xycoords='axes fraction', color=font_color)
            plt.annotate('Beta \n Regio', xy=(0.28, 0.67), xycoords='axes fraction', color=font_color)
            plt.annotate('Eistla \n Regio', xy=(0.55, 0.57), xycoords='axes fraction', color=font_color)
            plt.annotate('Atla \n Regio', xy=(0.04, 0.52), xycoords='axes fraction', color=font_color)
            plt.annotate('Phoebe \n Regio', xy=(0.28, 0.48), xycoords='axes fraction', color=font_color)
            plt.annotate('Ovda \n Regio', xy=(0.75, 0.48), xycoords='axes fraction', color=font_color)
            plt.annotate('Thetis \n Regio', xy=(0.82, 0.44), xycoords='axes fraction', color=font_color)
            plt.annotate('Themis \nRegio', xy=(0.3, 0.17), xycoords='axes fraction', color=font_color)
            plt.annotate('Artemis \nChasma', xy=(0.82, 0.22), xycoords='axes fraction', color=font_color)
            plt.annotate('Ulfrun \nRegio', xy=(0.11, 0.58), xycoords='axes fraction', color=font_color)

            plt.annotate('Sedna \n Planitia', xy=(0.42, 0.8), xycoords='axes fraction', color=font_color) if w <75*1e18 else plt.annotate('Sedna \n Mare', xy=(0.42, 0.8), xycoords='axes fraction', color=font_color)
            plt.annotate('Guinevere \n Planitia', xy=(0.3, 0.83), xycoords='axes fraction', color=font_color) if w <15*1e18 else plt.annotate('Guinevere \n Mare', xy=(0.3, 0.83), xycoords='axes fraction', color=font_color)
            plt.annotate('Aino \n Planitia', xy=(0.75, 0.15), xycoords='axes fraction', color=font_color) if w <90*1e18 else plt.annotate('Aino \n Mare', xy=(0.75, 0.15), xycoords='axes fraction', color=font_color)
            plt.annotate('Ganiki \n Planitia', xy=(0.07, 0.75), xycoords='axes fraction', color=font_color) if w <120*1e18 else plt.annotate('Ganiki \n Mare', xy=(0.07, 0.75), xycoords='axes fraction', color=font_color)
            plt.annotate('Lavinia \n Planitia', xy=(0.45, 0.15), xycoords='axes fraction', color=font_color) if w <34*1e18 else plt.annotate('Lavinia \n Mare', xy=(0.45, 0.15), xycoords='axes fraction', color=font_color)
            plt.annotate('Helen \n Planitia', xy=(0.18, 0.15), xycoords='axes fraction', color=font_color) if w <60*1e18 else plt.annotate('Helen \n Mare', xy=(0.18, 0.15), xycoords='axes fraction', color=font_color)
            plt.annotate('Niobe \n Planitia', xy=(0.82, 0.62), xycoords='axes fraction', color=font_color) if w <75*1e18 else plt.annotate('Niobe \n Mare', xy=(0.82, 0.62), xycoords='axes fraction', color=font_color)
            plt.annotate('Rusalka \n Planitia', xy=(0.93, 0.50), xycoords='axes fraction', color=font_color) if w <100*1e18 else plt.annotate('Rusalka \n Gulf', xy=(0.93, 0.50), xycoords='axes fraction', color=font_color)
            plt.annotate('Kawelu \n Planitia', xy=(0.18, 0.65), xycoords='axes fraction', color=font_color) if w <120*1e18 else plt.annotate('Kawelu \n Mare', xy=(0.18, 0.65), xycoords='axes fraction', color=font_color)
            plt.annotate('Bereghinya \n Planitia', xy=(0.53, 0.75), xycoords='axes fraction', color=font_color) if w <90*1e18 else plt.annotate('Bereghinya \n Mare', xy=(0.53, 0.75), xycoords='axes fraction', color=font_color)
            plt.annotate('Atalanta \n Planitia', xy=(0.79, 0.92), xycoords='axes fraction', color=font_color) if w <15*1e18 else plt.annotate('Atalanta \n Mare', xy=(0.79, 0.92), xycoords='axes fraction', color=font_color)
            plt.annotate('Zhibek \n Planitia', xy=(0.8, 0.05), xycoords='axes fraction', color=font_color) if w <60*1e18 else plt.annotate('Zhibek \n Mare', xy=(0.8, 0.05), xycoords='axes fraction', color=font_color)

            plt.annotate('Maxwell \n Montes', xy=(0.52, 0.94), xycoords='axes fraction', color=font_color)
        
        elif self.planet=='Earth':
            
            plt.annotate(' North \nAmerica', xy=(0.24, 0.77), xycoords='axes fraction', color=font_color)
            plt.annotate('Asia', xy=(0.65, 0.82), xycoords='axes fraction', color=font_color)
            plt.annotate('Africa', xy=(0.55, 0.60), xycoords='axes fraction', color=font_color) 
            plt.annotate('Europe', xy=(0.52, 0.84), xycoords='axes fraction', color=font_color)
            plt.annotate('South \nAmerica', xy=(0.32, 0.40), xycoords='axes fraction', color=font_color)

            plt.annotate('Australia', xy=(0.83, 0.30), xycoords='axes fraction', color=font_color)
            plt.annotate('Antarctica', xy=(0.70, 0.02), xycoords='axes fraction', color=font_color)
            
        elif self.planet=='Mercury':
            plt.annotate('Caloris \nPlanitia', xy=(0.43+0.5, 0.7), xycoords='axes fraction', color=font_color)
            plt.annotate('Tir Planitia', xy=(0.5-0.5, 0.5), xycoords='axes fraction', color=font_color)
            plt.annotate('  Budh  \nPlanitia', xy=(0.55-0.5+0.01, 0.6), xycoords='axes fraction', color=font_color) 
            plt.annotate('Borealis Planitia', xy=(0.65-0.5+0.11, 0.95), xycoords='axes fraction', color=font_color)
            plt.annotate('Sobkou \nPlanitia', xy=(0.60-0.5, 0.75), xycoords='axes fraction', color=font_color)
            plt.annotate(' Suisei \nPlanitia', xy=(0.55-0.5+0.1, 0.88), xycoords='axes fraction', color=font_color)
            plt.annotate('  Odin  \nPlanitia', xy=(0.50-0.5+0.02, 0.65), xycoords='axes fraction', color=font_color)
            plt.annotate(' Sihtu \nPlanitia', xy=(0.82-0.5, 0.44), xycoords='axes fraction', color=font_color)

            plt.annotate('Beethoven', xy=(0.62-0.5, 0.33), xycoords='axes fraction', color=font_color)###
            plt.annotate('Rachmaninoff', xy=(0.13+0.5, 0.73), xycoords='axes fraction', color=font_color)
            plt.annotate('Tolstoj', xy=(0.52-0.5, 0.36), xycoords='axes fraction', color=font_color)
            plt.annotate('Raditladi', xy=(0.3+0.5, 0.68), xycoords='axes fraction', color=font_color)
            plt.annotate('Mozart', xy=(0.45+0.5, 0.54), xycoords='axes fraction', color=font_color)
            plt.annotate('Rembrandt', xy=(0.22+0.5, 0.22), xycoords='axes fraction', color=font_color)
            plt.annotate('Dostoevskij', xy=(0.52-0.5+0.07, 0.16), xycoords='axes fraction', color=font_color)

            plt.annotate('Holst', xy=(0.12+0.5, 0.36), xycoords='axes fraction', color=font_color)
            plt.annotate('Schubert', xy=(0.78-0.5-0.06, 0.2), xycoords='axes fraction', color=font_color)
            plt.annotate('Sanai', xy=(0.95-0.5, 0.40), xycoords='axes fraction', color=font_color)
            plt.annotate('Aneirin', xy=(0.03+0.5-0.03, 0.25), xycoords='axes fraction', color=font_color)
            plt.annotate('Goethe', xy=(0.75-0.5+0.15, 0.97), xycoords='axes fraction', color=font_color)
            plt.annotate('Mendelssohn', xy=(0.3+0.5-0.08, 0.97), xycoords='axes fraction', color=font_color)
            plt.annotate('Raphael', xy=(0.75-0.5, 0.33), xycoords='axes fraction', color=font_color)
            plt.annotate('Shakespeare', xy=(0.55-0.5+0.05, 0.84), xycoords='axes fraction', color=font_color)
            plt.annotate('Vyasa', xy=(0.70-0.5+0.05, 0.85), xycoords='axes fraction', color=font_color)
            
        elif self.planet=='Moon':
            plt.annotate('  Oceanus  \nProcellarum', xy=(0.3, 0.68), xycoords='axes fraction', color=font_color)
            plt.annotate('    Mare    \nSerenitatis', xy=(0.51, 0.70), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nImbrium', xy=(0.43, 0.7), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nCrisium', xy=(0.64, 0.61), xycoords='axes fraction', color=font_color)
            plt.annotate('      Mare     \nHumboldtianum', xy=(0.65, 0.90), xycoords='axes fraction', color=font_color)
            plt.annotate('    Mare    \nTranqullitatis', xy=(0.57, 0.55), xycoords='axes fraction', color=font_color) 
            plt.annotate('   Mare \nOrientale', xy=(0.22, 0.32), xycoords='axes fraction', color=font_color)
            plt.annotate('   Mare  \nFrigoris', xy=(0.45, 0.90), xycoords='axes fraction', color=font_color)
            plt.annotate('   Mare  \nFecunditatis', xy=(0.63, 0.47), xycoords='axes fraction', color=font_color)
            plt.annotate(' Mare \nNubium', xy=(0.44, 0.31), xycoords='axes fraction', color=font_color)
            plt.annotate('   Mare  \nInsularum', xy=(0.40, 0.55), xycoords='axes fraction', color=font_color)
            plt.annotate('   Mare  \nHumorum', xy=(0.36, 0.31), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nSmythii', xy=(0.72, 0.47), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nNectaris', xy=(0.57, 0.36), xycoords='axes fraction', color=font_color) 
            plt.annotate('   Mare  \nMoscoviense', xy=(0.86, 0.69), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nMarginis', xy=(0.72, 0.60), xycoords='axes fraction', color=font_color)

            plt.annotate('Hertzsprung', xy=(0.11, 0.50), xycoords='axes fraction', color=font_color)
            plt.annotate('Korolev', xy=(0.04, 0.46), xycoords='axes fraction', color=font_color)
            plt.annotate('Apollo', xy=(0.09, 0.22), xycoords='axes fraction', color=font_color)
            plt.annotate('Mendeleev', xy=(0.86, 0.54), xycoords='axes fraction', color=font_color)
            plt.annotate('Planck', xy=(0.78, 0.10), xycoords='axes fraction', color=font_color)
            plt.annotate('Schrodinger', xy=(0.70, 0.02), xycoords='axes fraction', color=font_color)

            plt.annotate('Selenean \nsummit', xy=(0.03, 0.53), xycoords='axes fraction', color=font_color)
            
    def add_toponyms_cyl(self, w, font_color):
        
        if self.planet=='Mars':
            
            plt.annotate('Olympus \nMons', xy=(0.1, 0.62), xycoords='axes fraction', color=font_color)
            plt.annotate('Tharsis \nMontes', xy=(0.16, 0.52), xycoords='axes fraction', color=font_color)
            plt.annotate('Alba \nMons', xy=(0.19, 0.74), xycoords='axes fraction', color=font_color)
            plt.annotate('Elysium \nMons', xy=(0.87, 0.64), xycoords='axes fraction', color=font_color)

            plt.annotate('Hellas \nPlanitia', xy=(0.67, 0.23), xycoords='axes fraction', color=font_color) if w <1*1e18 else plt.annotate('Hellas \nMare', xy=(0.67, 0.23), xycoords='axes fraction', color=font_color)
            plt.annotate('Argyre \nPlanitia', xy=(0.37, 0.18), xycoords='axes fraction', color=font_color) if w <62*1e18 else plt.annotate('Argyre \nMare', xy=(0.37, 0.18), xycoords='axes fraction', color=font_color)

            plt.annotate('Arcadia \nPlanitia', xy=(0.06, 0.8), xycoords='axes fraction', color=font_color) if w <16*1e18 else plt.annotate('Arcadia \nMare', xy=(0.06, 0.8), xycoords='axes fraction', color=font_color)
            plt.annotate('Amazonis \nPlanitia', xy=(0.02, 0.63), xycoords='axes fraction', color=font_color) if w <20*1e18 else plt.annotate('Amazonis \nMare', xy=(0.02, 0.63), xycoords='axes fraction', color=font_color)
            plt.annotate('Acidalia \nPlanitia', xy=(0.45, 0.82), xycoords='axes fraction', color=font_color) if w <6*1e18 else plt.annotate('Acidalia \nMare', xy=(0.45, 0.82), xycoords='axes fraction', color=font_color)
            plt.annotate('Chryse \nPlanitia', xy=(0.38, 0.7), xycoords='axes fraction', color=font_color) if w <34*1e18 else plt.annotate('Chryse \nMare', xy=(0.38, 0.7), xycoords='axes fraction', color=font_color)
            plt.annotate('Utopia \nPlanitia', xy=(0.78, 0.75), xycoords='axes fraction', color=font_color) if w <8*1e18 else plt.annotate('Utopia \nMare', xy=(0.78, 0.75), xycoords='axes fraction', color=font_color)
            plt.annotate('Isidis \nPlanitia', xy=(0.72, 0.57), xycoords='axes fraction', color=font_color) if w <25*1e18 else plt.annotate('Isidis \nMare', xy=(0.72, 0.57), xycoords='axes fraction', color=font_color)

            plt.annotate('Arabia \nTerra', xy=(0.5, 0.6), xycoords='axes fraction', color=font_color) if w <168*1e18 else plt.annotate('Arabia \nMare', xy=(0.5, 0.6), xycoords='axes fraction', color=font_color)
            plt.annotate('Noachis \nTerra', xy=(0.53, 0.18), xycoords='axes fraction', color=font_color)
            plt.annotate('Promethei \nTerra', xy=(0.79, 0.10), xycoords='axes fraction', color=font_color)# if w <300*1e18 else plt.annotate('Promethei \n Islands', xy=(0.8, 0.13), xycoords='axes fraction', color=font_color)
            plt.annotate('Aonia \nTerra', xy=(0.27, 0.10), xycoords='axes fraction', color=font_color)

            plt.annotate('Terra \nCimmeria', xy=(0.91, 0.08), xycoords='axes fraction', color=font_color)
            plt.annotate('Tyrrhena \nTerra', xy=(0.75, 0.40), xycoords='axes fraction', color=font_color)
            plt.annotate('Terra \nSabaea', xy=(0.6, 0.40), xycoords='axes fraction', color=font_color)

            plt.annotate('Terra \nSirenum', xy=(0.06, 0.13), xycoords='axes fraction', color=font_color)
            plt.annotate('Solis \nPlanum', xy=(0.25, 0.30), xycoords='axes fraction', color=font_color)
            plt.annotate('Valles Marineris', xy=(0.26, 0.41), xycoords='axes fraction', color=font_color) if w <24*1e18 else plt.annotate('Marineris Channels', xy=(0.26, 0.41), xycoords='axes fraction', color=font_color)

            plt.annotate('Vastitas Borealis', xy=(0.45, 0.94), xycoords='axes fraction', color=font_color) if w <10*1e18 else plt.annotate('Borealis Oceanus', xy=(0.45, 0.94), xycoords='axes fraction', color=font_color)
            plt.annotate('Planum Australe', xy=(0.45, 0.02), xycoords='axes fraction', color=font_color)
            
        elif self.planet=='Venus':
            
            plt.annotate('Aphrodite Terra', xy=(0.75, 0.40), xycoords='axes fraction', color=font_color)
            plt.annotate('Lada \n Terra', xy=(0.48, 0.09), xycoords='axes fraction', color=font_color)
            plt.annotate('Ishtar Terra', xy=(0.42, 0.83), xycoords='axes fraction', color=font_color) 
            plt.annotate('Lakshmi \n Planum', xy=(0.4, 0.87), xycoords='axes fraction', color=font_color)

            plt.annotate('Alpha \n Regio', xy=(0.5, 0.33), xycoords='axes fraction', color=font_color)
            plt.annotate('Beta \n Regio', xy=(0.28, 0.67), xycoords='axes fraction', color=font_color)
            plt.annotate('Eistla \n Regio', xy=(0.55, 0.57), xycoords='axes fraction', color=font_color)
            plt.annotate('Atla \n Regio', xy=(0.04, 0.52), xycoords='axes fraction', color=font_color)
            plt.annotate('Phoebe \n Regio', xy=(0.28, 0.48), xycoords='axes fraction', color=font_color)
            plt.annotate('Ovda \n Regio', xy=(0.75, 0.48), xycoords='axes fraction', color=font_color)
            plt.annotate('Thetis \n Regio', xy=(0.82, 0.44), xycoords='axes fraction', color=font_color)
            plt.annotate('Themis \nRegio', xy=(0.3, 0.22), xycoords='axes fraction', color=font_color)
            plt.annotate('Artemis \nChasma', xy=(0.82, 0.27), xycoords='axes fraction', color=font_color)
            plt.annotate('Ulfrun \nRegio', xy=(0.11, 0.58), xycoords='axes fraction', color=font_color)

            plt.annotate('Sedna \n Planitia', xy=(0.42, 0.73), xycoords='axes fraction', color=font_color) if w <75*1e18 else plt.annotate('Sedna \n Mare', xy=(0.42, 0.73), xycoords='axes fraction', color=font_color)
            plt.annotate('Guinevere \n Planitia', xy=(0.3, 0.75), xycoords='axes fraction', color=font_color) if w <15*1e18 else plt.annotate('Guinevere \n Mare', xy=(0.3, 0.75), xycoords='axes fraction', color=font_color)
            plt.annotate('Aino \n Planitia', xy=(0.77, 0.20), xycoords='axes fraction', color=font_color) if w <90*1e18 else plt.annotate('Aino \n Mare', xy=(0.77, 0.20), xycoords='axes fraction', color=font_color)
            plt.annotate('Ganiki \n Planitia', xy=(0.07, 0.72), xycoords='axes fraction', color=font_color) if w <120*1e18 else plt.annotate('Ganiki \n Mare', xy=(0.07, 0.72), xycoords='axes fraction', color=font_color)
            plt.annotate('Lavinia \n Planitia', xy=(0.45, 0.20), xycoords='axes fraction', color=font_color) if w <34*1e18 else plt.annotate('Lavinia \n Mare', xy=(0.45, 0.20), xycoords='axes fraction', color=font_color)
            plt.annotate('Helen \n Planitia', xy=(0.10, 0.15), xycoords='axes fraction', color=font_color) if w <60*1e18 else plt.annotate('Helen \n Mare', xy=(0.10, 0.15), xycoords='axes fraction', color=font_color)
            plt.annotate('Niobe \n Planitia', xy=(0.82, 0.62), xycoords='axes fraction', color=font_color) if w <75*1e18 else plt.annotate('Niobe \n Mare', xy=(0.82, 0.62), xycoords='axes fraction', color=font_color)
            plt.annotate('Rusalka \n Planitia', xy=(0.93, 0.50), xycoords='axes fraction', color=font_color) if w <100*1e18 else plt.annotate('Rusalka \n Gulf', xy=(0.93, 0.50), xycoords='axes fraction', color=font_color)
            plt.annotate('Kawelu \n Planitia', xy=(0.18, 0.65), xycoords='axes fraction', color=font_color) if w <120*1e18 else plt.annotate('Kawelu \n Mare', xy=(0.18, 0.65), xycoords='axes fraction', color=font_color)
            plt.annotate('Bereghinya \n Planitia', xy=(0.53, 0.69), xycoords='axes fraction', color=font_color) if w <90*1e18 else plt.annotate('Bereghinya \n Mare', xy=(0.53, 0.69), xycoords='axes fraction', color=font_color)
            plt.annotate('Atalanta \n Planitia', xy=(0.90, 0.85), xycoords='axes fraction', color=font_color) if w <15*1e18 else plt.annotate('Atalanta \n Mare', xy=(0.90, 0.85), xycoords='axes fraction', color=font_color)
            plt.annotate('Zhibek \n Planitia', xy=(0.87, 0.10), xycoords='axes fraction', color=font_color) if w <60*1e18 else plt.annotate('Zhibek \n Mare', xy=(0.87, 0.10), xycoords='axes fraction', color=font_color)

            plt.annotate('Maxwell \n Montes', xy=(0.52, 0.87), xycoords='axes fraction', color=font_color)
        
        elif self.planet=='Earth':
            
            plt.annotate(' North \nAmerica', xy=(0.24, 0.75), xycoords='axes fraction', color=font_color)
            plt.annotate('Asia', xy=(0.75, 0.80), xycoords='axes fraction', color=font_color)
            plt.annotate('Africa', xy=(0.55, 0.60), xycoords='axes fraction', color=font_color) 
            plt.annotate('Europe', xy=(0.52, 0.78), xycoords='axes fraction', color=font_color)
            plt.annotate('South \nAmerica', xy=(0.32, 0.40), xycoords='axes fraction', color=font_color)

            plt.annotate('Australia', xy=(0.83, 0.35), xycoords='axes fraction', color=font_color)
            plt.annotate('Antarctica', xy=(0.70, 0.02), xycoords='axes fraction', color=font_color)
            
        elif self.planet=='Mercury':
            plt.annotate('Caloris \nPlanitia', xy=(0.43+0.5, 0.7), xycoords='axes fraction', color=font_color)
            plt.annotate('Tir Planitia', xy=(0.5-0.5, 0.5), xycoords='axes fraction', color=font_color)
            plt.annotate('  Budh  \nPlanitia', xy=(0.55-0.5+0.01, 0.6), xycoords='axes fraction', color=font_color) 
            plt.annotate('Borealis Planitia', xy=(0.65-0.5+0.11, 0.95), xycoords='axes fraction', color=font_color)
            plt.annotate('Sobkou \nPlanitia', xy=(0.60-0.5, 0.71), xycoords='axes fraction', color=font_color)
            plt.annotate(' Suisei \nPlanitia', xy=(0.55-0.5, 0.82), xycoords='axes fraction', color=font_color)
            plt.annotate('  Odin  \nPlanitia', xy=(0.50-0.5+0.02, 0.65), xycoords='axes fraction', color=font_color)
            plt.annotate(' Sihtu \nPlanitia', xy=(0.82-0.5, 0.44), xycoords='axes fraction', color=font_color)

            plt.annotate('Beethoven', xy=(0.62-0.5, 0.35), xycoords='axes fraction', color=font_color)###
            plt.annotate('Rachmaninoff', xy=(0.13+0.5, 0.68), xycoords='axes fraction', color=font_color)
            plt.annotate('Tolstoj', xy=(0.52-0.5, 0.38), xycoords='axes fraction', color=font_color)
            plt.annotate('Raditladi', xy=(0.3+0.5, 0.68), xycoords='axes fraction', color=font_color)
            plt.annotate('Mozart', xy=(0.45+0.5, 0.54), xycoords='axes fraction', color=font_color)
            plt.annotate('Rembrandt', xy=(0.22+0.5, 0.26), xycoords='axes fraction', color=font_color)
            plt.annotate('Dostoevskij', xy=(0.52-0.5+0.02, 0.19), xycoords='axes fraction', color=font_color)

            plt.annotate('Holst', xy=(0.12+0.5, 0.38), xycoords='axes fraction', color=font_color)
            plt.annotate('Schubert', xy=(0.78-0.5-0.06-0.02, 0.22), xycoords='axes fraction', color=font_color)
            plt.annotate('Sanai', xy=(0.95-0.5, 0.40), xycoords='axes fraction', color=font_color)
            plt.annotate('Aneirin', xy=(0.03+0.5-0.03, 0.29), xycoords='axes fraction', color=font_color)
            plt.annotate('Goethe', xy=(0.75-0.5+0.15, 0.97), xycoords='axes fraction', color=font_color)
            plt.annotate('Mendelssohn', xy=(0.3+0.5-0.08, 0.97), xycoords='axes fraction', color=font_color)
            plt.annotate('Raphael', xy=(0.75-0.5, 0.33), xycoords='axes fraction', color=font_color)
            plt.annotate('Shakespeare', xy=(0.55-0.5+0.05, 0.80), xycoords='axes fraction', color=font_color)
            plt.annotate('Vyasa', xy=(0.70-0.5+0.05, 0.80), xycoords='axes fraction', color=font_color)
            
        elif self.planet=='Moon':
            plt.annotate('  Oceanus  \nProcellarum', xy=(0.3, 0.68), xycoords='axes fraction', color=font_color)
            plt.annotate('    Mare    \nSerenitatis', xy=(0.51, 0.67), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nImbrium', xy=(0.43, 0.7), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nCrisium', xy=(0.64, 0.59), xycoords='axes fraction', color=font_color)
            plt.annotate('      Mare     \nHumboldtianum', xy=(0.67, 0.83), xycoords='axes fraction', color=font_color)
            plt.annotate('    Mare    \nTranqullitatis', xy=(0.57, 0.55), xycoords='axes fraction', color=font_color) 
            plt.annotate('   Mare \nOrientale', xy=(0.22, 0.35), xycoords='axes fraction', color=font_color)
            plt.annotate('   Mare  \nFrigoris', xy=(0.45, 0.83), xycoords='axes fraction', color=font_color)
            plt.annotate('   Mare  \nFecunditatis', xy=(0.63, 0.47), xycoords='axes fraction', color=font_color)
            plt.annotate(' Mare \nNubium', xy=(0.44, 0.34), xycoords='axes fraction', color=font_color)
            plt.annotate('   Mare  \nInsularum', xy=(0.40, 0.55), xycoords='axes fraction', color=font_color)
            plt.annotate('   Mare  \nHumorum', xy=(0.36, 0.34), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nSmythii', xy=(0.72, 0.47), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nNectaris', xy=(0.57, 0.38), xycoords='axes fraction', color=font_color) 
            plt.annotate('   Mare  \nMoscoviense', xy=(0.87, 0.67), xycoords='axes fraction', color=font_color)
            plt.annotate('  Mare  \nMarginis', xy=(0.72, 0.60), xycoords='axes fraction', color=font_color)

            plt.annotate('Hertzsprung', xy=(0.11, 0.50), xycoords='axes fraction', color=font_color)
            plt.annotate('Korolev', xy=(0.04, 0.46), xycoords='axes fraction', color=font_color)
            plt.annotate('Apollo', xy=(0.05, 0.25), xycoords='axes fraction', color=font_color)
            plt.annotate('Mendeleev', xy=(0.86, 0.54), xycoords='axes fraction', color=font_color)
            plt.annotate('Planck', xy=(0.83, 0.05), xycoords='axes fraction', color=font_color)
            plt.annotate('Schrodinger', xy=(0.70, 0.02), xycoords='axes fraction', color=font_color)

            plt.annotate('Selenean \nsummit', xy=(0.03, 0.53), xycoords='axes fraction', color=font_color)
            
    def add_toponyms_ortho(self, w, font_color, lon_0=0, lat_0=0):
        
        if self.planet=='Mars':  
            
            title_df = pd.DataFrame(columns=['title','x','y'])
            title_df.loc[0,:] = ['Olympus \nMons',226,18]
            title_df.loc[1,:] = ['Tharsis \nMontes',-112,2]
            title_df.loc[2,:] = ['Alba \nMons',250,40]
            title_df.loc[3,:] = ['Elysium \nMons',147,25]
            title_df.loc[4,:] = ['Hellas \nPlanitia', 70, -42] if w <1*1e18 else ['Hellas \nMare', 70, -42]
            title_df.loc[5,:] = ['Argyre \nPlanitia', 316, -50] if w <62*1e18 else ['Argyre \nMare', 316, -50]
            title_df.loc[6,:] = ['Arcadia \nPlanitia', 184, 47] if w <16*1e18 else ['Arcadia \nMare', 184, 47]
            title_df.loc[7,:] = ['Amazonis \nPlanitia', 196, 25] if w <20*1e18 else ['Amazonis \nMare', 196, 25]
            title_df.loc[8,:] = ['Acidalia \nPlanitia', 339, 50] if w <6*1e18 else ['Acidalia \nMare', 339, 50]
            title_df.loc[9,:] = ['Chryse \nPlanitia', 320, 28] if w <34*1e18 else ['Chryse \nMare', 320, 28]
            title_df.loc[10,:] = ['Utopia \nPlanitia', 118, 47] if w <8*1e18 else ['Utopia \nMare', 118, 47]
            title_df.loc[11,:] = ['Isidis \nPlanitia', 88, 14] if w <25*1e18 else ['Isidis \nMare', 88, 14]
            title_df.loc[12,:] = ['Arabia \nTerra', 355, 23] if w <168*1e18 else ['Arabia \nMare', 355, 23]
            title_df.loc[13,:] = ['Valles Marineris', -59, -14] if w <24*1e18 else ['Marineris Channels', -59, -14]
            title_df.loc[14,:] = ['Vastitas Borealis', 32, 87] if w <10*1e18 else ['Borealis Oceanus', 32, 87]
            title_df.loc[15,:] = ['Noachis \nTerra', 350, -45]
            title_df.loc[16,:] = ['Promethei \nTerra', 100, -58]
            title_df.loc[17,:] = ['Aonia \nTerra', -100, -62]
            title_df.loc[18,:] = ['Terra \nCimmeria', 145, -35]
            title_df.loc[19,:] = ['Tyrrhena \nTerra', 90, -15]
            title_df.loc[20,:] = ['Terra \nSabaea', 42, 2]
            title_df.loc[21,:] = ['Terra \nSirenum', -150, -40]
            title_df.loc[22,:] = ['Solis \nPlanum', 270, -26]
            title_df.loc[23,:] = ['Planum Australe', 160, -84]
                    
        elif self.planet=='Venus':
            
            title_df = pd.DataFrame(columns=['title','x','y'])
            title_df.loc[0,:] = ['Aphrodite  Terra',100,-15]
            title_df.loc[1,:] = ['Lada \n Terra',20,-60]
            title_df.loc[2,:] = ['Ishtar Terra',27.5,55]
            title_df.loc[3,:] = ['Lakshmi \n Planum',339,69]

            title_df.loc[4,:] = ['Alpha \n Regio', 5, -22]
            title_df.loc[5,:] = ['Beta \n Regio', 282, 25]
            title_df.loc[6,:] = ['Eistla \n Regio', 21, 10]
            title_df.loc[7,:] = ['Atla \n Regio', 200, 9]
            title_df.loc[8,:] = ['Phoebe \n Regio', 283, -6]
            title_df.loc[9,:] = ['Ovda \n Regio', 86, -3]
            title_df.loc[10,:] = ['Thetis \n Regio', 130, -11]
            title_df.loc[11,:] = ['Themis \nRegio', 284, -37]
            title_df.loc[12,:] = ['Artemis \nChasma', 138, -35]
            title_df.loc[13,:] = ['Ulfrun \nRegio', 225, 27]
            title_df.loc[14,:] = ['Maxwell \n Montes', 13, 65]

            title_df.loc[15,:] = ['Sedna \n Planitia', 345, 41] if w <75*1e18 else ['Sedna \n Mare', 345, 41]
            title_df.loc[16,:] = ['Guinevere \n Planitia', 325, 22] if w <15*1e18 else ['Guinevere \n Mare', 325, 22]
            title_df.loc[17,:] = ['Aino \n Planitia', 94, -40] if w <90*1e18 else ['Aino \n Mare', 94, -40]
            title_df.loc[18,:] = ['Ganiki \n Planitia', 202, 40] if w <120*1e18 else ['Ganiki \n Mare', 202, 40]
            title_df.loc[19,:] = ['Lavinia \n Planitia', 347, -47] if w <34*1e18 else ['Lavinia \n Mare', 347, -47]
            title_df.loc[20,:] = ['Helen \n Planitia', 264, -52] if w <60*1e18 else ['Helen \n Mare', 264, -52]
            title_df.loc[21,:] = ['Niobe \n Planitia', 112, 21] if w <75*1e18 else ['Niobe \n Mare', 112, 21]
            title_df.loc[22,:] = ['Rusalka \n Planitia', 170, 10] if w <100*1e18 else ['Rusalka \n Gulf', 170, 10]
            title_df.loc[23,:] = ['Kawelu \n Planitia', 246, 33] if w <120*1e18 else ['Kawelu \n Mare', 246, 33]
            title_df.loc[24,:] = ['Bereghinya \n Planitia', 24, 29] if w <90*1e18 else ['Bereghinya \n Mare', 24, 29]
            title_df.loc[25,:] = ['Atalanta \n Planitia', 166, 50] if w <15*1e18 else ['Atalanta \n Mare', 166, 50]
            title_df.loc[26,:] = ['Zhibek \n Planitia', 157, -50] if w <60*1e18 else ['Zhibek \n Mare', 157, -50]

            title_df.loc[27,:] = ['Tinatin \n Planitia', 15, -15] if w <35*1e18 else ['Tinatin \n Mare', 15, -15]
            title_df.loc[28,:] = ['Hinemoa \n Planitia', 265, 5] if w <40*1e18 else ['Hinemoa \n Mare', 265, 5]
            title_df.loc[29,:] = ['Kanykey \n Planitia', 350, -10] if w <80*1e18 else ['Kanykey \n Mare', 350, -10]
            title_df.loc[30,:] = ['Tahmina \n Planitia', 80, -23] if w <90*1e18 else ['Tahmina \n Mare', 80, -23]

            title_df.loc[31,:] = ['Dali \nChasma', 167, -18]
            title_df.loc[32,:] = ['Diana \nChasma', 155, -15]
        
        elif self.planet=='Earth':
            
            title_df = pd.DataFrame(columns=['title','x','y'])
            title_df.loc[0,:] = [' North \nAmerica',-90,40 ]
            title_df.loc[1,:] = ['Asia',70,45 ]
            title_df.loc[2,:] = ['Africa',30,15 ]
            title_df.loc[3,:] = ['Europe',20,47 ]
            title_df.loc[4,:] = ['South \nAmerica',-60,-15 ]
            title_df.loc[5,:] = ['Australia',125,-25 ]
            title_df.loc[6,:] = ['Antarctica',120,-80 ]
            
        elif self.planet=='Mercury':
            
            title_df = pd.DataFrame(columns=['title','x','y'])
            title_df.loc[0,:] = ['Caloris \nPlanitia',-190, 30]
            title_df.loc[1,:] = ['  Tir \nPlanitia',-176, 1]
            title_df.loc[2,:] = ['  Budh  \nPlanitia',-151, 22]
            title_df.loc[3,:] = ['Borealis \nPlanitia',-79, 73]
            title_df.loc[4,:] = ['Sobkou \nPlanitia',-128, 39]
            title_df.loc[5,:] = [' Suisei \nPlanitia',-150, 59]
            title_df.loc[6,:] = ['  Odin  \nPlanitia',-172,23]
            title_df.loc[7,:] = [' Sihtu \nPlanitia',-55,-3]
            
            title_df.loc[8,:] = ['Beethoven',-124,-20]
            title_df.loc[9,:] = ['Rachmaninoff',-302,27]
            title_df.loc[10,:] = ['Tolstoj',-163,-16]
            title_df.loc[11,:] = ['Raditladi',-240,27]
            title_df.loc[12,:] = ['Mozart',-190,8]
            title_df.loc[13,:] = ['Rembrandt',-271,-33]
            title_df.loc[14,:] = ['Dostoevskij',-176,-45]
            
            title_df.loc[15,:] = ['Holst',-315,-17]
            title_df.loc[16,:] = ['Schubert',-54,-43]
            title_df.loc[17,:] = ['Sanai',-7,-13]
            title_df.loc[18,:] = ['Aneirin',-3,-27]
            title_df.loc[19,:] = ['Goethe',-54,81]
            title_df.loc[20,:] = ['Mendelssohn',-258,70]
            title_df.loc[21,:] = ['Raphael',-76,-20]
            title_df.loc[22,:] = ['Shakespeare',-152,48]
            title_df.loc[23,:] = ['Vyasa',-80,43]
            
        elif self.planet=='Moon':
            
            title_df = pd.DataFrame(columns=['title','x','y'])
            title_df.loc[0,:] = ['  Oceanus  \nProcellarum', -54, 18]
            title_df.loc[1,:] = ['    Mare    \nSerenitatis',17,28 ]
            title_df.loc[2,:] = ['  Mare  \nImbrium',-16,33 ]
            title_df.loc[3,:] = ['  Mare  \nCrisium',59,17 ]
            title_df.loc[4,:] = ['      Mare     \nHumboldtianum',81,57 ]
            title_df.loc[5,:] = ['    Mare    \nTranqullitatis',31,8 ]
            title_df.loc[6,:] = ['   Mare \nOrientale',-93,-19 ]
            title_df.loc[7,:] = ['   Mare  \nFrigoris',1,56 ]
            title_df.loc[8,:] = ['   Mare  \nFecunditatis',51,-8 ]
            title_df.loc[9,:] = [' Mare \nNubium',-17,-21 ]
            title_df.loc[10,:] = ['   Mare  \nInsularum',-31,7 ]
            title_df.loc[11,:] = ['   Mare  \nHumorum',-37,-24 ]
            title_df.loc[12,:] = ['  Mare  \nSmythii',87,1 ]
            title_df.loc[13,:] = ['  Mare  \nNectaris',35,-15 ]
            title_df.loc[14,:] = ['   Mare  \nMoscoviense',148,27 ]
            title_df.loc[15,:] = ['  Mare  \nMarginis',86,13 ]
            
            title_df.loc[16,:] = ['Hertzsprung',-129,1 ]
            title_df.loc[17,:] = ['Korolev',-157,-4 ]
            title_df.loc[18,:] = ['Apollo',-151,-36 ]
            title_df.loc[19,:] = ['Mendeleev',141,6 ]
            title_df.loc[20,:] = ['Planck',137,-58 ]
            title_df.loc[21,:] = ['Schrödinger',132,-75 ]
            title_df.loc[22,:] = ['Selenean \nsummit',-159,5 ]
        
        title_df['x'] = title_df['x'].astype(float)/180*np.pi
        title_df['y'] = title_df['y'].astype(float)/180*np.pi
        lon_1 = lon_0/180*np.pi
        lat_1 = lat_0/180*np.pi
        title_df['cor_x'] = 0.5 + 0.5*np.cos(title_df['y'])*np.sin(title_df['x']-lon_1)
        title_df['cor_y'] = 0.5 + 0.5*(np.cos(lat_1)*np.sin(title_df['y'])-np.sin(lat_1)*np.cos(title_df['y'])*np.cos(title_df['x']-lon_1))
        title_df['cos_c'] = np.sin(lat_1)*np.sin(title_df['y'])-np.cos(lat_1)*np.cos(title_df['y'])*np.cos(title_df['x']-lon_1)
        
        for t in range(len(title_df)):
            if title_df.loc[t,'cos_c'] < -0.1 :
                plt.annotate(title_df.loc[t,'title'], xy=(title_df.loc[t,'cor_x'], title_df.loc[t,'cor_y']), 
                             xycoords='axes fraction', color=font_color, va='center', ha='center', 
                             fontsize=(1-np.abs(title_df.loc[t,'cor_x']-0.5))*12)        
                    
        del title_df
        gc.collect()
            
    def add_caption(self, font_color, projection='eck4'):
        
        if self.planet=='Mars':
            if projection=='eck4':
                plt.annotate('Data: Mars MGS MOLA - MEX HRSC Blended DEM Global 200m v2 (USGS Astrogeology Science Center), Design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)
            elif projection=='ortho':
                plt.annotate('Data: Mars MGS MOLA - MEX HRSC Blended DEM Global 200m v2 (USGS Astrogeology Science Center), design: Anatoly Ilyenkov                                       ', 
                             xy=(-0.2, -0.1), xycoords='axes fraction', color=font_color)
            elif projection=='cyl':
                plt.annotate('Data: Mars MGS MOLA - MEX HRSC Blended DEM Global 200m v2 (USGS Astrogeology Science Center), Design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)
        elif self.planet=='Venus':
            if projection=='eck4':
                plt.annotate('Data: Magellan Global Topography 4641m (USGS Astrogeology Science Center), design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)
            elif projection=='ortho':
                plt.annotate('Data: Magellan Global Topography 4641m (USGS Astrogeology Science Center), design: Anatoly Ilyenkov                                       ', 
                             xy=(-0.2, -0.1), xycoords='axes fraction', color=font_color)
            elif projection=='cyl':
                plt.annotate('Data: Magellan Global Topography 4641m (USGS Astrogeology Science Center), design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)
        elif self.planet=='Earth':
            if projection=='eck4':
                plt.annotate('Data: Earth ETOPO5 5-minute gridded elevation data, design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)
            elif projection=='ortho':
                plt.annotate('Data: Earth ETOPO5 5-minute gridded elevation data, design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)
            elif projection=='cyl':
                plt.annotate('Data: Earth ETOPO5 5-minute gridded elevation data, design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)
        elif self.planet=='Mercury':
            if projection=='eck4':
                plt.annotate('Data: Mercury MESSENGER Global DEM 665m v2 (USGS Astrogeology Science Center), design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)    
            elif projection=='ortho':
                plt.annotate('Data: Mercury MESSENGER Global DEM 665m v2 (USGS Astrogeology Science Center), design: Anatoly Ilyenkov                                      ', 
                             xy=(-0.2, -0.1), xycoords='axes fraction', color=font_color)
            elif projection=='cyl':
                plt.annotate('Data: Mercury MESSENGER Global DEM 665m v2 (USGS Astrogeology Science Center), design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)   
        elif self.planet=='Moon':
            if projection=='eck4':
                plt.annotate('Data: Moon LRO LOLA DEM 118m v1 (USGS Astrogeology Science Center), design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)
            elif projection=='ortho':
                plt.annotate('Data: Moon LRO LOLA DEM 118m v1 (USGS Astrogeology Science Center), design: Anatoly Ilyenkov                                      ', 
                             xy=(-0.2, -0.1), xycoords='axes fraction', color=font_color)
            elif projection=='cyl':
                plt.annotate('Data: Moon LRO LOLA DEM 118m v1 (USGS Astrogeology Science Center), design: Anatoly Ilyenkov', 
                             xy=(0.0, -0.1), xycoords='axes fraction', color=font_color)
        
    
    def save_map_file(self, fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection, solar_irradiance=1361, 
                      sea_level_pressure=101325, maptype='altitude', read_file=True, facecolor='white', hs=None, shade=None):
        
        image_stream = io.BytesIO()
        fig.savefig(image_stream, dpi=self.dpi, quality=95, pad_inches=0.1, bbox_inches='tight', facecolor=facecolor)

        plt.draw()
        fig.clf()
        ax.cla()
        plt.close('all')
        plt.close(fig)     
        del fig, ax, key, cbar
        plt.ioff()
        gc.collect()

        image_stream.seek(0)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            
        if hs is not None:
            img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
            hs = cv.cvtColor(hs, cv.COLOR_RGB2HSV)
            img[-hs.shape[0]:, :hs.shape[1], 2] = ((img[-hs.shape[0]:, :hs.shape[1], 2]).astype(float)*(3/4+hs[:,:,2]/(255*4))).astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
            
        if shade is not None:
            img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
            shade = cv.cvtColor(shade, cv.COLOR_RGB2HSV)
            img[-shade.shape[0]:, :shade.shape[1], 2] = ((img[-hs.shape[0]:, :hs.shape[1], 2]).astype(float)*(0.1+shade[:,:,2]/255*0.9)).astype(np.uint8)
            img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
        
        if not read_file:
            directory = 'out_'+self.planet.lower()+'_maps'
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            fn = directory+'/'+self.planet+'_'+maptype+'_'+projection+'_map_n_'+'%04d'%i+'_angle_'+str(int(angle_lon))+\
                    '_water_volume_'+str(w)+'_sea_level_'+str(sl)+'_sea_share_'+str(ss)+'_insol_'+\
                    str(round(solar_irradiance))+'_pressure_'+str(round(sea_level_pressure))+'_quality_'+\
                    str(self.quality)+'.jpg'
        
            img = cv.imwrite(fn, img)
            return None            

        else:
            return img
            
    
    def make_title_text(self, maptype, title=None, w=None, sl=None ,ss=None, sea_level_pressure=None, solar_irradiance=None):
        
        WEG_layer = round(w / self.full_square / 1000, 1)  
        
        if title=='long':

            title_text = 'Sea share '+str(ss)+"% of "+self.planet+" total surface  |  Land share "+\
                            str(round(self.full_square*(100-ss)/(1.49*1e14), 1))+"% of Earth's land surface"
            title_text += '\nTotal water mass '+'%.02f'%(w/self.params['scale_for_title'])+\
                            str(self.params['scale_for_title'])[1:]+'kg  (WEG layer '+str(WEG_layer)+\
                            'm)  |  Max sea depth '+str(round((sl+self.min_new-self.min_orig)/1000,1))+\
                            'km  |  '+self.params['highest_point']+' '+\
                            str(round((self.max_delta-sl+self.max_orig-self.max_new)/1000, 1))+'km'
            
            if self.planet=='Mars':
                if w>=5*1e18 and w<=6*1e18:
                    title_text += '\n Current martian water deposits \n'
                elif w>=10*1e18 and w<=11*1e18:
                    title_text += '\n Current martian water deposits + Hyperion moon melted \n'
                elif w>=20*1e18 and w<=21*1e18:
                    title_text += '\n Past martian water deposits \n'
                elif w>=42*1e18 and w<=43*1e18:
                    title_text += '\n Current martian water deposits + Mimas moon melted \n'
                elif w>6*1e18:
                    title_text += '\n Current martian water deposits + '+str(round((w-5*1e18)/(1e14)/1200,1))+\
                                    ' comets melted per month during 100 years (d 7km, density 600kg/m^3, mass 1e14kg) \n'
                elif w<5*1e18:
                    title_text += '\n '+str(round(w/(5*1e18)*100)) + '% of current martian water deposits melted \n'
                else:
                    title_text += '\n\n'
            else:
                title_text += '\n'+str(round((w-5*1e18)/(1e14)/1200,1))+\
                                ' comets melted per month during 100 years (d 7km, density 600kg/m^3, mass 1e14kg) \n'
                
            if maptype=='temperature':
                title_text += "Earth's rotation, axial tilt and albedo | Solar irradiance "+str(round((solar_irradiance)))+\
                            ' w/m^2, ' + str(round(solar_irradiance/1361*100,1)) + '% of Earth, ' + \
                            str(round(solar_irradiance/self.params['solar_irradiance']*100,1)) + '% of '+ self.planet + '\n'
                title_text += "Average temperature " + str(round(self.temp_avg-273.15)) + "°C (Earth's average 15°C), "+ \
                        'Min '+str(round(self.min_temperature))+'°C, Max '+ \
                        str(round(self.max_temperature))+'°C | ' + \
                        "Sea level pressure "+str(round(sea_level_pressure/1000, 1))+\
                        ' kPa, ' + str(round(sea_level_pressure/101325*100,1)) + "% of Earth's"+'\n'
                
            elif maptype=='pressure':
                title_text += "Earth's rotation, axial tilt and albedo | "+\
                              "Average annual temperature "+str(round(self.temp_avg-273.15))+"°C (Earth's average 15°C) \n"
                title_text += "Sea level pressure "+str(round(sea_level_pressure/1000,1))+' kPa, ' + \
                                str(round(sea_level_pressure/101325*100,1)) +"% of Earth | Minimal pressure "+\
                                str(round(self.min_pressure/1000,1))+' kPa,  '+\
                                str(round(self.min_pressure/1000/33.700*100)) +\
                                "% of Mount Everest summit \n"
                title_text += "Mass of the Atmosphere " + str(round(sea_level_pressure/self.g_0*self.full_square/1e18,1)) + \
                                'e+18kg, 75% of Nitrogen ('+str(round(sea_level_pressure/self.g_0*self.full_square*0.75/1e18,1))+\
                                'e+18kg) and 25% of Oxygen(' +str(round(sea_level_pressure/self.g_0*self.full_square*0.25/1e18,1))+\
                                'e+18kg) \n'
                
            elif maptype=='precipitations':
                title_text += "Earth's rotation, axial tilt and albedo"+" | Solar irradiance "+str(round((solar_irradiance)))+\
                                ' w/m^2, ' + str(round(solar_irradiance/1361*100,1)) + '% of Earth'+\
                                " | Sea level pressure "+str(round(sea_level_pressure/1000,1))+' kPa, ' + \
                                str(round(sea_level_pressure/101325*100,1)) + "% of Earth \n"
                title_text += "Average annual temperature "+str(round(self.temp_avg-273.15))+"°C (Earth's average 15°C) | "+\
                                "Average Earth-annual precipitations " +\
                                str(int(round(self.average_precipation)))+"mm (Earth's average 740mm)"+ '\n'
                
            elif maptype=='climate':
                title_text += "Earth's rotation, axial tilt and albedo"+" | Solar irradiance "+str(round((solar_irradiance)))+\
                              ' w/m^2, ' + str(round(solar_irradiance/1361*100,1)) + '% of Earth'+" | Sea level pressure "+\
                              str(round(sea_level_pressure/1000,1))+' kPa, ' + str(round(sea_level_pressure/101325*100,1)) + \
                              "% of Earth \n"
                title_text += "Average annual temperature "+str(round(self.temp_avg-273.15))+"°C (Earth's average 15°C) | "+\
                            "Average Earth-annual precipitations " +\
                                str(int(round(self.average_precipation)))+"mm (Earth's average 740mm)"+ '\n'
                
            elif maptype=='natural_colors':
                title_text += "Earth's rotation, axial tilt and albedo"+" | Solar irradiance "+str(round((solar_irradiance)))+\
                              ' w/m^2, ' + str(round(solar_irradiance/1361*100,1)) + '% of Earth'+" | Sea level pressure "+\
                              str(round(sea_level_pressure/1000,1))+' kPa, ' + str(round(sea_level_pressure/101325*100,1)) + \
                              "% of Earth \n"
                title_text += "Average annual temperature "+str(round(self.temp_avg-273.15))+"°C (Earth's average 15°C) | "+\
                            "Average Earth-annual precipitations " +\
                                str(int(round(self.average_precipation)))+"mm (Earth's average 740mm)"+ '\n'
                
            elif maptype=='radiation':
                title_text += "Earth's rotation, axial tilt and albedo | Solar irradiance "+str(round((solar_irradiance)))+\
                              ' w/m^2, ' + str(round(solar_irradiance/1361*100,1)) + '% of Earth, ' + \
                              str(round(solar_irradiance/self.params['solar_irradiance']*100,1)) + '% of '+self.planet + '\n'
                title_text += "Sea level pressure "+str(round(sea_level_pressure/1000,1))+' kPa, ' + \
                              str(round(sea_level_pressure/101325*100,1)) + "% of Earth | Minimal pressure "+\
                              str(round(self.min_pressure/1000,1))+' kPa,  '+ str(round(self.min_pressure/1000/33.700*100)) +\
                                "% of Mount Everest summit \n"
                title_text += "Mass of the Atmosphere " + str(round(sea_level_pressure/self.g_0*self.full_square/1e18,1)) + \
                                'e+18kg, 75% of Nitrogen ('+str(round(sea_level_pressure/self.g_0*self.full_square*0.75/1e18,1))+ \
                                'e+18kg) and 25% of Oxygen(' +str(round(sea_level_pressure/self.g_0*self.full_square*0.25/1e18,1))+\
                                'e+18kg) \n'
                title_text += "Average radiation " + str(round(self.mean_radiation,2))+"mSv/yr, min "+ \
                                str(round(self.min_radiation,2)) + "mSv/yr, max "+ \
                                str(round(self.max_radiation,2)) + \
                                "mSv/yr (Earth's sea level 0.5 mSv/yr, Earth's 10 km - 50 mSv/yr)" + '\n'
                
            elif maptype=='remoteness':
                title_text += "Average coast distance " + '{:,.0f}'.format(self.average_remoteness)+' km, Maximum coast distance ' + \
                                '{:,.0f}'.format(self.max_remoteness) + \
                                "km (Earth's average 236km, Earth's maximum 3,282km)" + '\n'
                
            elif maptype=='pop_den':
                title_text += "Earth's rotation, axial tilt and albedo | Solar irradiance "+str(round((solar_irradiance)))+\
                              ' w/m^2, ' + str(round(solar_irradiance/1361*100,1)) + '% of Earth, ' + \
                              str(round(solar_irradiance/self.params['solar_irradiance']*100,1)) + '% of '+self.planet + '\n'
                title_text += "Average temperature " + str(round(self.temp_avg-273.15)) + "°C (Earth's average 15°C), "+ \
                                'Min '+str(round(self.min_temperature))+'°C, Max '+ \
                                str(round(self.max_temperature))+'°C | ' + \
                                "Sea level pressure "+str(round(sea_level_pressure/1000, 1))+\
                            ' kPa, ' + str(round(sea_level_pressure/101325*100,1)) + "% of Earth's"+'\n'
                title_text += "Total population " + '{:,.0f}'.format(self.total_population) + \
                                " | Earth's 2020 population 7,794,798,739" +' \n'                
        
        elif title=='short':
            
            if maptype=='altitude':
                title_text = 'Altitude map\n'+\
                             ' '*10
            elif maptype=='temperature':
                title_text = 'Temperature map\n'+\
                             ' '*10
            elif maptype=='pressure':
                title_text = 'Pressure map\n'+\
                             ' '*10
            elif maptype=='precipitations':
                title_text = 'Precipitations map\n'+\
                             ' '*10
            elif maptype=='climate':
                title_text = 'Climate zones map\n'+\
                             ' '*10
            elif maptype=='natural_colors':
                title_text = 'Natural colors map\n'+\
                             ' '*10
            elif maptype=='radiation':
                title_text = 'Radiation map\n'+\
                             ' '*10
            elif maptype=='remoteness':
                title_text = 'Coast distance map\n'+\
                             ' '*10
            elif maptype=='pop_den':
                title_text = 'Population density map\n'+\
                             ' '*10
                
        else:
            title_text='' 
        
        return title_text
    
    def make_hillshade_map(self, i, projection='eck4', title_text='', angle_lon=0, angle_lat=0, read_file=True, verbose=False, 
                           caption=True, facecolor='white', view=''):
        
        start_time = time.time()   
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        extend='neither'#"max"#both
        font_color = '#FFFFFF'

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)    
            
        plt.title(title_text, color=font_color)
        m.contourf(self.lons, self.lats, self.hillshade, cmap=plt.get_cmap('gray'), alpha=0.3, latlon=True)
        
        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'

        cbar=1
        key=1
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)
        
        if caption:
            self.add_caption(font_color, projection)
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, maptype='hillshade', 
                                 read_file=read_file)
        
        if verbose:
            print('Hillshade map', round(time.time()-start_time,1), 'seconds')
        
        return img
    
    def make_shade_map(self, i, projection='eck4', title_text='', angle_lon=0, angle_lat=0, read_file=True, verbose=False, 
                           caption=True, facecolor='white', view=''):
        
        start_time = time.time()   
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        extend='neither'#"max"#both
        font_color = '#FFFFFF'

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)   
        
        self.shade = haversine_distances(np.array((self.lons.flatten()/360*2*math.pi, self.lats.flatten()/360*2*math.pi)).T, 
                                         np.array([[lon_0/360*2*math.pi, lat_0/360*2*math.pi]]))
        self.shade = math.pi/2 - self.shade
        self.shade[self.shade<0] = 0
        self.shade = np.cos(self.shade)**6
        self.shade = self.shade.reshape(self.lats.shape)
            
        plt.title(title_text, color=font_color)
        m.contourf(self.lons, self.lats, self.shade, cmap=plt.get_cmap('binary'), alpha=1.0, latlon=True, levels=100)
        
        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'

        cbar=1
        key=1
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)
        
        if caption:
            self.add_caption(font_color, projection)
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, maptype='shade', 
                                 read_file=read_file)
        
        if verbose:
            print('Shade map', round(time.time()-start_time,1), 'seconds')
        
        return img
        
    def make_altitude_map(self, i, projection='eck4', title='long', angle_lon=0, angle_lat=0, read_file=True, verbose=False, 
                          caption=True, view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        start_time = time.time()
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        level_step = self.params['level_step']
        level_step_map = self.params['level_step_map']
        bottom = int(-sl//level_step)*level_step
        up = -int((-self.max_delta+sl)//level_step)*level_step
        levels = list(range(bottom, up+1, level_step))
        levels_map = list(range(bottom, up+1, level_step_map))
        extend='neither'#"max"#both
        alpha=1.0
        font_color = '#333333'
        facecolor = 'white'

        colors_undersea = cm_delta(np.linspace(0.0, 0.40, -bottom//level_step_map))
        if up > self.highland: 
            colors_land = cm_europe_4(np.linspace(0, 1, self.highland//level_step_map))
            colors_highland = cm_des2(np.linspace(0.66, 1, (up-self.highland)//level_step_map))
            colors = np.vstack((colors_undersea, colors_land, colors_highland))
        else:
            colors_land = cm_europe_4(np.linspace(0, 1, self.highland//level_step_map))
            colors = np.vstack((colors_undersea, colors_land))

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)                           
        
        title_text = self.make_title_text(maptype='altitude', title=title, w=w, sl=sl, ss=ss)
        plt.title(title_text)
 
        m.contourf(self.lons, self.lats, self.heights-sl, 
                   colors=colors, levels=levels_map, extend=extend, latlon=True, alpha=alpha)
        
        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'

        cbar = m.colorbar(ax=ax, boundaries=levels, ticks=levels, drawedges=False, size=size, 
                          extend='neither', extendfrac=None) #для ortho сделать больший %
        cbar.ax.set_xticklabels(levels[::1],rotation=90)
        cbar.solids.set_edgecolor("face")
        cbar.ax.set_title('m')
        key=1
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)
        
        if caption:
            self.add_caption(font_color, projection)
            
        if show_hillshade:    
            hs = self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
            #self.make_hillshade_map(i=i, projection=projection, title_text=title_text, angle_lon=angle_lon, angle_lat=angle_lat, 
            #                             read_file=True, verbose=verbose, caption=caption, facecolor=facecolor, view=view)
        else:
            hs = None
            
        if show_shade:    
            shade = self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
            #self.make_hillshade_map(i=i, projection=projection, title_text=title_text, angle_lon=angle_lon, angle_lat=angle_lat, 
            #                             read_file=True, verbose=verbose, caption=caption, facecolor=facecolor, view=view)
        else:
            shade = None
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, maptype='altitude', 
                                 read_file=read_file, hs=hs, shade=shade)
        
        if verbose:
            print('Altitude map', round(time.time()-start_time,1), 'seconds')
        
        return img
            
    def make_temperature_map(self, i, solar_irradiance = 1361, sea_level_pressure = 101325, projection='eck4', 
                             title='long', angle_lon=0, angle_lat=0, read_file=True, verbose=False, caption=True, 
                             view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        start_time = time.time()
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        #self.temperatures[np.isnan(self.temperatures)] = 1000
        
        real_min_temp = -int(math.ceil((-self.temperatures[self.temperatures<1000].min()-30)/12))*12-30
        real_max_temp = int(math.ceil((self.temperatures[self.temperatures<1000].max()-30)/12))*12+30
        bottom_list = list(range(real_min_temp,-30,4))
        main_list = list(range(-30, 31, 1))
        upper_list = list(range(30+4,real_max_temp+1,4))
        #print(real_min_temp, real_max_temp, bottom_list, main_list, upper_list)
        levels = bottom_list+main_list+upper_list+[999]#, 1001]
        colors = np.vstack((cmap_windy_temp(np.linspace(0.0, 0.36, len(bottom_list))), cmap_windy_temp(np.linspace(0.36, 0.85, len(main_list))),
                            cmap_windy_temp(np.linspace(0.85, 1.0, len(upper_list))), cm_wiki(np.linspace(0.3, 0.3, 1)), #cm_wiki(np.linspace(0.3, 0.3, 1))
                           ))
        levels_for_show = list(range(real_min_temp,-30,12))+list(range(-30,31,3))+list(range(30+12,real_max_temp+1,12))+[999]#, 1001]
    
        extend="max"#both
        alpha=1.0
        font_color = '#333333'

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)  
        m.drawmapboundary(fill_color='#C6ECFF')
            
        title_text = self.make_title_text(maptype='temperature', title=title, w=w, sl=sl, ss=ss, 
                                  solar_irradiance = solar_irradiance, sea_level_pressure = sea_level_pressure)
        plt.title(title_text)

        m.contourf(self.lons, self.lats, self.temperatures, colors=colors, levels=levels, 
                   extend=extend, latlon=True, alpha=alpha)

        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'

        cbar = m.colorbar(ax=ax, boundaries=levels_for_show[:-1], size=size, extendfrac=0,
                          ticks=levels_for_show[:-1], drawedges=True, alpha=alpha)
        cbar.dividers.set_linewidth(0)
        cbar.ax.set_xticklabels(levels_for_show[:-1], rotation=90)
        cbar.ax.set_title('°C')
        key=1    
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)

        if caption:
            self.add_caption(font_color, projection)
            
        if show_hillshade:    
            hs = self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
            #self.make_hillshade_map(i=i, projection=projection, title_text=title_text, angle_lon=angle_lon, angle_lat=angle_lat, 
            #                             read_file=True, verbose=verbose, caption=caption, facecolor=facecolor, view=view)
        else:
            hs = None
            
        if show_shade:    
            shade = self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
            #self.make_hillshade_map(i=i, projection=projection, title_text=title_text, angle_lon=angle_lon, angle_lat=angle_lat, 
            #                             read_file=True, verbose=verbose, caption=caption, facecolor=facecolor, view=view)
        else:
            shade = None
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, 
                                 solar_irradiance=solar_irradiance, sea_level_pressure=sea_level_pressure, 
                                 maptype='temperature', read_file=read_file, hs=hs, shade=shade)
        
        if verbose:
            print('Temperature map', round(time.time()-start_time,1), 'seconds')
        
        return img
    
    def make_pressure_map(self, i, solar_irradiance = 1361, sea_level_pressure = 101325, projection='eck4', 
                          title='long', angle_lon=0, angle_lat=0, read_file=True, verbose=False, caption=True, 
                          view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        start_time = time.time()
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        self.pressures = self.pressures/1000
        #self.pressures[np.isnan(self.pressures)] = 15005
    
        levels = list(range(0,201,5))+[15000]
        #colors = np.vstack((cmap_windy_pressure(np.linspace(0.3, 0.95, len(levels)-1)), cm_wiki(np.linspace(0.3, 0.3, 1))))
        colors = np.vstack((cmap_windy_pressure(np.linspace(0.45, 0.8, len(levels)-1)), cm_wiki(np.linspace(0.3, 0.3, 1))))
        levels_for_show = list(range(0,201,10))+[15000]
    
        extend="max"#both
        alpha=1.0
        font_color = '#333333'

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)
        m.drawmapboundary(fill_color='#C6ECFF')
        
        title_text = self.make_title_text(maptype='pressure', title=title, w=w, sl=sl, ss=ss, 
                                  solar_irradiance = solar_irradiance, sea_level_pressure = sea_level_pressure)
        plt.title(title_text)
        m.contourf(self.lons, self.lats, self.pressures, 
                   colors=colors, levels=levels, extend=extend, latlon=True, alpha=alpha)

        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'
        
        cbar = m.colorbar(ax=ax, boundaries=levels_for_show[:-1], size=size, extendfrac=0,
                          ticks=levels_for_show[:-1], drawedges=True, alpha=alpha)
        cbar.dividers.set_linewidth(0)
        cbar.ax.set_xticklabels(levels_for_show[:-1], rotation=90)
        cbar.ax.set_title('kPa')
        key=1    
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)

        if caption:
            self.add_caption(font_color, projection)
            
        if show_hillshade:    
            hs = self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
            #self.make_hillshade_map(i=i, projection=projection, title_text=title_text, angle_lon=angle_lon, angle_lat=angle_lat, 
            #                             read_file=True, verbose=verbose, caption=caption, facecolor=facecolor, view=view)
        else:
            hs = None
            
        if show_shade:    
            shade = self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
            #self.make_hillshade_map(i=i, projection=projection, title_text=title_text, angle_lon=angle_lon, angle_lat=angle_lat, 
            #                             read_file=True, verbose=verbose, caption=caption, facecolor=facecolor, view=view)
        else:
            shade = None
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, 
                                 solar_irradiance=solar_irradiance, sea_level_pressure=sea_level_pressure, 
                                 maptype='pressure', read_file=read_file, hs=hs, shade=shade)
        
        if verbose:
            print('Pressure map', round(time.time()-start_time,1), 'seconds')
        
        return img
    
    def make_precipitations_map(self, i, solar_irradiance = 1361, sea_level_pressure = 101325, projection='eck4', 
                                title='long', angle_lon=0, angle_lat=0, read_file=True, verbose=False, caption=True, 
                                view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        start_time = time.time()
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        #self.precipitations[np.isnan(self.precipitations)] = 50005
    
        levels = list(range(0,3001,100))+[50000]
        #colors = np.vstack((cmap_windy_precipation(np.linspace(0.35, 1.0, len(levels)-1)), cm_wiki(np.linspace(0.3, 0.3, 1))))
        colors = np.vstack((cm_rain(np.linspace(0.15, 1.0, len(levels)-1)), cm_wiki(np.linspace(0.3, 0.3, 1))))
        levels_for_show = list(range(0,3001,100))+[50000]
    
        extend="max"#both
        alpha=1.0
        font_color = '#333333'

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)
        m.drawmapboundary(fill_color='#C6ECFF')
        
        title_text = self.make_title_text(maptype='precipitations', title=title, w=w, sl=sl, ss=ss, 
                                  solar_irradiance = solar_irradiance, sea_level_pressure = sea_level_pressure)
        plt.title(title_text)
        m.contourf(self.lons, self.lats, self.precipitations, 
                   colors=colors, levels=levels, extend=extend, latlon=True, alpha=alpha)

        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'
        
        cbar = m.colorbar(ax=ax, boundaries=levels_for_show[:-1], size=size, extendfrac=0,
                          ticks=levels_for_show[:-1], drawedges=True, alpha=alpha)
        cbar.dividers.set_linewidth(0)
        cbar.ax.set_xticklabels(levels_for_show[:-1], rotation=90)
        cbar.ax.set_title('mm')
        key=1    
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)

        if caption:
            self.add_caption(font_color, projection)
            
        if show_hillshade:    
            hs = self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
            #self.make_hillshade_map(i=i, projection=projection, title_text=title_text, angle_lon=angle_lon, angle_lat=angle_lat, 
            #                             read_file=True, verbose=verbose, caption=caption, facecolor=facecolor, view=view)
        else:
            hs = None
            
        if show_shade:    
            shade = self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            shade = None
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, 
                                 solar_irradiance=solar_irradiance, sea_level_pressure=sea_level_pressure, 
                                 maptype='precipitations', read_file=read_file, hs=hs, shade=shade)
        
        if verbose:
            print('Precipitations map', round(time.time()-start_time,1), 'seconds')
        
        return img
    
    def make_climate_map(self, i, solar_irradiance = 1361, sea_level_pressure = 101325, projection='eck4', 
                         title='long', angle_lon=0, angle_lat=0, read_file=True, verbose=False, caption=True, 
                         view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        start_time = time.time()
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        #self.climate[np.isnan(self.climate)] = 0
        #self.climate = self.climate + 0.5
    
        levels = list(range(0,10))
        colors = cmap_climate
        levels_for_show = levels
    
        extend="max"#both
        alpha=1.0
        font_color = '#333333'

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)
        m.drawmapboundary(fill_color='grey')
        
        title_text = self.make_title_text(maptype='climate', title=title, w=w, sl=sl, ss=ss, 
                                  solar_irradiance = solar_irradiance, sea_level_pressure = sea_level_pressure)
        plt.title(title_text)
        
        #m.contourf(self.lons, self.lats, self.climate+0.1, 
        #           colors=colors, levels=levels, extend=extend, latlon=True, alpha=alpha)
        
        for c in range(9):
            mask = self.climate != c
            if self.climate.shape[0]*self.climate.shape[1] - mask.astype(int).sum()>0:
                m.contourf(np.ma.MaskedArray(self.lons, mask), np.ma.MaskedArray(self.lats, mask) , 
                           np.ma.MaskedArray(self.climate+0.1, mask), 
                           colors=colors[c:c+1]+colors[c:c+1], levels=levels[c:c+2], extend=extend, latlon=True, alpha=alpha)

        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'

        mask = self.climate != 10
        m.contourf(np.ma.MaskedArray(self.lons, mask), np.ma.MaskedArray(self.lats, mask) , 
                    np.ma.MaskedArray(self.climate+0.1, mask), 
                    colors=colors[1:], levels=levels[1:], extend=extend, latlon=True, alpha=alpha)
        cbar = m.colorbar(ax=ax, boundaries=levels_for_show[1:], size=size, extendfrac=0,
                          ticks=levels_for_show[1:], drawedges=True, alpha=alpha)
        cbar.dividers.set_linewidth(0)
        loc    = [l + .5 for l in levels[1:] ]
        cbar.set_ticks(loc)
        cbar.ax.set_yticklabels(self.labels_text[1:], rotation=0)
        key=1    
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)

        if caption:
            self.add_caption(font_color, projection)
            
        if show_hillshade:    
            hs = self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            hs = None
            
        if show_shade:    
            shade = self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            shade = None
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, 
                                 solar_irradiance=solar_irradiance, sea_level_pressure=sea_level_pressure, 
                                 maptype='climate', read_file=read_file, hs=hs, shade=shade)
        
        if verbose:
            print('Climate zones map', round(time.time()-start_time,1), 'seconds')
        
        return img

    def make_natural_colors_map(self, i, solar_irradiance = 1361, sea_level_pressure = 101325, projection='eck4', title='long', 
                                angle_lon=0, angle_lat=0, read_file=True, verbose=False, caption=True, view='space', 
                                show_hillshade=True, show_shade=False, purple=False):
        
        #self.calc_climate(climate_prediction_type='neigh_detailed', verbose=verbose) ###################
        
        #################
            #self.climate_colors = np.random.randint(0, 100, (int(self.climate.shape[0]/2), int(self.climate.shape[1]/2)), dtype=np.uint8)
            #self.climate_colors = cv.resize(self.climate_colors, dsize=(self.climate.shape[1], self.climate.shape[0]), interpolation=cv.INTER_LINEAR)
        #self.climate_colors = np.random.randint(0, 100, self.climate.shape)
        
        #self.climate_thresholds = np.zeros(tuple(list(self.climate.shape)+[5]))
        #for c in range(8):
            #self.climate_thresholds[self.climate==c] = np.array(thresholds_dict[str(c)], dtype=np.uint8)

        #self.climate_resulted_colors = np.zeros(self.climate.shape)
        #for j in range(5):
            #if j==0:
                #self.climate_resulted_colors[self.climate_colors<=self.climate_thresholds[:,:,j]] = j
            #else:
                #self.climate_resulted_colors[(self.climate_colors<=self.climate_thresholds[:,:,j])&(self.climate_colors>self.climate_thresholds[:,:,j-1])] = j

        #self.climate_resulted_colors = self.climate*5+self.climate_resulted_colors            
        ##################    
            
        start_time = time.time()
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        #self.climate[np.isnan(self.climate)] = 0
        #self.climate = self.climate + 0.5
        
        if purple:
            colors = colors_list_purple
        else:
            colors = cmap_natural_colors_most_pop_8#cmap_natural_colors_pop#cmap_natural_colors#_full
        levels = list(range(0,len(colors)+1))        
        levels_for_show = levels
    
        extend="max"#both
        alpha=1.0
        font_color = '#FFFFFF'
        facecolor = 'white'

        fig = plt.figure(figsize=figsize)        
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            #caption=False
            #fig.patch.set_facecolor('#000000')
            title=''
            #ax.set_facecolor('#000000')
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)
        m.drawmapboundary(fill_color='#4c4c4c')#'darkslategrey')
        #http://www.html-color-names.com/dimgray.php
        
        title_text = self.make_title_text(maptype='natural_colors', title=title, w=w, sl=sl, ss=ss, 
                                  solar_irradiance = solar_irradiance, sea_level_pressure = sea_level_pressure)
        plt.title(title_text)
        #m.contourf(self.lons, self.lats, self.climate_resulted_colors+0.1, 
        #           colors=colors, levels=levels, extend=extend, latlon=True, alpha=alpha)
        for c in range(len(colors)):
            mask = self.climate != c
            if self.climate.shape[0]*self.climate.shape[1] - mask.astype(int).sum()>0:
                m.contourf(np.ma.MaskedArray(self.lons, mask), np.ma.MaskedArray(self.lats, mask) , 
                           np.ma.MaskedArray(self.climate+0.1, mask), 
                           colors=colors[c:c+1]+colors[c:c+1], levels=levels[c:c+2], extend=extend, latlon=True, alpha=0.8)
            #mask = self.climate_resulted_colors != c
            #m.contourf(np.ma.MaskedArray(self.lons, mask), np.ma.MaskedArray(self.lats, mask) , 
            #           np.ma.MaskedArray(self.climate_resulted_colors+0.1, mask), 
            #           colors=colors[c:c+1]+colors[c:c+1], levels=levels[c:c+2], extend=extend, latlon=True)#, alpha=0.8)
        
        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'

            #mask = self.climate != 100
            #m.contourf(np.ma.MaskedArray(self.lons, mask), np.ma.MaskedArray(self.lats, mask) , 
            #            np.ma.MaskedArray(self.climate+0.1, mask), 
            #            colors=colors[1:], levels=levels[1:], extend=extend, latlon=True, alpha=alpha)
            #cbar = m.colorbar(ax=ax, boundaries=levels_for_show[1:], size=size, extendfrac=0, 
            #                  ticks=levels_for_show[1:], drawedges=True, alpha=alpha)
            #cbar.dividers.set_linewidth(0)
            #loc    = [l + .5 for l in levels[1:] ]
            #cbar.set_ticks(loc)
            #cbar.ax.set_yticklabels(self.labels_text[1:], rotation=0)
            
        key=1
        cbar=1    
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)

        if caption:
            self.add_caption('#333333', projection)#font_color, projection)
            
        if show_hillshade:    
            hs = self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            hs = None
            
        if show_shade:    
            shade = self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            shade = None
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, 
                                 solar_irradiance=solar_irradiance, sea_level_pressure=sea_level_pressure, 
                                 maptype='natural_colors', read_file=read_file, facecolor=facecolor, hs=hs, shade=shade)
        
        if verbose:
            print('Natural colors map', round(time.time()-start_time,1), 'seconds')
        
        return img
    
    def make_radiation_map(self, i, solar_irradiance = 1361, sea_level_pressure = 101325, projection='eck4', 
                           title='long', angle_lon=0, angle_lat=0, read_file=True, verbose=False, caption=True, 
                           view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        start_time = time.time()
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        self.radiation[np.isnan(self.radiation)] = 505
        self.radiation[self.radiation<0.24] = 0.24
    
        levels = list(np.logspace(np.log10(0.24),np.log10(240), 37, base=10))+[500]
        colors = np.vstack((plt.get_cmap('RdYlGn_r')(np.linspace(0.0, 1.0, len(levels)-1)), cm_wiki(np.linspace(0.3, 0.3, 1))))
        levels_for_show = list(np.logspace(np.log10(0.24),np.log10(240), 19, base=10))+[500]
        bar_texts = ['{:.2f}'.format(l) for l in levels_for_show]
    
        extend="max"#both
        alpha=1.0
        font_color = '#333333'

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)
        
        title_text = self.make_title_text(maptype='radiation', title=title, w=w, sl=sl, ss=ss, 
                                  solar_irradiance = solar_irradiance, sea_level_pressure = sea_level_pressure)
        plt.title(title_text)
        m.contourf(self.lons, self.lats, self.radiation, 
                   colors=colors, levels=levels, extend=extend, latlon=True, alpha=alpha)

        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'
        
        cbar = m.colorbar(ax=ax, boundaries=levels_for_show[:-1], size=size, extendfrac=0,
                          ticks=levels_for_show[:-1], drawedges=True, alpha=alpha)
        cbar.dividers.set_linewidth(0)
        cbar.ax.set_xticklabels(levels_for_show[:-1], rotation=90)
        cbar.ax.set_title('mSv/yr')
        key=1    
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)

        if caption:
            self.add_caption(font_color, projection)
            
        if show_hillshade:    
            hs = self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            hs = None
            
        if show_shade:    
            shade = self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            shade = None
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, 
                                 solar_irradiance=solar_irradiance, sea_level_pressure=sea_level_pressure, 
                                 maptype='radiation', read_file=read_file, hs=hs, shade=shade)
        
        if verbose:
            print('Radiation map', round(time.time()-start_time,1), 'seconds')
        
        return img
    
    
    def make_pop_den_map(self, i, solar_irradiance = 1361, sea_level_pressure = 101325, projection='eck4', 
                         title='long', angle_lon=0, angle_lat=0, read_file=True, verbose=False, caption=True, 
                         view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        start_time = time.time()
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        pop_den_max = pd.DataFrame(pop_den_dict).max().max()
        na_pop_den = pop_den_max*1.1
        self.population_density[np.isnan(self.population_density)] = na_pop_den  
    
        levels = list(np.linspace(0, pop_den_max, 40))+[na_pop_den*0.99]
        colors = np.vstack((plt.get_cmap('cividis')(np.linspace(0.0, 1.0, len(levels)-1)), cm_wiki(np.linspace(0.3, 0.3, 1))))
        a=0.75
        colors[:-1] = colors[:-1]*a+1*(1-a)
        levels_for_show = list(np.linspace(0, pop_den_max, 20))+[na_pop_den*0.99]
        bar_texts = ['{: .1f}'.format(l) for l in levels_for_show]
    
        extend="max"#both
        alpha=1.0
        font_color = '#333333'

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)
        
        title_text = self.make_title_text(maptype='pop_den', title=title, w=w, sl=sl, ss=ss, 
                                  solar_irradiance = solar_irradiance, sea_level_pressure = sea_level_pressure)
        plt.title(title_text)
        m.contourf(self.lons, self.lats, self.population_density, 
                   colors=colors, levels=levels, extend=extend, latlon=True, alpha=alpha)

        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'
        
        cbar = m.colorbar(ax=ax, boundaries=levels_for_show[:-1], size=size, extendfrac=0,
                          ticks=levels_for_show[:-1], drawedges=True, alpha=alpha)
        cbar.dividers.set_linewidth(0)        
        cbar.ax.set_title('per km^2')
        cbar.ax.set_yticklabels(bar_texts[:-1], rotation=0)
        key=1    
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)

        if caption:
            self.add_caption(font_color, projection)
            
        if show_hillshade:    
            hs = self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            hs = None
            
        if show_shade:    
            shade = self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            shade = None
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, 
                                 solar_irradiance=solar_irradiance, sea_level_pressure=sea_level_pressure, 
                                 maptype='population_density', read_file=read_file, hs=hs, shade=shade)
        
        if verbose:
            print('Population density map', round(time.time()-start_time,1), 'seconds')
        
        return img
    
    
    def make_remoteness_map(self, i, solar_irradiance = 1361, sea_level_pressure = 101325, projection='eck4', 
                            title='long', angle_lon=0, angle_lat=0, read_file=True, verbose=False, caption=True, 
                            view='lines_toponyms', show_hillshade=True, show_shade=False):
        
        start_time = time.time()
        w = self.water_volumes[i]
        sl = self.sea_levels[i]
        ss = self.sea_shares[i]
        
        self.remoteness = self.remoteness / 1000
        self.remoteness[self.heights<sl] = np.nan
        na_remoteness = self.remoteness[~np.isnan(self.remoteness)].max()*1.5
        self.remoteness[np.isnan(self.remoteness)] = na_remoteness
        upper_limit = math.ceil(na_remoteness/1.5/100)*100
    
        levels = list(np.linspace(0, upper_limit, 40))+[na_remoteness*0.99]
        colors = np.vstack((plt.get_cmap('viridis')(np.linspace(0.0, 1.0, len(levels)-1)), cm_wiki(np.linspace(0.3, 0.3, 1))))
        levels_for_show = list(np.linspace(0, upper_limit, 20))+[na_remoteness*0.99]
        bar_texts = ['{: .0f}'.format(l) for l in levels_for_show]
    
        extend="max"#both
        alpha=1.0
        font_color = '#333333'
        facecolor = 'white'

        fig = plt.figure(figsize=figsize)
        grid = matplotlib.gridspec.GridSpec(5565, 5859)
        ax = fig.add_subplot(grid[39:5565-39, 186:5859-186])
        if view.find('space')>-1:
            facecolor='black'
            caption=False
            title=''
        else:
            facecolor='white'
        lon_0 = angle_lon%360
        lat_0 = angle_lat%360
        m = Basemap(projection=projection, resolution='l', lon_0=lon_0, lat_0=lat_0)
        
        title_text = self.make_title_text(maptype='remoteness', title=title, w=w, sl=sl, ss=ss, 
                                  solar_irradiance = solar_irradiance, sea_level_pressure = sea_level_pressure)
        plt.title(title_text)
        m.contourf(self.lons, self.lats, self.remoteness, 
                   colors=colors, levels=levels, extend=extend, latlon=True, alpha=alpha)

        if view.find('lines')>-1:            
            if projection == 'eck4':                
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
            elif projection == 'ortho':
                parallels = np.arange(-60.,90,30.)
                m.drawparallels(parallels,labels=[0,0,0,0], color='grey')
                meridians = np.arange(0.,360.,30.)
                m.drawmeridians(meridians,labels=[0,0,0,0], color='grey')
            elif projection == 'cyl':                
                parallels = np.arange(-90.,91,30.)
                m.drawparallels(parallels,labels=[1,0,0,0], color='grey')
                meridians = np.arange(0.,360.,60.)
                m.drawmeridians(meridians,labels=[0,0,0,1], color='grey')
                
        if view.find('toponyms')>-1:
            if projection=='eck4':
                self.add_toponyms_eck4(w, font_color) if view.find('toponyms')>-1 else None
            elif projection == 'ortho':
                self.add_toponyms_ortho(w, font_color, lon_0, lat_0) if view.find('toponyms')>-1 else None
            elif projection == 'cyl':
                self.add_toponyms_cyl(w, font_color) if view.find('toponyms')>-1 else None
        
        if projection=='ortho':
            size='2.0%'
        elif projection=='eck4' or projection=='cyl':
            size='4.0%'        
        
        cbar = m.colorbar(ax=ax, boundaries=levels_for_show[:-1], size=size, extendfrac=0,
                          ticks=levels_for_show[:-1], drawedges=True, alpha=alpha)
        cbar.dividers.set_linewidth(0)
        cbar.ax.set_title('km')
        cbar.ax.set_yticklabels(bar_texts[:-1], rotation=0)
        key=1    
        
        if projection == 'ortho':
            plt.annotate(' ', xy=(0.0, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(-0.2, -0.05), xycoords='axes fraction', color=facecolor)
            plt.annotate(' ', xy=(1.2, -0.05), xycoords='axes fraction', color=facecolor)
        
        if caption:
            self.add_caption(font_color, projection)
            
        if show_hillshade:    
            hs = self.hs_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            hs = None
            
        if show_shade:    
            shade = self.shade_dict[projection+'_'+view+'_'+str(caption)+'_'+facecolor]
        else:
            shade = None
        
        img = self.save_map_file(fig, ax, key, cbar, i, w, sl, ss, angle_lon, projection=projection, 
                                 solar_irradiance=solar_irradiance, sea_level_pressure=sea_level_pressure, 
                                 maptype='coast_distance', read_file=read_file, hs=hs, shade=shade)
        
        if verbose:
            print('Coast distance map', round(time.time()-start_time,1), 'seconds')
        
        return img
    
    def title_frame(self, maps_list, maps_dict, i, w, ss , sl, sea_level_pressure, solar_irradiance, size):
        
        #heights = []
        #widths = []
        
        #frames_sizes = [maps_dict[i].shape for i in maps_dict if maps_dict[i] is not None]
        #max_width = max([i[1] for i in frames_sizes])
        #max_height = max([i[0] for i in frames_sizes])
        
        frame = np.ones((size[0], size[1], 3))*255
        
        font = cv.FONT_HERSHEY_DUPLEX
        
        if self.quality == '8K':
            line_step = 280
            font_size = 4
            start_h = 400
            start_v = 400
        elif self.quality == '4K':
            line_step = 140
            font_size = 2
            start_h = 200
            start_v = 200
        elif self.quality == 'FHD':
            line_step = 70
            font_size = 1
            start_h = 100
            start_v = 100
            
        cv.putText(frame, 'Terraforming '+self.planet, (int(size[1]//2-start_h*5), int(start_v*1)), font, font_size*2, (0, 0, 0), 2, cv.LINE_AA)
        start_v = start_v*2
        
        WEG_layer = round(w / self.full_square / 1000, 1)
        text = 'Sea share '+str(ss)+"% of "+self.planet+" total surface  |  Land share "+\
                            str(round(self.full_square*(100-ss)/(1.49*1e14), 1))+"% of Earth's land surface"
        cv.putText(frame, text, (start_h, start_v+line_step*0), font, font_size, (0, 0, 0), 2, cv.LINE_AA)
        
        text = 'Total water mass '+'%.02f'%(w/self.params['scale_for_title'])+\
                            str(self.params['scale_for_title'])[1:]+'kg  (WEG layer '+str(WEG_layer)+\
                            'm)  |  Max sea depth '+str(round((sl+self.min_new-self.min_orig)/1000,1))+\
                            'km  |  '+self.params['highest_point']+' '+\
                            str(round((self.max_delta-sl+self.max_orig-self.max_new)/1000, 1))+'km'
        cv.putText(frame, text, (start_h, start_v+line_step*1), font, font_size, (0, 0, 0), 2, cv.LINE_AA)
        
        if self.planet=='Mars':
            if w>=5*1e18 and w<=6*1e18:
                text = 'Current martian water '
            #elif w>=10*1e18 and w<=11*1e18:
            #    text = 'Current martian water deposits + Hyperion moon melted '
            elif w>=20*1e18 and w<=21*1e18:
                text = 'Past martian water '
            #elif w>=42*1e18 and w<=43*1e18:
            #    text = 'Current martian water deposits + Mimas moon melted '
            elif w>6*1e18:
                text = 'Current martian water + '+str(round((w-5*1e18)/(1e14)/1200,1))+\
                                    ' comets per month during 100 years (d 7km, density 600kg/m^3, mass 1e14kg) '
            elif w<5*1e18:
                text = ' '+str(round(w/(5*1e18)*100)) + '% of current martian water deposits melted '
            else:
                text = ''
        else:
            text = str(round((w-5*1e18)/(1e14)/1200,1))+\
                                ' comets per month during 100 years (d 7km, density 600kg/m^3, mass 1e14kg) '
        cv.putText(frame, text, (start_h, start_v+line_step*2), font, font_size, (0, 0, 0), 2, cv.LINE_AA)
        
        text = "Earth's rotation, axial tilt and albedo | Solar irradiance "+str(round((solar_irradiance)))+\
               ' w/m^2, ' + str(round(solar_irradiance/1361*100,1)) + '% of Earth, ' + \
               str(round(solar_irradiance/self.params['solar_irradiance']*100,1)) + '% of '+ self.planet
        cv.putText(frame, text, (start_h, start_v+line_step*3), font, font_size, (0, 0, 0), 2, cv.LINE_AA)
        
        text = "Average temperature " + str(round(self.temp_avg-273.15)) + "C (Earth's average 15C), "+ \
               'Min '+str(round(self.min_temperature))+'C, Max '+ \
               str(round(self.max_temperature))+'C'
        cv.putText(frame, text, (start_h, start_v+line_step*4), font, font_size, (0, 0, 0), 2, cv.LINE_AA)#°
        
        text = "Sea level pressure "+str(round(sea_level_pressure/1000,1))+' kPa, ' + \
               str(round(sea_level_pressure/101325*100,1)) +"% of Earth | Minimal pressure "+\
               str(round(self.min_pressure/1000,1))+' kPa,  '+ \
               str(round(self.min_pressure/1000/33.700*100)) +\
               "% of Mount Everest summit"
        cv.putText(frame, text, (start_h, start_v+line_step*5), font, font_size, (0, 0, 0), 2, cv.LINE_AA)
        
        text = "Mass of the Atmosphere " + str(round(sea_level_pressure/self.g_0*self.full_square/1e18,1)) + \
               'e+18kg, 75% of Nitrogen ('+str(round(sea_level_pressure/self.g_0*self.full_square*0.75/1e18,1))+\
               'e+18kg) and 25% of Oxygen(' +str(round(sea_level_pressure/self.g_0*self.full_square*0.25/1e18,1))+\
               'e+18kg)'
        cv.putText(frame, text, (start_h, start_v+line_step*6), font, font_size, (0, 0, 0), 2, cv.LINE_AA)
        
        text = "Average Earth-annual precipitations " +\
                str(int(round(self.average_precipation)))+"mm (Earth's average 740mm)"
        cv.putText(frame, text, (start_h, start_v+line_step*7), font, font_size, (0, 0, 0), 2, cv.LINE_AA)
                
        text = "Coast distance: average " + '{:,.0f}'.format(self.average_remoteness)+' km, max ' + \
                        '{:,.0f}'.format(self.max_remoteness) + \
                    "km (Earth's average 236km, Earth's max 3,282km)"
        cv.putText(frame, text, (start_h, start_v+line_step*8), font, font_size, (0, 0, 0), 2, cv.LINE_AA)        
                
        text = "Total population " + '{:,.0f}'.format(self.total_population) + \
                        " | Earth's 2020 population 7,794,798,739"
        cv.putText(frame, text, (start_h, start_v+line_step*9), font, font_size, (0, 0, 0), 2, cv.LINE_AA)
        
        if self.planet == 'Mars':
            text = "Radiation, mSv/yr: average " + str(round(self.mean_radiation,2))+", min "+ \
                   str(round(self.min_radiation,2)) + ", max "+ \
                   str(round(self.max_radiation,2)) + \
                   " (Earth's: sea level - 0.5, 10 km height - 50)"
            cv.putText(frame, text, (start_h, start_v+line_step*10), font, font_size, (0, 0, 0), 2, cv.LINE_AA)                
        
        return frame