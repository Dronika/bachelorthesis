"""
Author:         Dronika Debisarun
Student Number: 10735968
Course:         Bachelor Project
Code:           1
Subject:        Crop and Filter stars to dataset used in analysis
"""
import time
start = time.time()
import numpy as np
from numpy import savetxt
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import CircularAperture, CircularAnnulus, aperture_photometry

# Load files necessary 
def data(file_name):
    data = np.loadtxt(file_name,delimiter=',')
    return data 

def get_data(path):
    image = fits.getdata(path)
    return image

# Load images
def load(object_name):
    image_L = get_data('C:/Bachelor_Project/Data/{}_L_Master_Final.fit'.format(object_name))
    image_V = get_data('C:/Bachelor_Project/Data/{}_V_Master_Final.fit'.format(object_name))
    image_B = get_data('C:/Bachelor_Project/Data/{}_B_Master_Final.fit'.format(object_name))
    image_Ha = get_data('C:/Bachelor_Project/Data/{}_Ha_Master_Final.fit'.format(object_name))
    image_R = get_data('C:/Bachelor_Project/Data/{}_R_Master_Final.fit'.format(object_name))
    return image_L,image_V,image_B,image_R,image_Ha

# Parameters
def parameters(object_name):
    if object_name == 'M51':
        center_guess = [2010,2060] 
        crop_width = 1000
        r_galaxy = 500
        star_width = 20
        star1, star1_magn_B, star1_magn_V, star1_magn_R = 18, 13.8,12.4,12.1 #GPM 202.306244+47.237732 for M51 
        star2, star2_magn_B, star2_magn_V, star2_magn_R = 33, 12.350, 11.510,10.600 # TYC 3463-582-1 for M51

    elif object_name == 'M81':
        center_guess = [2100,1900]
        crop_width = 1000
        r_galaxy = 1000
        star_width = 15
        star1, star1_magn_B, star1_magn_V, star1_magn_R = 44, 13.15,11.85,11.6 #GSC 04383-00224
        star2, star2_magn_B, star2_magn_V, star2_magn_R = 21, 12.8,13.81,12.26 #GSC 04383-00613

    elif object_name == 'M91':
        center_guess = [2170,1900]
        crop_width = 1000
        r_galaxy = 400
        star_width = 15
        star1, star1_magn_B, star1_magn_V, star1_magn_R = 18, 11.5,10.33,0 #BD+15 2479 GEEN R MAGNITUDE!!!
        star2, star2_magn_B, star2_magn_V, star2_magn_R = 16, 11.87,12.75,0 #TYC 881-35-1 GEEN R MAGNITUDE!!!

    elif object_name == 'M101': 
        center_guess = [2050,1960]
        crop_width = 500
        r_galaxy = 1300
        star_width = 15
        star1, star1_magn_B, star1_magn_V, star1_magn_R = 22,12.62,12.44,11.2 #TYC 3852-1069-1
        star2, star2_magn_B, star2_magn_V, star2_magn_R = 54,12.08,11.48,11.2 #TYC 3852-1108-1

    elif object_name == 'M109':
        center_guess = [1900,1950]
        crop_width = 1000
        r_galaxy = 500
        star_width = 15
        star1, star1_magn_B, star1_magn_V, star1_magn_R = 33, 12.44,11.87,0 #TYC 3833-756-1 GEEN R MAGNITUDE!!!
        star2, star2_magn_B, star2_magn_V, star2_magn_R = 37, 11.89,11.55,0 #TYC 3833-778-1 GEEN R MAGNITUDE!!!

    elif object_name == 'NGC2403':
        center_guess = [2190,1770]
        crop_width = 1000
        r_galaxy = 900
        star_width = 20
        star1, star1_magn_B, star1_magn_V, star1_magn_R = 41,10.38,9.31,9.22 #BD+65 577
        star2, star2_magn_B, star2_magn_V, star2_magn_R = 68,12.42,11.47,0 #2MASS J07371706+6548390 GEEN R MAGNITUDE!!!

    elif object_name == 'NGC2787':
        center_guess = [1940,1975]
        crop_width = 1500
        r_galaxy = 200
        star_width = 10
        star1, star1_magn_B, star1_magn_V, star1_magn_R = 32,10.09,8.49,7.8 #HD 79518
        star2, star2_magn_B, star2_magn_V, star2_magn_R = 33,9.64,8.63,8.4 #HD 79300

    elif object_name == 'NGC2903':
        center_guess = [1910,1740]
        crop_width = 1000
        r_galaxy = 700
        star_width = 15
        star1, star1_magn_B, star1_magn_V, star1_magn_R = 4, 11.79,11.42,0 #TYC 1409-668-1 GEEN R MAGNITUDE!!!
        star2, star2_magn_B, star2_magn_V, star2_magn_R = 57, 12.78,11.79,0 #TYC 1409-848-1 GEEN R MAGNITUDE!!!

    elif object_name == 'NGC4725':
        center_guess = [2165,1940]
        crop_width = 1000
        r_galaxy = 700
        star_width = 15
        star1, star1_magn_B, star1_magn_V, star1_magn_R = 42, 11.98,11.59,10.90 #TYC 1990-1204-1
        star2, star2_magn_B, star2_magn_V, star2_magn_R = 33, 12.90,11.40,11.50 #GPM 192.785344+25.526146

    return center_guess,crop_width, r_galaxy, star_width, star1, star1_magn_B, star1_magn_V, star1_magn_R, star2, star2_magn_B, star2_magn_V, star2_magn_R

object_name = 'NGC2787'

print('Starting to process object', object_name)

pixel_scale = 0.4448
shape = 4096
center_guess_width = 20
center_guess,crop_width,r_galaxy,star_width,star1, star1_magn_B, star1_magn_V, star1_magn_R, star2, star2_magn_B, star2_magn_V, star2_magn_R = parameters(object_name)
image_L,image_V,image_B,image_R,image_Ha = load(object_name)

# Find center in image using a guess
def find_centre(file_name,x_guess,y_guess,center_crop_width):
    # Making a small crop of the area around the guess to accurately determine center
    center_guess = [x_guess,y_guess]
    center_crop_data = file_name[center_guess[1]-center_crop_width:center_guess[1]+center_crop_width,center_guess[0]-center_crop_width:center_guess[0]+center_crop_width]

    # Blur and sum pixelcounts
    blur_image = gaussian_filter(center_crop_data,sigma=1)

    x_sum = np.sum(blur_image,axis=0)
    y_sum = np.sum(blur_image,axis=1)

    # Center has the highest pixelcounts 
    center_x = np.argmax(np.array(x_sum))
    center_y = np.argmax(np.array(y_sum))
    center = [center_x,center_y]

    # Convert coordinates back to original coordinates of entire image
    center[0] = center_guess[0] + (center[0]-center_crop_width)
    center[1] = center_guess[1] + (center[1]-center_crop_width)
    center = [center[0],center[1]]
    return center

# Crop image to save
def crop_image(file_name,crop_width,center):
    y,x = file_name.shape
    y_radius = y/2.
    x_radius = x/2.
    crop_data = file_name[center[1]-(int(y_radius) - crop_width):center[1]+(int(y_radius) - crop_width),center[0]-(int(x_radius) - crop_width):center[0]+(int(x_radius) - crop_width)]
    return crop_data

center = find_centre(image_L,center_guess[0],center_guess[1],center_guess_width)

crop_L = crop_image(image_L,crop_width,center)
crop_R = crop_image(image_R,crop_width,center)
crop_V = crop_image(image_V,crop_width,center)
crop_B = crop_image(image_B,crop_width,center)
crop_Ha = crop_image(image_Ha,crop_width,center)

def save(object_name):
    savetxt('{}_crop_L.csv'.format(object_name),crop_L,delimiter=',')
    savetxt('{}_crop_R.csv'.format(object_name),crop_R,delimiter=',')
    savetxt('{}_crop_B.csv'.format(object_name),crop_B,delimiter=',')
    savetxt('{}_crop_V.csv'.format(object_name),crop_V,delimiter=',')
    savetxt('{}_crop_Ha.csv'.format(object_name),crop_Ha,delimiter=',')

save(object_name)
print('crops saved after',np.round(time.time()-start),'seconds')

# Detect stars and flux area of stars in luminance filter with DAOstarfinder
def detecting_stars(file_name,threshold):
    mean, median, std = sigma_clipped_stats(file_name,sigma=sigma)
    mean = mean

    sources = DAOStarFinder(fwhm=fwhm,threshold=threshold*std)(file_name - median)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'

    sources = np.asarray(sources)
    savetxt('{}_sources.csv'.format(object_name),sources,delimiter=',')

    # plt.imshow(file_name,cmap='Greys_r',origin='lower',vmin=np.mean(file_name),vmax=np.mean(file_name)+np.std(file_name))

    positions = np.transpose((sources['xcentroid'],sources['ycentroid']))
    apertures = CircularAperture(positions, r=4*fwhm)
    annulus_aperture = CircularAnnulus(positions,r_in=6*fwhm,r_out=8*fwhm)
    # apertures.plot(color='yellow',lw=1.5,alpha=0.5)
    # annulus_aperture.plot(color='purple',lw=1.5,alpha=0.5)

    annulus_masks = annulus_aperture.to_mask(method='center')
    
    # plt.show()
    
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(file_name)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)
    phot = aperture_photometry(file_name, apertures)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * apertures.area
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    for col in phot.colnames:
        phot[col].info.format = '%.8g'
    
    phot = np.asarray(phot)
    # savetxt('{}_phot.csv'.format(object_name),phot,delimiter=',')
    return phot,sources

# Universel parameters for star detection
sigma = 3.0
fwhm = 5.0
# sigma_ha_regions = 2.0
# l = [3,5,10]
threshold_for_starfilter = 5.0
threshold_for_calibration = 100.0
# offset_magn = 10

# Sources for calibration is used for the calibration of the color filters
# Sources for starfilter is used for removing the datapoints of the stars in the image from total data
sources_for_calibration = detecting_stars(image_L,threshold_for_calibration)[0]
sources_for_starfilter = detecting_stars(image_L,threshold_for_starfilter)[1]

# Calibration process
def image_calibration(file_name,position_star1,ijk_magnitude_star1,position_star2,ijk_magnitude_star2):
    mag_star_list = [[position_star1,ijk_magnitude_star1],[position_star2,ijk_magnitude_star2]]

    avg_instr_magn = 0

    for i in range(len(mag_star_list)):
        instr_magn_star = -2.5*math.log10(sources_for_calibration[mag_star_list[i][0]-1][6]) - mag_star_list[i][1]
        mag_star_list[i].append(instr_magn_star)
        avg_instr_magn = instr_magn_star + avg_instr_magn

    avg_instr_magn = avg_instr_magn/len(mag_star_list)

    image_calibrated = -2.5*np.log10(file_name) - avg_instr_magn + 2.5*np.log10(pixel_scale**2)

    print ('calibration calculated')
    return image_calibrated

# Calibrate color images
V_calibrated = image_calibration(crop_V,star1,star1_magn_V,star2,star2_magn_V)
B_calibrated = image_calibration(crop_B,star1,star1_magn_B,star2,star2_magn_B)
R_calibrated = image_calibration(crop_R,star1,star1_magn_R,star2,star2_magn_R)
print('Calibration completed after',np.round(time.time()-start),'seconds')

# Cropping overall data and convert 2D array to dataset used in further analysis
def making_the_great_list(offset_intensity):
    # Making a list of every coordinate and remember center coordinates
    y,x = np.indices((crop_L.shape))
    ycrop,xcrop = crop_L.shape
    ycentercrop = ycrop/2.
    xcentercrop = xcrop/2.
    centercrop = [xcentercrop,ycentercrop]

    # Lists
    the_greatest_list = []

    x_list = []
    y_list = []

    I_list_L = []
    I_list_B = []
    I_list_V = []
    I_list_R = []
    I_list_Ha = []

    magn_list_B = []    
    magn_list_V = []    
    magn_list_R = []    
    
    distance_to_center_list = []
    x_to_center_list = []
    y_to_center_list = []
    theta_list = []

    # Voor elk coordinaat de elementen bepalen en toevoegen aan de betreffende lijst
    # Median of every image
    median_L= np.median(crop_L)
    median_B= np.median(crop_B)
    median_V= np.median(crop_V)
    median_R= np.median(crop_R)
    median_Ha= np.median(crop_Ha)
    
    print ('calculating information per pixel...')
  
    # Fill lists for every element
    for i,j in np.ndindex(x.shape):
        # Coordinates
        x_list.append(x[i,j])
        y_list.append(y[i,j])
       
        # intensity values minus median with perhars a offset to only get positive values if necessary
        intensity_L = crop_L[i,j] - median_L + offset_intensity
        intensity_B = crop_B[i,j] - median_B + offset_intensity 
        intensity_V = crop_V[i,j] - median_V + offset_intensity 
        intensity_R = crop_R[i,j] - median_R + offset_intensity 
        intensity_Ha = crop_Ha[i,j] - median_Ha + offset_intensity 

        I_list_L.append(intensity_L)
        I_list_B.append(intensity_B)
        I_list_V.append(intensity_V)
        I_list_R.append(intensity_R)
        I_list_Ha.append(intensity_Ha)

        # Magnitude of color bands
        magn_list_B.append(B_calibrated[i,j])
        magn_list_V.append(V_calibrated[i,j])
        magn_list_R.append(R_calibrated[i,j])

        # Relative coordinates with center as origin
        x_to_center = x[i,j] - xcentercrop
        y_to_center = y[i,j] - ycentercrop
        x_to_center_list.append(x_to_center)
        y_to_center_list.append(y_to_center)

        # Radius to center for every element
        r = np.round(math.sqrt((x_to_center)**2 + (y_to_center)**2),decimals=0)
        distance_to_center_list.append(r)
        
        # The appropriate angle of coordinate to center
        if x_to_center > 0 and y_to_center > 0:
            theta_list.append(math.atan(y_to_center/x_to_center))
        elif x_to_center < 0 and y_to_center > 0:
            theta_list.append(math.atan(abs(x_to_center)/y_to_center) + (0.5*math.pi))
        elif x_to_center < 0 and y_to_center < 0:
            theta_list.append(math.atan(abs(x_to_center)/abs(y_to_center)) + (math.pi))
        elif x_to_center > 0 and y_to_center < 0:
            theta_list.append(math.atan(x_to_center/abs(y_to_center)) + (1.5*math.pi))
        elif x_to_center == 0 and y_to_center == 0:
            theta_list.append(0)
        elif x_to_center > 0 and y_to_center == 0:
            theta_list.append(0)
        elif x_to_center == 0 and y_to_center > 0:
            theta_list.append(0.5*math.pi)
        elif x_to_center < 0 and y_to_center == 0:
            theta_list.append(math.pi)
        elif x_to_center == 0 and y_to_center < 0:
            theta_list.append(1.5*math.pi)
    
    # Adding all filled lists to the overall datalist which is used in further analysis
    the_greatest_list.append(x_list)
    the_greatest_list.append(y_list)
    the_greatest_list.append(I_list_L)
    the_greatest_list.append(I_list_B)
    the_greatest_list.append(I_list_V)
    the_greatest_list.append(I_list_R)
    the_greatest_list.append(I_list_Ha)
    the_greatest_list.append(magn_list_B)
    the_greatest_list.append(magn_list_V)
    the_greatest_list.append(magn_list_R)
    the_greatest_list.append(x_to_center_list)
    the_greatest_list.append(y_to_center_list)
    the_greatest_list.append(distance_to_center_list)
    the_greatest_list.append(theta_list)
    the_greatest_list = np.asarray(the_greatest_list)
    savetxt('{}_the_greatest_list.csv'.format(object_name),the_greatest_list,delimiter=',')
    return the_greatest_list,centercrop

great_list = making_the_great_list(0)
blabla = np.transpose(great_list[0])
print('greatest list shape:',great_list[0].shape, blabla.shape)
print('Greatest list completed after',np.round(time.time()-start),'seconds')

# From sources for starfilter, only remember coordinates
def star_list():
    coordinate_list = np.transpose([sources_starfilter[:,1]-(crop_width+center[0]-shape/2),sources_starfilter[:,2]-(crop_width+center[1]-shape/2)])
    coordinate_list = np.round(coordinate_list)
    return coordinate_list

sources_starfilter = data('{}_sources.csv'.format(object_name))
coordinate_list = star_list()
print('coordinate list:',len(coordinate_list)+1, 'stars found')

# Delete star datapoints of edges and center
del_list=[]
for i in range(len(coordinate_list)):
    if coordinate_list[i][1] < 0:
        del_list.append(i)
    elif coordinate_list[i][1] > 4096 -2*crop_width:
        del_list.append(i)
    elif coordinate_list[i][0] < 0:
        del_list.append(i)
    elif coordinate_list[i][0] > 4096 -2*crop_width:
        del_list.append(i)
    elif (abs(coordinate_list[i,0] -  np.int(crop_L.shape[0])/2) < 5*star_width and abs(coordinate_list[i,1] -  np.int(crop_L.shape[0])/2) < 5*star_width):
        del_list.append(i)

print(coordinate_list[3,0] - np.int(crop_L.shape[0])/2)
coordinate_list_filtered = np.delete(coordinate_list,del_list,0)

print('coordinate list filtered (edges outside crop and center). Stars left:', len(coordinate_list_filtered))

# Crop entire dataset based on the radius of galaxy
def smaller_list(file_name,r_galaxy):
    galaxy_list = file_name[:,file_name[12]<r_galaxy]
    return galaxy_list

print('max value r (before R galaxy filter):',np.max(great_list[0][12]))
print('filter with R_galaxy...')

smaller_list = smaller_list(great_list[0],r_galaxy)
print('max value r (After R galaxy filter):',np.max(smaller_list[12]))
print('shape of smaller list:',smaller_list.shape)

print('R_galaxy filter completed after',np.round(time.time()-start),'seconds')

# Make column of every star coordinate and surroundings
def star_column(file_name,star_file,center_galaxy_file,width,j):
    kolom14 = []
    for i in range(len(file_name)):
        
        if (abs(file_name[i,0] - star_file[j][0]) < width and abs(file_name[i,1] - star_file[j][1]) < width):
            if (star_file[j][0] == center_galaxy_file[0] and star_file[j][1] == center_galaxy_file[1]):
                kolom14.append(0)
            else:
                kolom14.append(1)
        else:
            kolom14.append(0)
            
    return np.asarray(kolom14)

# Delete selected star datapoints for dataset
def star_filter(file_name,star_file,center_galaxy_file,width):
    matrix = []
    for i in range(len(star_file)):
        column = star_column(file_name,star_file,center_galaxy_file,width,i)
        matrix.append(column)
        print('Removed star ',i+1,'from ', len(coordinate_list_filtered), 'total stars')
    np.transpose(matrix)

    matrix = np.asarray(matrix)
    transposed_matrix = np.transpose(matrix)

    star = []
    for i in range(len(transposed_matrix)):
        star.append(max(transposed_matrix[i]))

    full_list = np.column_stack((file_name,np.asarray(star)))

    clean_list_temp = np.delete(full_list, np.argwhere(full_list[:,14] == 1), axis=0)

    clean_list = np.delete(clean_list_temp,14, axis=1)
    return clean_list

print('running starfilter...')

star_width = 30
clean = star_filter(np.transpose(smaller_list),coordinate_list_filtered,great_list[1],star_width)
print('shape:',smaller_list.shape,clean.shape)

print('Star filter completed after',np.round(time.time()-start),'seconds')

# Save dataset 
print('saving clean list')
savetxt('{}_clean_list.csv'.format(object_name),clean,delimiter=',')

print('It took',np.round(time.time()-start),'seconds to complete')