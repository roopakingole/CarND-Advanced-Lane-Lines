# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 05:43:40 2019

@author: Roopak Ingole
"""



import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
import os
import collections
import math

# HYPERPARAMETERS
imgH = 720
imgW = 1280
winH = imgH/9
# Set minimum number of pixels found to recenter window
minpix = 50
# Choose the number of sliding windows
nwindows = np.int(imgH/winH)
# Set the width of the windows +/- margin
margin = 100
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
l_lines_collection = collections.deque([], 5)
r_lines_collection = collections.deque([], 5)
radius_col = collections.deque([],5)

def gen_objpoints(path,nx=9,ny=6):
    images = glob.glob(path)

    objpoints = []
    imgpoints = []
    
    #prepare obj points like (0,0,0), (1,0,0)...()
    objp = np.zeros([nx*ny,3], np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x,y coordinates
    
    for fname in images:
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
    return imgpoints,objpoints


def undistort(img, mtx, dist):
    # Use cv2.calibrateCamera() and cv2.undistort()
    #undist = np.copy(img)  # Delete this line
    #img = np.copy(img)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    #undist = np.copy(img)  # Delete this line
    #img = np.copy(img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist, mtx, dist

def calibrate_camera(path, test_img_path):
    imgpoints, objpoints = gen_objpoints(path)
    undist, mtx, dist = cal_undistort(mpimg.imread(test_img_path),objpoints,imgpoints)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "cam_calibration.p", "wb" ) )

def getCamCal():    
    dist_pickle = pickle.load( open( "cam_calibration.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx,dist

def prespectiveTransform(img, src, dst):
    #img = np.copy(img)
    img_size = (img.shape[1],img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped
    

def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    # 2) Convert to grayscale
    # 3) Find the chessboard corners
    # 4) If corners found: 
            # a) draw corners
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
            # e) use cv2.warpPerspective() to warp your image to a top-down view
    #delete the next two lines
    #M = None
    #warped = np.copy(img) 
    img_size = (img.shape[1],img.shape[0])
    #img = cv2.imread(fname)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        #plt.imshow(img)
        #print(corners)
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
        

        src = np.float32([corners[0],corners[nx-1],corners[-1],corners[-nx]])
        #print(src)
        #dst = np.float32([[462, 161],[1030,269],[1030,343],[462,274]])
        dst = np.float32([[100, 100],[img_size[0]-100,100],[img_size[0]-100,img_size[1]-100],[100,img_size[1]-100]])
        #dst = np.float32([[462, 161],[1030,161],[1034,343],[462,274]])
        
        M = cv2.getPerspectiveTransform(src, dst)
        #Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M

# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if(orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    abs_sobel = np.absolute(sobel)
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output  = np.zeros_like(scaled_sobel)
    binary_output [(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return binary_output
# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    img1 = np.copy(img) 
    gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    
    if(orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    
    abs_sobel = np.absolute(sobel)
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    binary_output  = np.zeros_like(scaled_sobel)
    binary_output [(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    
        #binary_output = np.copy(img) # Remove this line
    #img1 = np.copy(img) 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    
    abs_sobelx = np.square(sobelx)
    abs_sobely = np.square(sobely)
    abs_sobelxy = np.sqrt(abs_sobelx + abs_sobely)
    
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    binary_output  = np.zeros_like(scaled_sobel)
    binary_output [(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    #img1 = np.copy(img) 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_sobelxy = np.arctan2(abs_sobely , abs_sobelx)
    
    #scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))

    binary_output  = np.zeros_like(abs_sobelxy)
    binary_output [(abs_sobelxy >= thresh[0]) & (abs_sobelxy <= thresh[1])] = 1    
    return binary_output

def color_threshold(img, hsl='s', thd=(170,255)):
    #img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    
    # Threshold color channel
    binary = np.zeros_like(s_channel)
    if hsl == 'h':
        binary[(h_channel >= thd[0]) & (h_channel <= thd[1])] = 1    
    if hsl == 'l':
        binary[(l_channel >= thd[0]) & (l_channel <= thd[1])] = 1    
    if hsl == 's':
        binary[(s_channel >= thd[0]) & (s_channel <= thd[1])] = 1    

    return binary

def combined_thd(image):
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(15, 210))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(15, 210))
    mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(50, 200))
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    color_binary = color_threshold(image,hsl='s',thd=(100,255))
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))| (color_binary == 1)] = 1
    return combined

# Edit this function to create your own pipeline.
def color_n_gradient_thd(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    #img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,image.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,image.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

def convolutional_method(warped, window_width, window_height, margin):
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:
    
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)
    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
    	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
    	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
    	    # Add graphic points from window mask here to total pixels found 
    	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
    
        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

def measure_curvature_pixels(ploty, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    #ploty, left_fit, right_fit = generate_data()
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    #rCurve = np.sqrt((1 + (2Ay+B)**2)**3)/np.abs(2A)
    left_curverad = 0  ## Implement the calculation of the left line here
    right_curverad = 0  ## Implement the calculation of the right line here
    
    y = np.max(ploty)
    A = left_fit[0]
    B = left_fit[1]
    left_curverad = np.sqrt((1+((2*A*y_eval)+B)**2)**3)/np.abs(2*A)  ## Implement the calculation of the left line here
    A = right_fit[0]
    B = right_fit[1]
    right_curverad = np.sqrt((1+((2*A*y_eval)+B)**2)**3)/np.abs(2*A)  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    #ym_per_pix = 30/720 # meters per pixel in y dimension
    #xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    #ploty, left_fit_cr, right_fit_cr = generate_data(ym_per_pix, xm_per_pix)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    A = left_fit_cr[0]
    B = left_fit_cr[1]
    left_curverad = np.sqrt((1+((2*A*y_eval)+B)**2)**3)/np.abs(2*A)  ## Implement the calculation of the left line here
    A = right_fit_cr[0]
    B = right_fit_cr[1]
    right_curverad = np.sqrt((1+((2*A*y_eval)+B)**2)**3)/np.abs(2*A)  ## Implement the calculation of the right line here
    
    return left_curverad, right_curverad

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    #binary_output = np.copy(img) # placeholder line
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1    
    return binary_output

def calc_center(window):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(window[window.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx = np.argmax(histogram[:midpoint])
    rightx = np.argmax(histogram[midpoint:]) + midpoint
    return leftx, rightx

def get_line(img_shape, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    # Generate x and y values for plotting
    left_fitx, right_fitx, ploty = get_line(img_shape, left_fit, right_fit)
    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped,left_fit,right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
#    left_lane_inds = None
#    right_lane_inds = None
    
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]
    right_fitx = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]
    
    left_lane_inds = ((nonzerox >= left_fitx-margin) & (nonzerox <= left_fitx+margin))
    right_lane_inds = ((nonzerox >= right_fitx-margin) & (nonzerox <= right_fitx+margin))
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
#    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    return leftx, lefty, rightx, righty

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
#    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current-margin  # Update this
        win_xleft_high = leftx_current+margin  # Update this
        win_xright_low = rightx_current-margin  # Update this
        win_xright_high = rightx_current+margin  # Update this
        
        # Draw the windows on the visualization image
#        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
#        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
       # pass # Remove this when you add your function
        if(len(good_left_inds)>minpix):
           leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if(len(good_right_inds)>minpix):
           rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
       

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def get_lane_coeifficient(leftx,lefty,rightx,righty):
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    return left_fit, right_fit

def fit_lane_polynomial(binary_warped, left_fit=None, right_fit=None):
    # Find our lane pixels first

    
    if left_fit is None and right_fit is None:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    else:
        leftx, lefty, rightx, righty = search_around_poly(binary_warped,left_fit,right_fit)
        

    
    #print(left_fit)
    #print(right_fit)

#    # Generate x and y values for plotting
#    # Fit new polynomials
#    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
#    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
#    try:
#        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
#        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
#    except TypeError:
#        # Avoids an error if `left` and `right_fit` are still none or incorrect
#        print('The function failed to fit a line!')
#        left_fitx = 1*ploty**2 + 1*ploty
#        right_fitx = 1*ploty**2 + 1*ploty

    
    ## Visualization ##
    # Colors in the left and right lane regions
#    out_img[lefty, leftx] = [255, 0, 0]
#    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return leftx, lefty, rightx, righty

def find_lane(img):
    out_img = np.dstack((img, img, img))
    
    left_fit = None
    right_fit = None
    if len(l_lines_collection) > 1:
        left_fit = np.array(np.sum(l_lines_collection, 0)/len(l_lines_collection))
    if len(r_lines_collection) > 1:
        right_fit = np.array(np.sum(r_lines_collection, 0)/len(r_lines_collection))
    
    leftx, lefty, rightx, righty = fit_lane_polynomial(img,left_fit, right_fit)

    left_fit,right_fit = get_lane_coeifficient(leftx,lefty,rightx,righty)

    l_lines_collection.append(left_fit)
    r_lines_collection.append(right_fit)

    l_avg_lines = np.array(np.sum(l_lines_collection, 0)/len(l_lines_collection))
    r_avg_lines = np.array(np.sum(r_lines_collection, 0)/len(r_lines_collection))
    
    # Generate x and y values for plotting
    # Fit new polynomials
    #left_fitx, right_fitx, ploty = fit_poly(img.shape, leftx, lefty, rightx, righty)
    left_fitx, right_fitx, ploty = get_line(img.shape, l_avg_lines, r_avg_lines)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    #left_fit_rad, right_fit_rad = get_lane_coeifficient(lefty*ym_per_pix,leftx*xm_per_pix,righty*ym_per_pix,rightx*xm_per_pix)
    left_fit_rad, right_fit_rad = get_lane_coeifficient(left_fitx*xm_per_pix,ploty*ym_per_pix,right_fitx*xm_per_pix,ploty*ym_per_pix)



    l_points = np.squeeze(np.array(np.dstack((left_fitx, ploty)), dtype='int32'))
    r_points = np.squeeze(np.array(np.dstack((right_fitx, ploty)), dtype='int32'))
    points_rect = np.concatenate((r_points, l_points[::-1]), 0)
    cv2.fillPoly(out_img, [points_rect], (0, 255, 0))
    cv2.polylines(out_img, [l_points], False, (255, 0, 0), 15)
    cv2.polylines(out_img, [r_points], False, (0, 0, 255), 15)
    
    left_r, right_r = measure_curvature_pixels(ploty, left_fit, right_fit)
    left_rad, right_rad = measure_curvature_real(ploty*ym_per_pix, left_fit_rad, right_fit_rad)
    dist_from_center = left_fitx[-1] + ((right_fitx[-1] - left_fitx[-1])/2)
    dist_from_center = (imgW/2 - dist_from_center)*xm_per_pix
    return out_img, left_rad, right_rad, dist_from_center

def pipeline(img,mtx,dist):
    #Pipeline
    #1. Load Camera Calibration
    #2. Distortation correction
    #3. Color & Gradient threshold
    #4. Prespective Transform
    #5. Find Lane
    #6. Measure curvature
    #7. Draw Lane markers & text on original image.
    img = np.copy(img)
    #imgpoints,objpoints = gen_objpoints('./camera_cal/calibration*.jpg')
    #mtx, dist = getCamCal()
    #Camera Calibration
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    
    #Distortation Correction
    #undist = cv2.undistort(img, mtx, dist, None, mtx)
    undist = undistort(img, mtx, dist)
#    cv2.imwrite('./output_images/undist.jpg', undist)
    
    #Apply threshold
    thd = combined_thd(undist)
#    cv2.imwrite('./output_images/thresholded.jpg', thd)
    
    #Prespective Transform
    src = np.float32([[595,451], [680,451], [233,720],[1067,720]])
    dst = np.float32([[350,0],   [930,0],  [350,720],[930,720]])
    warped = prespectiveTransform(thd, src, dst)
#    cv2.imwrite('./output_images/warped.jpg', warped)
    #result = corners_unwarp(thd, nx, ny, mtx, dist)
    
    #Find Lane
    lane, left_r, right_r, dist_from_center = find_lane(warped)
#    cv2.imwrite('./output_images/lane.jpg', lane)
    
    #Calculate curvature
    radius = ((left_r+right_r)/2)
    radius_col.append(radius)
    avg_radius = np.sum(radius_col,0)/len(radius_col)
    
    #draw on original image
    unwraped = prespectiveTransform(lane, dst, src)
    result = cv2.addWeighted(img, 1, unwraped, .3, 0.0, dtype=0)
    dist_str = 'right'
    if dist_from_center >= 0:
        dist_str = 'left'
        
    cv2.putText(result, "Radius of Curvature = %.2f (m)" % avg_radius, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    cv2.putText(result, "Vehicle is %.2fm %s of center" % (np.abs(dist_from_center),dist_str), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    return result


def processTestImage(image,mtx,dist):
    # Read in an image
    img = cv2.imread(image)
    path, fname = os.path.split(image)
    print(fname)

    result = pipeline(img, mtx, dist)
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=40)
    
    ax2.imshow(result)
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    cv2.imwrite('./output_images/'+fname, result)

def processAll(mtx,dist):
    images = glob.glob('./test_images/*.jpg')
    for image in images:
        reset_lines_collection()
        processTestImage(image,mtx,dist)        

def process_image(image):
    img = pipeline(image, mtx, dist)
    return img


def reset_lines_collection():
    l_lines_collection.clear()
    r_lines_collection.clear()

def processVideo(video_path):
    file = os.path.splitext(video_path)
    output_video = file[0] + '_sol' + file[1]
    
    reset_lines_collection()

#    clip1 = VideoFileClip(video_path).subclip(0,5)
    clip1 = VideoFileClip(video_path)
    
    processed_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!\n",
    processed_clip.write_videofile(output_video, audio=False)    
    

#calibrate_camera('./camera_cal/calibration*.jpg', './test_images/test1.jpg')

mtx, dist = getCamCal()

#processTestImage('./test_images/test1.jpg', mtx,dist)
#processAll(mtx,dist)

processVideo('project_video.mp4')
#processVideo('challenge_video.mp4')
#processVideo('harder_challenge_video.mp4')