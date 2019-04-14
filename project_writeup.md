## Advanced Lane Finding Project Writeup

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image_ori]: ./test_images/test1.jpg "Original"
[image_camcal]: ./output_images/cam_cal.jpg "Camera Calibration Undistorted"
[image_undistor]: ./output_images/undist.jpg "Undistorted"
[image_binaryThd]: ./output_images/thresholded.png "Binary Thresholded"
[image_warped]: ./output_images/warped.jpg "Warped"
[image_lane]: ./output_images/lane.jpg "Fit Visual"
[image_final]: ./output_images/test1.jpg "Output"
[video_final]: ./project_video_sol.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
All the code for this project is contained in [advanced_lane_line.py](https://github.com/roopakingole/CarND-Advanced-Lane-Lines/blob/master/advanced_lane_line.py) in the root folder.
---

### Camera Calibration

The code for this step is contained in function `calibrate_camera()` and `gen_objpoints()`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result (code for this is contained in function `cal_undistort()` ): 
```python
calibrate_camera('./camera_cal/calibration*.jpg', './test_images/test1.jpg')
```
![alt text][image_camcal]

### Pipeline (single images)
I will describe my Pipeline to detect the lane lines consist in following 7 steps:
1. Load Camera Calibration
2. Distortion correction
3. Color & Gradient threshold
4. Perspective Transform
5. Find Lane
6. Measure curvature
7. Draw Lane markers & text on original image.

#### Step 1. Load Camera Calibration:
First we will load the camera calibration from the pickle file. I have created this pickle file in the previous step. The reason I choose to store the pickle file is performance. Calibrating camera take significant amount of time and by storing the calibration as pickle can save that time. The code for this can be found in `getCamCal()`
```python
mtx, dist = getCamCal()
```

#### Step 2: Distortion Correction
For distortion correction I used OpenCV `cv2.undistort()` function. The code for this can be found in `pipeline()` like below. The outcome of distort correction is shown below:
```python
undist = undistort(img, mtx, dist)
```
![alt text][image_undistor]

#### Step 3: Color & Gradient Threshold 
I have created the function `combined_thd(image)` to apply different thresholds to the image to create the thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in `combined_thd(image) function`).  Here's an example of my output for this step.  
```python
    thd = combined_thd(undist)
```

![alt text][image_binaryThd]

#### Step 4: Perspective Transform 
Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `prespectiveTransform()`.  The `prespectiveTransform(img, src, dst)` function takes as inputs an binary thresholded image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32([[595,451], [680,451], [233,720],[1067,720]])
    dst = np.float32([[350,0],   [930,0],  [350,720],[930,720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595,451       | 350,0         | 
| 680,451       | 930,0         |
| 233,720       | 350,720       |
| 1067,720      | 930,720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
```python
    src = np.float32([[595,451], [680,451], [233,720],[1067,720]])
    dst = np.float32([[350,0],   [930,0],  [350,720],[930,720]])
    warped = prespectiveTransform(thd, src, dst)
```
![alt text][image_warped]

#### Step5: Finding Lane
I have defined all lane finding code in one function `find_lane(img)`. This function takes warped image as input and identifies the lane line, draw the lane markers on the warped image and calculate radius of curvature of left & right lane. This function also calculates the distance from center.
To find the lane lines on warped image, I used sliding window method followed by search the next window around polynomial method. The code for this can be found in `fit_lane_polynomial()`. To find the curves of the lane I used 2nd order polynomial equation and tries to find the coefficients using `np.polyfit()`.

In order to smooth the transition from frame to frame in the video, I used averaging method. I have created collections of left and right coefficients for last 5 frames and take the average of those to draw on the image. The code for this can be found in `find_lane()` at line #662 - #668. 

At the end of this function `find_lane()`, I calculate the radius of curvature for each lane and distance from center.
```python
    lane, left_r, right_r, dist_from_center = find_lane(warped)
```
The processed image is as:
![alt text][image_lane]

#### Step 6: Calculate Radius of Curvature & Distance from center
Part of the radius calcualation was done as part of `find_lane()`. To measure the curvature I created two seperate function, one for pixel `measure_curvature_pixels()` and other in meters `measure_curvature_real()`. 

```python
    radius = ((left_r+right_r)/2)
    radius_col.append(radius)
    avg_radius = np.sum(radius_col,0)/len(radius_col)
```

#### Step 7: Draw Lane markers & text on original image
As part of last step, I unwarped the image to its original prespective by just reversing the **source** and **destination** coordinates to function `prespectiveTransform(lane, dst, src)`
Lastly, overlay the new unwarped image to original image and write the radius of curvature and distance to the image.

```python
    unwraped = prespectiveTransform(lane, dst, src)
    result = cv2.addWeighted(img, 1, unwraped, .3, 0.0, dtype=0)
    dist_str = 'right'
    if dist_from_center >= 0:
        dist_str = 'left'
        
    cv2.putText(result, "Radius of Curvature = %.2f (m)" % avg_radius, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    cv2.putText(result, "Vehicle is %.2fm %s of center" % (np.abs(dist_from_center),dist_str), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
```
![alt text][image_ori] ![alt text][image_final]

---

### Pipeline (video)


Here's a [link to my video result](./project_video_sol.mp4)

---

### Discussion

In my lane finding, I am facing the issue of in threshold where in extreme lighting conditions where fixed threshold doesn't work all the time. Also, my finding line in the image doesn't work robustly when there is no line marks at the bottom of the frame image.
If I were to pursue this project further, I would definitely improve the color threshold.