# FCV_Project2

## Algorithm used
The used algorithm is based on the non dense optical flow by Lucas-Kanade method.  
The algorithm selects up to 100 points to track and tracks those points. 
The algorithm considers a movement detected if any point moved with more than a threshold (configurable)

Because optical flow algorithms assume brightness consistency, only this algorithm will fail in case of sudden changes in brightness (turn on/off the light in the room)  
To overcome this downside, the algorithm also checks for these sudden changes in the brightness. This check is done based on the average difference between __Y__ component of the images, transformed to __YUV__.  

The configuration file looks like this:  
* feature_params:  
  * maxCorners: 100  
  * qualityLevel: 0.3  
  * minDistance: 7  
  * blockSize: 7  
* lk_params:  
  * winSize: !!python/tuple [15, 15]  
  * maxLevel: 2  
  * criteria: !!python/tuple [3, 10, 0.03]  
* diff_limit: 1.0  
* brightness_limit: 20.0  

The important parameters are __diff_limit__ and __brightness_limit__, which should be modified accordingly for each use-case separately.  
__diff_limit__ will be used to detect movement. If all the movements are lower than this value, than no movement is considered.  
For any brightness level difference higher than __brightness_limit__, the movement detection will be inhibited, considering that only a sudden change in brightness occurred. 
  
 