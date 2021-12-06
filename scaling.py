import cv2
import numpy as np
from numpy.linalg import norm


# Triangulation
def triangulatePoints(R, t, prev_R, prev_t, prev_kp, cur_kp):
	##Triangulates the feature correspondence points with
	##the camera intrinsic matrix, rotation matrix, and translation vector.
	##It creates projection matrices for the triangulation process.

	# The canonical matrix (set as the previous)
        P1 = np.hstack((prev_R, prev_t))
        P1 = K.dot(P1)

        # Rotated and translated using P0 as the reference point
        P2 = np.hstack((R, t))
        P2 = K.dot(P2)

        # Reshaped the point correspondence arrays to cv2.triangulatePoints's format
        point1 = prev_kp.reshape(2, -1)
        point2 = cur_kp.reshape(2, -1)

        pts4D = cv2.triangulatePoints(P1, P2, point1, point2).reshape(-1, 4)#[:, :3]

	pts3D = pts4D[:, :3]/np.repeat(pts4D[:, 3], 3).reshape(-1, 3)
	
	#print ('pts3D',pts3D)

	return pts3D


# Function that returns a scale value for translation vector from 3D points
def getRelativeScale(prev_pts3D, pts3D):
	##Returns the relative scale based on the 3-D point clouds
	##produced by the triangulation_3D function. Using a pair of 3-D corresponding points
	##the distance between them is calculated. This distance is then divided by the
	##corresponding points' distance in another point cloud.

	min_idx = min([pts3D.shape[0], prev_pts3D.shape[0]])
	ratios = []  # List to obtain all the ratios of the distances
	for i in xrange(min_idx):
		if i > 0:
			Xk = pts3D[i]
			p_Xk = pts3D[i-1]
			Xk_1 = prev_pts3D[i]
			p_Xk_1 = prev_pts3D[i - 1]

			if norm(p_Xk - Xk) != 0:
				ratios.append(norm(p_Xk_1 - Xk_1) / norm(p_Xk - Xk))

	#d_ratio = np.median(ratios) # Take the median of ratios list as the final ratio
	
	m = 2
	m2 = 0.1
	data = []
	data2 = []
	d = np.abs(ratios - np.median(ratios))
	mdev = np.median(d)
	s = d/(mdev if mdev else 1.0)
	mean = np.mean(ratios)
	std = np.std(ratios)

	for i in range(0,len(ratios)):
		if s[i] <m:
			data.append(ratios[i])
	
		if abs(ratios[i] - mean) < m2 * std :
			data2.append(ratios[i])

	d_ratio1 = np.mean(data)
	d_ratio2 = np.mean(data2)
	
	#print('ratios:',data)
	print '********************************************************'
	print ('d_ratio1:',d_ratio1)	
	#print ('d_ratio2:',d_ratio2)	

	d_ratio = (d_ratio1 + d_ratio2)/2
	#print ('d_ratio:',d_ratio)	

	return d_ratio1



# Function that returns a scale from the GPS data
def getAbsoluteScale(f, frame_id):
	x_pre, y_pre, z_pre = f[frame_id-1][3], f[frame_id-1][7], f[frame_id-1][11]
	x_cur, y_cur, z_cur = f[frame_id][3], f[frame_id][7], f[frame_id][11]
	scale = np.sqrt((x_cur-x_pre)**2 + (y_cur-y_pre)**2 + (z_cur-z_pre)**2)
	return x_cur, y_cur, z_cur, scale

