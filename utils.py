import cv2
import numpy as np
import math
from math import sqrt




def getImages(i):
    return cv2.imread('/home/ali/Desktop/Data/kitti/00/image_0/{0:06d}.png'.format(i), 0)



def getTruePose():
    file = '/home/ali/Desktop/Data/kitti/poses/00.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)


# Function that takes a vector and returns its normilized form
def Norm_Vector(t):
	return t/sqrt([0]*t[0] + t[1]*t[1] + t[2]*t[2])



def PointPositionInNormalPlane(pts3D, R, t):
	pts1 = np.zeros(2,1)
	pts2 = np.zeros(2,1)

	pts1[0] = pts3D[0]/pts3D[2]
	pts1[1] = pts3D[1]/pts3D[2]
	depth1 = pts3D[2]

	pts3D_tmp = t + R.dot(pts3D)

	pts2[0] = pts3D_tmp[0]/pts3D_tmp[2]
	pts2[1] = pts3D_tmp[1]/pts3D_tmp[2]
	depth2 = pts3D_tmp[2]

	return pts1,depth1,pts2,depth2



# Compute the error for 2 points pts1 and pts2
def ComputeEpipolarError(pts1, pts2, R, t, K):
	Y1 = np.zeros(3,1)
	Y2 = np.zeros(3,1)

	Y1[0] = (pts1[0]-K[0][2])/K[0][0]
	Y1[1] = (pts1[1]-K[1][2])/K[1][1]
	Y1[2] = 1
	
	Y2[0] = (pts2[0]-K[0][2])/K[0][0]
	Y2[1] = (pts2[1]-K[1][2])/K[1][1]
	Y2[2] = 1

	skew_t = np.zeros(3,3)
	skew_t[0][1] = t[0]
	skew_t[1][0] = -t[0]
	skew_t[0][2] = t[1]
	skew_t[2][0] = -t[1]
	skew_t[1][2] = t[2]
	skew_t[2][1] = -t[2]

	d = Y2.T.dot(skew_t.dot(R.dot(Y1)))

	return d



# Compute the error for a vector of points pts1 and pts2
def ComputeEpipolarErrorTotal(pts1, pts2, R, t, K):
	error = 0
	for i in range(0,pts.shape):
		e = ComputeEpipolarError(pts1[i], pts2[i], R, t, K)
		error += e*e

	return sqrt(error/pts.shape)




def isclose(x, y, rtol=1.e-5, atol=1.e-8):
	return abs(x-y) <= atol + rtol * abs(y)

def euler_angles_from_rotation_matrix(R):

	# From a paper by Gregory G. Slabaugh (undated),
	# Computing Euler angles from a rotation matrix
	phi = 0.0
	if isclose(R[2,0],-1.0):
		theta = math.pi/2.0
		psi = math.atan2(R[0,1],R[0,2])

	elif isclose(R[2,0],1.0):
		theta = -math.pi/2.0
		psi = math.atan2(-R[0,1],-R[0,2])
    
	else:
		theta = -math.asin(R[2,0])
		cos_theta = math.cos(theta)
		psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
		phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)

	psi = (psi/math.pi)*180
	theta = (theta/math.pi)*180
	phi = (phi/math.pi)*180

	print('psi', psi)
	print('theta', theta)
	print('phi', phi)
	
	return psi, theta, phi
