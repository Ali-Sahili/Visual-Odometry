import cv2
import numpy as np
from numpy.linalg import inv,norm




# Function that computes a score for the essential matrix E
def CheckEssentialScore(E, K, pts1, pts2, inliers_index, sigma, q):
	F = ((inv(K)).T.dot(E)).dot(inv(K))# Fundamental_matrix: F = (inv(K)T)*E*inv(K)

	th = 3.841 #tuning value from paper 
	thscore = 5.991 #tuning value from paper 

	score = 0
	for i in range(0,len(pts1)):
		pts1_out = np.zeros([3,1])
		pts2_out = np.zeros([3,1])

		pts1_tmp = np.zeros([3,1])
		pts2_tmp = np.zeros([3,1])

		pts1_tmp[0] = pts1[i][0]
		pts1_tmp[1] = pts1[i][1]
		pts1_tmp[2] = 1

		pts2_tmp[0] = pts2[i][0]
		pts2_tmp[1] = pts2[i][1]
		pts2_tmp[2] = 1

		pts1_out = F.dot(pts1_tmp)
		pts2_out = F.dot(pts2_tmp)

		num1 = pts1_out.T.dot(pts1_tmp)
		num2 = pts2_out.T.dot(pts2_tmp)

		squareDist1 = (num1*num1)/(pts1_out[0]*pts1_out[0] + pts1_out[1]*pts1_out[1])
		squareDist2 = (num2*num2)/(pts2_out[0]*pts2_out[0] + pts2_out[1]*pts2_out[1])

		chisquare1 = squareDist1/(sigma*sigma)
		chisquare2 = squareDist2/(sigma*sigma)

		if (chisquare1>th):
			score +=0
		else:
			score += -chisquare1 + thscore

		if (chisquare2>th):
			score +=0
		else:
			score += -chisquare2 + thscore

	q.put(score) ## used for returning value after threading
	return score



# Function that computes a score for the homography matrix H
def CheckHomographyScore(H, pts1, pts2, inliers_index, sigma, q):
	th = 5.991 #tuning value from paper

	score = 0
	for i in range(0,len(pts1)):
		pts2in1 = np.zeros([3,1])
		pts1in2 = np.zeros([3,1])

		pts1_tmp = np.zeros([3,1])
		pts2_tmp = np.zeros([3,1])

		pts1_tmp[0] = pts1[i][0]
		pts1_tmp[1] = pts1[i][1]
		pts1_tmp[2] = 1

		pts2_tmp[0] = pts2[i][0]
		pts2_tmp[1] = pts2[i][1]
		pts2_tmp[2] = 1

		pts2in1 = inv(H).dot(pts2_tmp)
	
		u2in1 = pts2in1[0]/pts2in1[2]
		v2in1 = pts2in1[1]/pts2in1[2]

		squareDist1 = (pts1_tmp[0] - u2in1)*(pts1_tmp[0] - u2in1) + \
						(pts1_tmp[1] - v2in1)*(pts1_tmp[1] - v2in1)
		chisquare1 = squareDist1/(sigma*sigma)

		if (chisquare1>th):
			score +=0
		else:
			score += th - chisquare1 


		pts1in2 = H.dot(pts1_tmp)
	
		u1in2 = pts1in2[0]/pts1in2[2]
		v1in2 = pts1in2[1]/pts1in2[2]

		squareDist2 = (pts2_tmp[0] - u1in2)*(pts2_tmp[0] - u1in2) + \
						(pts2_tmp[1] - v1in2)*(pts2_tmp[1] - v1in2)
		chisquare2 = squareDist2/(sigma*sigma)

		if (chisquare2>th):
			score +=0
		else:
			score += th - chisquare2

	q.put(score) ## used for returning value after threading
	return score
 


# Function that takes the previous and the current keypoints and 
# computes the essential matrix and its inliers
def FindEssentialMatrix(cur_kp, prev_kp, K, q):
	E, E_mask = cv2.findEssentialMat(cur_kp, prev_kp, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

	E = E/float(E[2][2])

	inliers_E = []

	for ii in range(0,len(E_mask)):
			if (E_mask[ii] == 1):
				inliers_E.append(ii)

	# We select only inlier points
	mask = E_mask.ravel()
	Inlier_prev_kp_E = prev_kp[mask == 1]
	Inlier_cur_kp_E = cur_kp[mask == 1]


	q.put((E, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E))
	return E, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E



# Function that takes the previous and the current keypoints and 
# computes the homography matrix and its inliers
def FindHomographyMatrix(cur_kp, prev_kp, q):
	H, H_mask = cv2.findHomography(prev_kp, cur_kp, cv2.RANSAC, 3.0)

	H = H/float(H[2][2])

	inliers_H = []

	for ii in range(0,len(H_mask)):
		if (H_mask[ii] == 1):
			inliers_H.append(ii)

	mask = H_mask.ravel()
	Inlier_prev_kp_H = prev_kp[mask == 1]
	Inlier_cur_kp_H = cur_kp[mask == 1]


	q.put((H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H))
	return H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H



# Function that tries to choose a matrix (E or H) and 
# estimate the motion of the robot(transaltion vector + rotation matrix)
def EstimateMotion(E, H, K, cur_kp, prev_kp, Score_EH, Score_EH_threshold, list_R, list_t, list_Rs, list_Ts, list_Ns):

	if(Score_EH > Score_EH_threshold):
		#choosing E
		print ('E is chosen.')
		# Estimate Rotation and translation vectors
		_, R, t, mask = cv2.recoverPose(E, cur_kp, prev_kp, K)

		t = Norm_Vector(t)

		list_R.append(R)
		list_t.append(t)

	else:
		#choosing H
		print ('H is chosen.')
		#num possible solutions will be returned.
		#Rs contains a list of the rotation matrix.
		#Ts contains a list of the translation vector.
		#Ns contains a list of the normal vector of the plane.
		num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(H, K)
	
		for iter_Ts in range(0,num):
			Ts[iter_Ts] = Norm_Vector(Ts[iter_Ts])

		list_Rs.append(Rs)
		list_Ts.append(Ts)
		list_Ns.append(Ns)

		R = Rs[0]
		t = Ts[0]

	return R, t ,list_R, list_t, list_Rs, list_Ts, list_Ns
