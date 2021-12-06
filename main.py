import cv2
import numpy as np
from math import sqrt
from Queue import Queue
from threading import Thread

from utils import *
from scaling import *
from matrix_estimation import *
from features_extractor import detectFeatures




"""
  - Change the directories of the Kitti dataset in the code below 
    ( in the functions: getImages and getTruePose ) from utils.py

  - In this code, I try several methods of features extraction and description which 
    differ with accuracy, time, and transformation invariant,
    and it contains only the tracking part using optical flow 
    without matching methods (tracking is faster than matching).

  - for more details about matching, extraction, and the entire algorithm, read the report.
"""



# CONSTANTS

lk_params = dict(winSize=(21, 21),  # Parameters used for cv2.calcOpticalFlowPyrLK (KLT tracker)
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


detectors = ['FAST_BRIEF']

#Intrinsic parameters
#camera_matrix 
K = np.array([[718.8560,   0.0   , 607.1928],
              [  0.0   , 718.8560, 185.2157],
              [  0.0   ,   0.0   ,   1.0   ]])


trajectory = np.zeros((600, 600, 3), dtype=np.uint8)
ground_truth = getTruePose()

#Initial values
initial_t = np.zeros((3, 1))
initial_R = np.eye(3)

cur_t = initial_t
cur_R = initial_R

T_vectors = []
R_matrices = []

x_vectors_real = []
z_vectors_real = []

x_vectors_real.append(0)
z_vectors_real.append(0)

T_vectors.append(tuple(cur_t))
R_matrices.append(tuple(cur_R))

list_R = []
list_t = []

list_Rs = []
list_Ts = []
list_Ns = []

scale_method = 'GPS'#'FromPts3D'#

#nb_frames_MAX = 4300
nb_frames_MAX = 2100


Scale = 1.0

Score_EH_threshold = 0.01

Error_temp = []
Time_temp = []


# Apply the algorithm for the different features detectors
for detector in detectors:

	print '##############################################################'
	print detector

	e1 = cv2.getTickCount()

	trajectory = np.zeros((600, 600, 3), dtype=np.uint8)
	

	prev_img = getImages(0)

	prev_kp = detectFeatures(prev_img, detector)

	cur_img = getImages(1)

	# tracking
	cur_kp, st, err = cv2.calcOpticalFlowPyrLK(prev_img, cur_img, prev_kp, None, **lk_params)

	
	## without threading
        # Estimate the essential matrix
	#E, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E = FindEssentialMatrix(cur_kp, prev_kp, K)
	
	#Score_E = CheckEssentialScore(E, K, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E, 1.0, q)

	# Estimate the Homography matrix
	#H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H = FindHomographyMatrix(cur_kp, prev_kp)
	
	#Score_H = CheckHomographyScore(H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H, 1.0)


	## with threading
	q3 = Queue()
	thread3 = Thread(target=FindEssentialMatrix, args=(cur_kp, prev_kp, K, q3))

	q4 = Queue()
	thread4 = Thread(target=FindHomographyMatrix, args=(cur_kp, prev_kp, q4))

	thread3.start()
	thread4.start()

        thread3.join()
	thread4.join()

	E, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E = q3.get()
	H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H = q4.get()




	q1 = Queue()
	thread1 = Thread(target=CheckEssentialScore, args=(E, K, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E, 1.0, q1))

	q2 = Queue()
	thread2 = Thread(target=CheckHomographyScore, args=(H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H, 1.0, q2))

	thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

	Score_E = q1.get()
	Score_H = q2.get()


	#thread3.join()
	#thread4.join()
	#thread1.join()
	#thread2.join()


	Score_EH  = Score_H/(Score_H + Score_E)

	print ('Score_E:',Score_E)
	print ('Score_H:',Score_H)
	print ('Score_EH:',Score_EH)

	R, t ,list_R, list_t, list_Rs, list_Ts, list_Ns = EstimateMotion(E, H, K, cur_kp, prev_kp, Score_EH, Score_EH_threshold, list_R, list_t, list_Rs, list_Ts, list_Ns)

	
	psi, theta, phi = euler_angles_from_rotation_matrix(R)

	# triangulation
	pts3D = triangulatePoints(R, t, initial_R, initial_t, Inlier_prev_kp_E, Inlier_cur_kp_E)

	Ys = pts3D[:, 0]
	Zs = pts3D[:, 1]
	Xs = pts3D[:, 2]

	prev_pts3D = pts3D



# Scaling the trajectory
	if scale_method == 'GPS':
		truth_x, truth_y, truth_z, Scale = getAbsoluteScale(ground_truth, 0)

	elif scale_method == 'FromPts3D':
		Scale = getRelativeScale(prev_pts3D, pts3D)
	
	else:
		#default scale 1.0
		Scale = 1.0
	
	
	
	
	cur_t += cur_R.dot(t) * Scale  # Concatenate the translation vectors -- t = t + s* R*t
	cur_R = R.dot(cur_R)  # Concatenate the rotation matrix -- R = R1*R2
	

	# only for the first time
	cur_t = t
	cur_R = R
	
	T_vectors.append(tuple(cur_t))
	R_matrices.append(tuple(cur_R))
	
	truth_x, truth_y, truth_z, Scale = getAbsoluteScale(ground_truth, 0)

	x_vectors_real.append(truth_x)    
	z_vectors_real.append(truth_z) 

       
	# The new frame becomes the previous frame
	prev_kp = cur_kp
	
	prev_R = R
	prev_t = t
	
	prev_img = cur_img
	
	prev_pts3D = pts3D
	

	# in case of a big change in the rotation matrix
	condition = 0

	for nb_frames in range(2,nb_frames_MAX):

		#print detector
		print ('nb_frames:',nb_frames)
	
		prev_kp = detectFeatures(prev_img, detector)
	
		cur_img = getImages(nb_frames)
	
		cur_kp, st, err = cv2.calcOpticalFlowPyrLK(prev_img, cur_img, prev_kp, None, **lk_params)


		## without threading
		# Estimate the essential matrix
		#E, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E = FindEssentialMatrix(cur_kp, prev_kp, K)
	
		#Score_E = CheckEssentialScore(E, K, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E, 1.0, q)

		# Estimate the Homography matrix
		#H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H = FindHomographyMatrix(cur_kp, prev_kp)
		
		#Score_H = CheckHomographyScore(H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H, 1.0)
	
	
		## with threading
		q3 = Queue()
		thread3 = Thread(target=FindEssentialMatrix, args=(cur_kp, prev_kp, K, q3))

		q4 = Queue()
		thread4 = Thread(target=FindHomographyMatrix, args=(cur_kp, prev_kp, q4))

		thread3.start()
		thread4.start()

        	#thread3.join()
		#thread4.join()

		E, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E = q3.get()
		H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H = q4.get()


		q1 = Queue()
		thread1 = Thread(target=CheckEssentialScore, args=(E, K, Inlier_prev_kp_E, Inlier_cur_kp_E, inliers_E, 1.0, q1))

		q2 = Queue()
		thread2 = Thread(target=CheckHomographyScore, args=(H, Inlier_prev_kp_H, Inlier_cur_kp_H, inliers_H, 1.0, q2))

		thread1.start()
        	thread2.start()

        	#thread1.join()
        	#thread2.join()

		Score_E = q1.get()
		Score_H = q2.get()

		thread3.join()
		thread4.join()
		thread1.join()
		thread2.join()

		
		Score_EH  = Score_H/(Score_H + Score_E)

		print ('Score_E:',Score_E)
		print ('Score_H:',Score_H)
		print ('Score_EH:',Score_EH)
		
		R, t ,list_R, list_t, list_Rs, list_Ts, list_Ns = EstimateMotion(E, H, K, cur_kp, prev_kp, Score_EH, Score_EH_threshold, list_R, list_t, list_Rs, list_Ts, list_Ns)
	

		
		psi, theta, phi = euler_angles_from_rotation_matrix(R)
		

		#triangulation	
		pts3D = triangulatePoints(R, t, prev_R, prev_t, Inlier_prev_kp_E, Inlier_cur_kp_E)

		Ys = pts3D[:, 0]
		Zs = pts3D[:, 1]
		Xs = pts3D[:, 2]

		prev_pts3D = pts3D

		# Scaling the trajectory
		if scale_method == 'GPS':
			truth_x, truth_y, truth_z, Scale = getAbsoluteScale(ground_truth, nb_frames)
	
		elif scale_method == 'FromPts3D':
			Scale = getRelativeScale(prev_pts3D, pts3D)
	
		else:
			#default scale 1.0
			Scale = 1.0
	
		print ('Scale:',Scale)
	
		#if Scale >1.0:
		#	Scale = 1.0
	
	
		# in case of a big change in the rotation matrix
		condition = sqrt(np.sum(np.square(R-prev_R)))
	
		print('condition:',condition)
		
		# in case of a big change in the rotation matrix
		if condition<1.5:
	
			
			if Scale>0.1:
				cur_t = cur_t + Scale * cur_R.dot(t)  # Concatenate the translation vectors
				cur_R = R.dot(cur_R)  # Concatenate the rotation matrix
		
		
			T_vectors.append(tuple(cur_t))
			R_matrices.append(tuple(cur_R))
	        	
			x_vectors_real.append(truth_x)    
			z_vectors_real.append(truth_z)


			# The new frame becomes the previous frame
	        	prev_kp = cur_kp
		
			prev_R = R
			prev_t = t
		
			prev_img = cur_img
		
			prev_pts3D = pts3D
	

			e2 = cv2.getTickCount()
			time = (e2 - e1)/ cv2.getTickFrequency()

			Time_temp.append(time)
			print ('Time: ',time)
	#################################################################################################
	
			truth_x, truth_y, truth_z, _ = getAbsoluteScale(ground_truth, nb_frames)
	
			Error = sqrt((cur_t[0]-float(truth_x))**2 + (cur_t[1]-float(truth_y))**2 + 	(cur_t[2]-float(truth_z))**2)/sqrt(float(truth_x)*float(truth_x) + float(truth_y)*float(truth_y) 	+ float(truth_z)*float(truth_z))
			Error *= 100
			print ('Error: ',Error,'%')
			print '*********************************'

			Error_temp.append(Error)
		
			####Visualization of the result
			draw_x, draw_y, draw_z = cur_t[0]+300, cur_t[1]+100, cur_t[2]+100
			draw_tx, draw_ty, draw_tz = truth_x+300, truth_y+100, truth_z+100
		
			draw_tx = float(draw_tx)
			draw_tz = float(draw_tz)
		
			cv2.circle(trajectory, (draw_x, draw_z) ,1, (0,0,255), 2);
			cv2.circle(trajectory, (int(draw_tx), int(draw_tz)) ,1, (255,0,0), 2);
		
			cv2.imshow('trajectory',trajectory)
	
			

		
		k = cv2.waitKey(1) & 0xFF
	    	if k == 27:
	        	break
	
	print '##################################################################'
	
	

	#cv2.imwrite("/home/ali/Desktop/Results_new/Trajectory_"+str(detector)+"_"+str(scale_method)+".jpg",trajectory)
	cv2.imwrite("/home/ali/Desktop/Trajectory_"+str(detector)+"_"+str(scale_method)+".jpg",trajectory)


	Error_average = sum(Error_temp)/len(Error_temp)
	Time_average = sum(Time_temp)/len(Time_temp)

	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		cv2.destroyAllWindows()
	
print 'Finishing...'
print ('Error_average:',Error_average,'%')
print ('Time_average for one iteration:',Time_average)
print '##################################################################'
	



