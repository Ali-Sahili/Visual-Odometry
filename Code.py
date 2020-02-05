import cv2
import numpy as np
import math
from numpy.linalg import inv,norm
from math import sqrt
from Queue import Queue
from threading import Thread


# Change the directories of the Kitti dataset in the code below ( in the functions: getImages and getTruePose )

# in this code, I try several methods of features extraction and description which differ from accuracy, time, 
# and transformation invariant,
# and it contains only the tracking part using optical flow without matching methods (tracking is faster than matching).
# for more details about matching, extraction, and the entire algorithm, read the report.


####################################################################################################
## Extract features and descriptors
def detectFeatures(img, detector):
	##Detects new features in the frame.
        ##Uses the Feature Detector selected.

	if detector == 'SIFT':
		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts, des = sift.detectAndCompute(img,None)

	elif detector == 'SURF':
		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts,des = surf.detectAndCompute(img,None)


	elif detector == 'ORB':
		orb = cv2.ORB_create(1000)
		feature_pts, des = orb.detectAndCompute(img,None)

	elif detector == 'BRISK':
		brisk = cv2.BRISK_create()
		feature_pts, des = brisk.detectAndCompute(img, None)


	elif detector == 'KAZE':
		kaze = cv2.KAZE_create()
		feature_pts, des = kaze.detectAndCompute(img, None)

	
	elif detector == 'AKAZE':
		akaze = cv2.AKAZE_create()
		feature_pts, des = akaze.detectAndCompute(img, None)

########################### SimpleBlobDetector as detector#############################################
	elif detector == 'SimpleBlobDetector_BRIEF':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)


	elif detector == 'SimpleBlobDetector_VGG':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'SimpleBlobDetector_BoostDesc':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'SimpleBlobDetector_LATCH':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'SimpleBlobDetector_DAISY':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'SimpleBlobDetector_FREAK':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)

	elif detector == 'SimpleBlobDetector_LUCID':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

################################### FAST as detector##############################################
	elif detector == 'FAST_BRIEF':
		fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
		feature_pts = fast.detect(img, None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)

	elif detector == 'FAST_VGG':
		fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
		feature_pts = fast.detect(img, None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'FAST_BoostDesc':
		fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
		feature_pts = fast.detect(img, None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'FAST_LATCH':
		fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
		feature_pts = fast.detect(img, None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'FAST_DAISY':
		fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
		feature_pts = fast.detect(img, None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'FAST_FREAK':
		fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
		feature_pts = fast.detect(img, None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)


	elif detector == 'FAST_LUCID':
		fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
		feature_pts = fast.detect(img, None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

#################################### GFTT as detector#############################################

	elif detector == 'GFTT_BRIEF':
		retval = cv2.GFTTDetector_create()
		feature_pts = retval.detect(img,None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)


	elif detector == 'GFTT_VGG':
		retval = cv2.GFTTDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'GFTT_BoostDesc':
		retval = cv2.GFTTDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'GFTT_LATCH':
		retval = cv2.GFTTDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'GFTT_DAISY':
		retval = cv2.GFTTDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'GFTT_FREAK':
		retval = cv2.GFTTDetector_create()
		feature_pts = retval.detect(img,None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)


	elif detector == 'GFTT_LUCID':
		retval = cv2.GFTTDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

###################################  AGAST as detector ################################################

	elif detector == 'AGAST_BRIEF':
		retval = cv2.AgastFeatureDetector_create()
		feature_pts = retval.detect(img,None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)


	elif detector == 'AGAST_VGG':
		retval = cv2.AgastFeatureDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'AGAST_BoostDesc':
		retval = cv2.AgastFeatureDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'AGAST_LATCH':
		retval = cv2.AgastFeatureDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'AGAST_DAISY':
		retval = cv2.AgastFeatureDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'AGAST_FREAK':
		retval = cv2.AgastFeatureDetector_create()
		feature_pts = retval.detect(img,None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)


	elif detector == 'AGAST_LUCID':
		retval = cv2.AgastFeatureDetector_create()
		feature_pts = retval.detect(img,None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

#################################  STAR  as detector ##################################################

	elif detector == 'STAR_BRIEF':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)


	elif detector == 'STAR_VGG':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'STAR_BoostDesc':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'STAR_LATCH':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'STAR_DAISY':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'STAR_FREAK':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)


	elif detector == 'STAR_LUCID':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

################################  MSER  as detector ################################################

	elif detector == 'MSER_BRIEF':
		mser = cv2.MSER_create()
		feature_pts = mser.detect(img, None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)


	elif detector == 'MSER_VGG':
		mser = cv2.MSER_create()
		feature_pts = mser.detect(img, None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'MSER_BoostDesc':
		mser = cv2.MSER_create()
		feature_pts = mser.detect(img, None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'MSER_LATCH':
		mser = cv2.MSER_create()
		feature_pts = mser.detect(img, None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'MSER_DAISY':
		mser = cv2.MSER_create()
		feature_pts = mser.detect(img, None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'MSER_FREAK':
		mser = cv2.MSER_create()
		feature_pts = mser.detect(img, None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)


	elif detector == 'MSER_LUCID':
		mser = cv2.MSER_create()
		feature_pts = mser.detect(img, None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

#####################################  SIFT  as detector #################################################

	elif detector == 'SIFT_BRIEF':
		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts = sift.detect(img,None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)


	elif detector == 'SIFT_VGG':
		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts = sift.detect(img,None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'SIFT_BoostDesc':
		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts = sift.detect(img,None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'SIFT_LATCH':
		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts = sift.detect(img,None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'SIFT_DAISY':
		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts = sift.detect(img,None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'SIFT_FREAK':
		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts = sift.detect(img,None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)


	elif detector == 'SIFT_LUCID':
		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts = sift.detect(img,None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

###################################  SURF  as detector ###################################################

	elif detector == 'SURF_BRIEF':
		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts = surf.detect(img,None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)


	elif detector == 'SURF_VGG':
		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts = surf.detect(img,None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'SURF_BoostDesc':
		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts = surf.detect(img,None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'SURF_LATCH':
		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts = surf.detect(img,None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'SURF_DAISY':
		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts = surf.detect(img,None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'SURF_FREAK':
		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts = surf.detect(img,None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)


	elif detector == 'SURF_LUCID':
		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts = surf.detect(img,None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

#####################################  ORB  as detector ###############################################

	elif detector == 'ORB_BRIEF':
		orb = cv2.ORB_create(1000)
		feature_pts = orb.detect(img,None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)


	elif detector == 'ORB_VGG':
		orb = cv2.ORB_create(1000)
		feature_pts = orb.detect(img,None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'ORB_BoostDesc':
		orb = cv2.ORB_create(1000)
		feature_pts = orb.detect(img,None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'ORB_LATCH':
		orb = cv2.ORB_create(1000)
		feature_pts = orb.detect(img,None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'ORB_DAISY':
		orb = cv2.ORB_create(1000)
		feature_pts = orb.detect(img,None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'ORB_FREAK':
		orb = cv2.ORB_create(1000)
		feature_pts = orb.detect(img,None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)


	elif detector == 'ORB_LUCID':
		orb = cv2.ORB_create(1000)
		feature_pts = orb.detect(img,None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

####################################  BRISK  as detector ################################################

	elif detector == 'BRISK_BRIEF':
		brisk = cv2.BRISK_create()
		feature_pts = brisk.detect(img, None)
		
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		feature_pts, des = brief.compute(img, feature_pts)


	elif detector == 'BRISK_VGG':
		brisk = cv2.BRISK_create()
		feature_pts = brisk.detect(img, None)

		retval = cv2.xfeatures2d.VGG_create()
		des = retval.compute(img,feature_pts)


	elif detector == 'BRISK_BoostDesc':
		brisk = cv2.BRISK_create()
		feature_pts = brisk.detect(img, None)

		retval = cv2.xfeatures2d.BoostDesc_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'BRISK_LATCH':
		brisk = cv2.BRISK_create()
		feature_pts = brisk.detect(img, None)

		retval = cv2.xfeatures2d.LATCH_create()
		feature_pts,des = retval.compute(img,feature_pts)

	elif detector == 'BRISK_DAISY':
		brisk = cv2.BRISK_create()
		feature_pts = brisk.detect(img, None)

		retval = cv2.xfeatures2d.DAISY_create()
		feature_pts,des = retval.compute(img,feature_pts)

	
	elif detector == 'BRISK_FREAK':
		brisk = cv2.BRISK_create()
		feature_pts = brisk.detect(img, None)

		freakExtractor = cv2.xfeatures2d.FREAK_create()
		feature_pts,des= freakExtractor.compute(img,feature_pts)


	elif detector == 'BRISK_LUCID':
		brisk = cv2.BRISK_create()
		feature_pts = brisk.detect(img, None)

		retval = cv2.xfeatures2d.LUCID_create()
		feature_pts, des = retval.compute(img,feature_pts)

#######################################  SIFT as Descriptor #######################################

	elif detector == 'SimpleBlobDetector_SIFT':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)
		
		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts, des = sift.compute(img,feature_pts)


	elif detector == 'FAST_SIFT':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts, des = sift.compute(img,feature_pts)


	elif detector == 'GFTT_SIFT':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts, des = sift.compute(img,feature_pts)

	elif detector == 'AGAST_SIFT':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts, des = sift.compute(img,feature_pts)

	elif detector == 'STAR_SIFT':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts, des = sift.compute(img,feature_pts)

	
	elif detector == 'MSER_SIFT':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		sift = cv2.xfeatures2d.SIFT_create()
		feature_pts, des = sift.compute(img,feature_pts)

###################################### SURF as Descriptor #############################################

	elif detector == 'SimpleBlobDetector_SURF':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)
		
		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts,des = surf.compute(img,feature_pts)


	elif detector == 'FAST_SURF':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts,des = surf.compute(img,feature_pts)


	elif detector == 'GFTT_SURF':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts,des = surf.compute(img,feature_pts)

	elif detector == 'AGAST_SURF':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts,des = surf.compute(img,feature_pts)

	elif detector == 'STAR_SURF':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts,des = surf.compute(img,feature_pts)

	
	elif detector == 'MSER_SURF':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		surf = cv2.xfeatures2d.SURF_create(1000)
		feature_pts,des = surf.compute(img,feature_pts)

####################################  ORB as Descriptor  #############################################

	elif detector == 'SimpleBlobDetector_ORB':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)
		
		orb = cv2.ORB_create(1000)
		feature_pts, des = orb.compute(img,feature_pts)


	elif detector == 'FAST_ORB':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		orb = cv2.ORB_create(1000)
		feature_pts, des = orb.compute(img,feature_pts)


	elif detector == 'GFTT_ORB':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		orb = cv2.ORB_create(1000)
		feature_pts, des = orb.compute(img,feature_pts)

	elif detector == 'AGAST_ORB':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		orb = cv2.ORB_create(1000)
		feature_pts, des = orb.compute(img,feature_pts)

	elif detector == 'STAR_ORB':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		orb = cv2.ORB_create(1000)
		feature_pts, des = orb.compute(img,feature_pts)

	
	elif detector == 'MSER_ORB':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		orb = cv2.ORB_create(1000)
		feature_pts, des = orb.compute(img,feature_pts)

################################### BRISK as Descriptor #############################################

	elif detector == 'SimpleBlobDetector_BRISK':
		retval = cv2.SimpleBlobDetector_create()
		feature_pts = retval.detect(img,None)
		
		brisk = cv2.BRISK_create()
		feature_pts, des = brisk.compute(img, feature_pts)


	elif detector == 'FAST_BRISK':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		brisk = cv2.BRISK_create()
		feature_pts, des = brisk.compute(img, feature_pts)


	elif detector == 'GFTT_BRISK':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		brisk = cv2.BRISK_create()
		feature_pts, des = brisk.compute(img, feature_pts)

	elif detector == 'AGAST_BRISK':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		brisk = cv2.BRISK_create()
		feature_pts, des = brisk.compute(img, feature_pts)

	elif detector == 'STAR_BRISK':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		brisk = cv2.BRISK_create()
		feature_pts, des = brisk.compute(img, feature_pts)

	
	elif detector == 'MSER_BRISK':
		star = cv2.xfeatures2d.StarDetector_create()
		feature_pts = star.detect(img,None)

		brisk = cv2.BRISK_create()
		feature_pts, des = brisk.compute(img, feature_pts)

	elif detector == 'Tomasi_ORB':
		orb = cv2.ORB_create()
    		features = cv2.goodFeaturesToTrack(img, 1000, qualityLevel=0.01, minDistance=7)

		keypoints = [ cv2.KeyPoint(x = feature[0][0], y = feature[0][1], _size = 20) for feature in features]
		feature_pts, des = orb.compute(img, keypoints)

	# return features extracted

	feature_pts = np.array([x.pt for x in feature_pts], dtype=np.float32)

	return feature_pts

#################################################################################################

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

#################################################################################################
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

#################################################################################################
# Function that returns a scale from the GPS data
def getAbsoluteScale(f, frame_id):
	x_pre, y_pre, z_pre = f[frame_id-1][3], f[frame_id-1][7], f[frame_id-1][11]
	x_cur, y_cur, z_cur = f[frame_id][3], f[frame_id][7], f[frame_id][11]
	scale = np.sqrt((x_cur-x_pre)**2 + (y_cur-y_pre)**2 + (z_cur-z_pre)**2)
	return x_cur, y_cur, z_cur, scale


#################################################################################################

def getImages(i):
    return cv2.imread('/home/ali/Desktop/Data/kitti/00/image_0/{0:06d}.png'.format(i), 0)

#################################################################################################

def getTruePose():
    file = '/home/ali/Desktop/Data/kitti/poses/00.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)

#################################################################################################
# Function that takes a vector and returns its normilized form
def Norm_Vector(t):
	return t/sqrt([0]*t[0] + t[1]*t[1] + t[2]*t[2])

#################################################################################################

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

#################################################################################################

def ComputeEpipolarError(pts1, pts2, R, t, K):# Compute the error for 2 points pts1 and pts2
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

#################################################################################################

def ComputeEpipolarErrorTotal(pts1, pts2, R, t, K):# Compute the error for a vector of points pts1 and pts2
	error = 0
	for i in range(0,pts.shape):
		e = ComputeEpipolarError(pts1[i], pts2[i], R, t, K)
		error += e*e

	return sqrt(error/pts.shape)

#################################################################################################
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

#################################################################################################
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

		squareDist1 = (pts1_tmp[0] - u2in1)*(pts1_tmp[0] - u2in1) + (pts1_tmp[1] - v2in1)*(pts1_tmp[1] - v2in1)
		chisquare1 = squareDist1/(sigma*sigma)

		if (chisquare1>th):
			score +=0
		else:
			score += th - chisquare1 


		pts1in2 = H.dot(pts1_tmp)
	
		u1in2 = pts1in2[0]/pts1in2[2]
		v1in2 = pts1in2[1]/pts1in2[2]

		squareDist2 = (pts2_tmp[0] - u1in2)*(pts2_tmp[0] - u1in2) + (pts2_tmp[1] - v1in2)*(pts2_tmp[1] - v1in2)
		chisquare2 = squareDist2/(sigma*sigma)

		if (chisquare2>th):
			score +=0
		else:
			score += th - chisquare2

	q.put(score) ## used for returning value after threading
	return score
 
#################################################################################################
# Function that takes the previous and the current keypoints and computes the essential matrix and its inliers
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

#################################################################################################
# Function that takes the previous and the current keypoints and computes the homography matrix and its inliers
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

#################################################################################################
# Function that tries to choose a matrix (E or H) and estimate the motion of the robot(transaltion vector + rotation matrix)
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

#################################################################################################

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

#################################################################################################
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
###############################################################################################
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
	



