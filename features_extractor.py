import cv2
import numpy as np


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
