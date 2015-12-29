
import nibabel
import sys
import cv2

mr = sys.argv[1]

mr = "brain_parcellation_mcinet_basc_asym_111clusters.nii.gz"
nii = nibabel.load(mr)

# Take some random slice (could be user selected)
slice = nii.get_data()[:,:,20]

# Can we convert numpy array to grayscale?
im = numpy.array(slice * 255, dtype = numpy.uint8)  # Note - will need to save mapping here from labels to new values
threshed = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(im, 50, 100)
cv2.imshow("Original", edged)
 
# find contours in the image
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
mask = numpy.ones(slice.shape[:2], dtype="uint8") * 255
 
# loop over the contours
for c in cnts:
    # write me - output coordinates to interactive image
 
# remove the contours from the image and show the resulting images
image = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask", mask)
cv2.imshow("After", image)
cv2.waitKey(0)
