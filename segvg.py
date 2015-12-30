import nibabel
import pandas
import sys
import cv2
from scipy.spatial.distance import pdist
import numpy

#mr = sys.argv[1]

mr = "brain_parcellation_mcinet_basc_asym_33clusters.nii.gz"
axis == 0
nii = nibabel.load(mr)

# We will squash the image from the top, for each unique value
regions = numpy.unique(nii.get_data()).tolist()

# We don't want to contour value of 0 (nothingness)
regions.pop(regions.index(0))

# We will save a list of segments
# needs to look like var segments = [{
#                                     points:[[14, 44], [14, 46],[15, 46],[14, 45],[15, 44]]
#                                     center:[15,19],
#                                  }]
segments = []

for value in regions:
    empty = numpy.zeros(nii.shape)
    empty[nii.get_data()==value] = value
    squash = empty.sum(axis=axis) # user can select view
    
    # Rotate brain depending on view
    if axis==0:
        squash = numpy.rot90(squash,1)

    squash[squash!=0] = 1      # binarize
    # Can we convert numpy array to grayscale?
    im = numpy.array(squash * 255, dtype = numpy.uint8)  # Note - will need to save mapping 
    ret, thresh = cv2.threshold(im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    # Note: need way to separate right and left
    
    if len(cnts)==2:
        # We need to get the center
        bx,by,bw,bh = cv2.boundingRect(cnts[1])

        # Use by as a proxy for distance to front of scene
        if axis == 0:
            sideview = pandas.DataFrame(empty.sum(axis=1))
            listy=sideview.sum(axis=1).tolist()
            # Where is the first nonzero index? (distance to object)
            opacity = listy.index(filter(lambda x: x!=0, listy)[0])
            opacity = opacity/float(empty.shape[2])        

        new_segment = {"points":[x[0] for x in cnts[1].tolist()],
                       "center":[31,8],
                       "opacity":opacity} # This is actually top corner
        segments.append(new_segment)


# Make substitution in template
filey = open("template.html","r")
template = "\n".join(filey.readlines())
filey.close()
template = template.replace("{{DATA}}",str(segments))
filey = open("index.html","w")
filey.writelines(template)
filey.close()
