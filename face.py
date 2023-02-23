import os, copy, glob, contextlib, pydicom

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2 as cv
import nibabel as nib

from scipy import ndimage
from mtcnn import MTCNN

d = MTCNN(min_face_size = 10, scale_factor = 0.95)
hc = cv.CascadeClassifier(
        os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml'))

def load_dicom(dirname):
    """
    Load a 3d PET or CT, or a 4d PET
    4d PET doesn't currently work with any code in this repo
    """

    slice_files = glob.glob(os.path.join(dirname, '*'))
    slices = [pydicom.dcmread(s) for s in slice_files]
    s = slices[0]
    modality = str(s.Modality)

    if modality == 'PT':
        slices = sorted(slices, key = lambda s: s.ImageIndex)
        pix_size = tuple(s.PixelSpacing)
        pix_size += (float(s.SliceThickness),)

        if s.NumberOfSlices == len(slice_files):
            # static PET
            vol_dim = (s.NumberOfSlices,)
        else:
            # dynamic PET
            vol_dim = (s.NumberOfTimeSlices, s.NumberOfSlices)
            pix_size += (1000.0,) # use a dummy frame duration

    elif modality == 'CT':
        slices = sorted(slices, key = lambda s: float(s.SliceLocation), reverse = True)
        pix_size = tuple(s.PixelSpacing)
        pix_size += (float(s.SliceThickness),)
        vol_dim = (int(s.ImagesInAcquisition),)

    else:
        raise ValueError(f'Unknown modality {modality}')

    slices = np.array(slices)
    img = np.zeros(vol_dim + s.pixel_array.shape)
    slices = slices.reshape(vol_dim)

    for idx in np.ndindex(slices.shape):
        img[idx] = slices[idx].pixel_array
        img[idx] *= slices[idx].RescaleSlope

    img = img.transpose()
    img = np.flip(img, (0,1,2))
    nimg = nib.Nifti1Image(img, np.eye(4))
    nimg.header.set_zooms(pix_size)
    return nimg, modality

def make_offset_img(nimg, modality):
    """
    projects the original 3d image into 2d by thresholding,
    then finding the distance from the back of the volume to
    the first surface in the image.
    This is referred to as the 'offset' image
    """

    d = nimg.get_fdata()

    if modality == 'CT':
        # center transverse slice
        middle = int(d.shape[2]/2)
        slc = d[:,:,middle]

        # rescale to 8 bits and use otsu thresholding
        smax = slc.max()
        slc = (slc / smax * 255.0).astype(np.uint8)
        thr, mask = cv.threshold(slc, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thr = (thr / 255) * smax
        mask = d > thr
        return mask[:,::-1].argmax(1)

    elif modality == 'PT':
        # center coronal slice
        middle = int(d.shape[1]/2)
        slc = d[:,middle,:]

        thr = slc.mean() / 2
        mask = d > thr

        mask = ndimage.binary_opening(mask)
        mask = ndimage.binary_closing(mask)
        
        img = mask[:,::-1].argmax(1)

        # remove some image noise, probably due to scatters
        img = ndimage.grey_closing(img, size = (3,3))
        return img

    else:
        raise RuntimeError(f'Unrecognized modality: {modality}')

    # calculate the distance of the surface from the back of the volume

def make_grad_img(offset, gmax):
    """
    compute the 2d gradient of the 'offset' image, and rescale
    The gradient is very high at the edges, so clip min/max values
    to emphasize the surface contours
    """
    gradx = ndimage.sobel(offset, 0)
    grady = ndimage.sobel(offset, 1)
    grad = np.hypot(gradx,grady)
    grad = np.clip(grad, -gmax, gmax)
    grad = 1 - (grad / grad.max())
    return (grad * 255).astype(np.uint8).T

def disp_face(grad, coords):
    """ plot the gradient image and draw a box on the detected face
    """
    l,b,w,h = coords

    fig, ax = plt.subplots()
    ax.imshow(grad, cmap = 'gray')

    rect = patches.Rectangle((l,b),w,h,
            linewidth = 1,
            edgecolor = 'r',
            facecolor = 'none')

    ax.add_patch(rect)

    ax.invert_yaxis()
    plt.show()

def find_face_mtcnn(grad):
    """ use the MTCNN detector to find a face in the gradient image
    """
    img = cv.cvtColor(np.flipud(grad), cv.COLOR_GRAY2RGB)

    with open(os.devnull, 'w') as null:
        with contextlib.redirect_stdout(null):
            face = d.detect_faces(img)

    if len(face) == 0:
        return

    if len(face) > 1:
        print(f'found {len(face)} faces: {face}')

    # list of faces is ranked by probability (right?)
    l,b,w,h = face[0]['box']
    dy,dx = grad.shape
    b = dy - b - h - 1
    return l, b, w, h

def find_face_haar(grad, pixdim):
    """ Find faces with the HCC and pick the one with the highest confidence
    """

    minsize = int(100.0 / pixdim) # 10cm
    maxsize = int(200.0 / pixdim) # 20cm

    faces, _, level_weights = hc.detectMultiScale3(
            grad,
            scaleFactor = 1.02,
            minNeighbors = 2,
            minSize = (minsize,minsize),
            maxSize = (maxsize,maxsize),
            outputRejectLevels = True)

    if len(faces) == 0:
        return

    # pick the face with the greatest detection confidence
    return faces[np.argmax(level_weights)]

def find_face(render, pixdim, plot = False):
    """ Attempt to find a face with MTCNN, and fall-back to HCC if necessary
    """

    # pick a general window in the image where the face ought to be
    # reduces false positive, but would need customization by protocol
    yo = int(render.shape[0]/3)*2
    xo = int(render.shape[1]/4)
    face_region = render[yo:,xo:(3*xo)] # top 1/3 and central 1/2

    coords = find_face_mtcnn(face_region)
    if coords is None:
        print('falling back to Haar-cascade face detection')
        coords = find_face_haar(face_region, pixdim)

    if coords is None:
        plt.imshow(face_region, cmap = 'gray')
        plt.gca().invert_yaxis()
        plt.show()
        raise RuntimeError('Failed to detect a face in image')

    l, b, w, h = coords
    coords = (l+xo, b+yo, w, h)

    if plot:
        disp_face(render, coords)

    return coords

def find_head_in_offset_img(offset):
    """
    Find the largest nonzero image region around the face. Helps
    to cover the whole front of the head and exclude shoulders.
    """

    # get connected components in the head box
    mask = (offset != 0).astype(np.uint8)
    se = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    mask = cv.erode(mask, se, iterations = 3)
    _, cc, st, ct = cv.connectedComponentsWithStats(mask)

    # remove regions with value 0 in the mask
    regions = np.unique(cc[mask.astype(bool)])
    st = st[regions]

    # take the largest region
    headidx = regions[np.argmax(st[:,4])]

    mask = (cc == headidx).astype(np.uint8)
    mask = cv.dilate(mask, se, iterations = 3)
    return mask.astype(bool)

def blur_face(nimg, offset, coords, order = 1):
    """
    Take a 3d image, an offset image, and coordinates of the face in the offset image,
    and return an 3d image where the specified face surface is blurred
    """

    zooms = nimg.header.get_zooms()
    depth = nimg.shape[1]

    back = int(10.0 / zooms[1])     # depth to blur into face (1cm)
    forward = int(30.0 / zooms[1])  # depth to blur out from face (3cm)
    pix = 8                         # downsampling factor when pixilating

    # ensure that the face dims are a multiple of the down-sampling
    l, b, w, h = coords
    w, h = w - w%pix, h - h%pix
    r, t = l + w, b + h
    
    img = copy.deepcopy(nimg.get_fdata())
    head = img[l:r,:,b:t] # 3D volume containing the head

    facemask = np.zeros_like(head, dtype = bool) # 3D mask of face pixels
    faceoffset = copy.deepcopy(offset[l:r, b:t]) # 2D mask of face offsets
    mask = find_head_in_offset_img(faceoffset) # fine-tune the mask to exclude shoulders

    faceoffset[~mask] = int(depth/2) # set non-face regions to the mid-scale value
    faceoffset = depth - faceoffset - 1 # invert face offset image

    # iterate over the coronal plane of the head volume
    # based on the face depth, create the 3D 'face mask'
    ni, nj = faceoffset.shape
    for i in range(ni):
        for j in range(nj):
            if mask[i,j]:
                # offset of the face from the front of the volume
                off = faceoffset[i,j]

                # ensure we don't extend outside the volume
                rear  = np.clip(off-back, 0, depth-1)
                front = np.clip(off+forward, 0, depth-1)

                facemask[i,rear:front,j] = True

    # create the pixilated face image, and ensure volume size doesn't change
    blur = ndimage.zoom(head, 1/pix, order = order)
    blur = ndimage.zoom(blur, pix, order = order)
    head[facemask] = blur[facemask]

    # create and return a new image
    blur = nib.Nifti1Image(img, np.eye(4))
    blur.header.set_zooms(zooms)
    return blur

def anon_image(dicom_dir, plot = False):
    """ Run the full workflow, taking a 3d image and returning an anonymized image
    """

    nimg, modality = load_dicom(dicom_dir)
    zooms = nimg.header.get_zooms()
    aspect = zooms[2]/zooms[0]
    pixdim = zooms[0]

    initial_offset = make_offset_img(nimg, modality)
    offset = ndimage.zoom(initial_offset, (1,aspect))
    
    gmax = pixdim*20 if modality == 'CT' else pixdim*10
    render = make_grad_img(offset, gmax)

    x,y,w,h = find_face(render, pixdim, plot)
    coords = (x, int(y/aspect), w, int(h/aspect))
    anon = blur_face(nimg, initial_offset, coords)

    if plot:
        offset = make_offset_img(anon, modality)
        offset = ndimage.zoom(offset, (1,aspect))
        render = make_grad_img(offset, gmax)
        plt.imshow(render, cmap = 'gray')
        plt.gca().invert_yaxis()
        plt.show()

    return anon
