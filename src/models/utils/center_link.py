import numpy as np


class CenterLink():
    '333333333333333333333333333333333333333333333333333333333333333333'

    def __init__(self, center: tuple, intensity: float):
        self.next = None
        self.prev = None
        self.center = center
        self.intensity = intensity

    def set_next(self, center):
        self.next = center

    def set_prev(self, center):
        self.prev = center

    def set_intensity(self, intensity):
        self.intensity = intensity

    def get_next(self):
        return self.next

    def get_prev(self):
        return self.prev

    def get_center(self):
        return self.center

    def get_intensity(self):
        return self.intensity


class CenterChain():

    def __init__(self, center: CenterLink):
        links = []
        while True:
            next = center.get_next()
            if next in links:
                raise ValueError('Circular list of links detected')
            if next:
                links.append(next)
                center = next
            else:
                break
        centers = np.array([link.center for link in links])
        intensities = np.array([link.intensity for link in links])

        self.center = np.mean(centers)
        self.intensity = np.mean(intensities)

    def get_center(self):
        return self.center

    def get_intensity(self):
        return self.intensity


def get_bbox_center(bbox, z):
    '''
    Returns the center of the input bounding box.

    Inputs:
        bbox: list containing xmin, ymin, xmax, ymax
        z: integer, the z-dimension coordinate of the bounding box
    Outut:
        center of bbox in 3D coordinates (X, Y, Z)
    '''
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    center = int(x1 + w/2), int(y1 + h/2), z
    return center


def calculate_distance(center1, center2):
    return np.linalg.norm(center2-center1)


def find_closest_center(z: int, center: CenterLink,
                        centers: list, searchrange: int = 3):
    '''
    Finds the next closest center in the slice after slice i.
    The found center must be within `searchrange` distance of `center`.

    Inputs:
        z: the current slice.
        center: the current center.
        centers: centers of next slice (z+1).
        searchrange: the range of which to search for a 
        candidate center, in (x,y) coordinates.

    Output:
        a center within `searchrange` distance, or None if none is found.
    '''
    candidates = []
    for candidate in centers:
        distance = calculate_distance(center, candidate)
        if distance <= searchrange:
            candidates.append(candidate)

    return np.min(candidates)
        


def get_chains(timepoint: np.array, preds: list, searchrange: int = 1):
    '''
    Returns the chains of the input timepoint.
    Each chain represents a cell. Each chain is composed of connected links,
    with links to next and/or previous links in the chain.
    Beginning link has no previous link. Last link has no next link.

    Input arguments:
        timepoint: 3D Numpy array of dimensions
            (D, 1024, 1024) where D is the z-dimension length.
        pred: List of model outputs of type dictionary,
            containing boxes, masks, etc. (From Mask R-CNN).
            Each element in this list represents a slice from
            the timepoint.
        searchrange: the tolerance when searching for a center belonging
            to current cell. Default: 3 pixels

    Output:
        centroids: A list for each slice. Each element in the list
            contains the centroids for that slice.
            The centroids are of type CenterLink.
            As before, the average is taken over all occurrences of the
            cell across all slices.

     963 33 2 3
    '''
    # need pixel intensity also
    # fetched from original image
    # for each instance of mask (displayed in a
    # separate tensor) we get the coordinates
    # for which the mask is nonzero.
    # Take from these coordinates, in the original
    # image, the average pixel intensity.

    # Total bounding boxes and masks (each element is for each slice)
    # Each element is also a list; but is now all elements inside the slice;
    # bounding boxes in bbox_tot, and masks in masks_tot.
    bboxes_tot = [pred['boxes'] for pred in preds]
    masks_tot = [pred['masks'] for pred in preds]
    centers_tot = []

    # Get average pixel intensities
    # for z_slice, masks in zip(timepoint, masks_tot):
    #     intensities = []
    #     for mask in masks:
    #         mask_nonzero = np.argwhere(mask)
    #         # Coordinates in original slice
    #         region = z_slice[mask_nonzero]
    #         intensity = np.mean(region)
    #         intensities.append(intensity)
    #     intensities_tot.append(intensities)
    # pred is 3d

    centroids = None
    image_iter = zip(timepoint, masks_tot, bboxes_tot)
    for z, (z_slice, masks, bboxes) in enumerate(image_iter):  # each slice
        centers = []
        for mask, bbox in zip(masks, bboxes):  # each box in slice
            center = get_bbox_center(bbox, z)
            # Coordinates in original slice
            mask_nonzero = np.argwhere(mask)
            # Get cell coordinates in original image (z_slice)
            region = z_slice[mask_nonzero]
            # Calculate average pixel intensity
            intensity = np.mean(region)

            center_link = CenterLink(center, intensity)
            centers.append(center_link)

        centers_tot.append(centers)

    for i, centers in enumerate(centers_tot):  # each slice
        for j, center in enumerate(centers):  # each center in slice
            # search in the direction of z-dimension
            # a total of `searchrange` pixels in all directions
            closest_center = find_closest_center(
                j, center, centers[i + 1], searchrange)
            if closest_center is not None:
                closest_center.set_prev(center)
            center.set_next(closest_center)

    # Get only first link of each chain because that's all we need
    # (link is first occurring part of each cell)
    # link = cell occurrence in slice, chain = whole cell
    centroids = [center for center in centers_tot if center.get_prev() is None]
    # Get chains (= whole cells)
    # which contains average pixel intensity and average center
    # for the whole chain (i.e. over all links in the chain)
    chains = [CenterChain(center) for center in centroids]
    return chains
