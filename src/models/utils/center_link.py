import numpy as np
import torch


class CenterLink():

    def __init__(self, center: tuple, intensity: float):
        self.next = None
        self.prev = None
        self.center = center
        self.intensity = intensity

    def __sub__(self, other):
        x, y, _ = self.get_center()
        x_other, y_other, _ = other.get_center()
        return (x - x_other, y - y_other)

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
                raise RuntimeError('Circular list of links detected')
            if next:
                links.append(next)
                center = next
            else:
                break
        centers = [link.get_center()
                   for link in links]
        intensities = [link.get_intensity()
                       for link in links]

        self.center = np.mean(centers, axis=0)
        self.intensity = np.mean(intensities, axis=0)
        self.links = links

    def __repr__(self):
        return self.get_center(), self.get_intensity()

    def get_center(self):
        return self.center

    def get_intensity(self):
        return self.intensity

    def get_links(self):
        return self.links


def get_bbox_center(bbox, z):
    '''
    Returns the center of the input bounding box.

    Inputs:
        bbox: list containing xmin, ymin, xmax, ymax
        z: integer, the z-dimension coordinate of the bounding box
    Outut:
        center of bbox in 3D coordinates (X, Y, Z)
    '''
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    w = x2 - x1
    h = y2 - y1
    center = int(x1 + w / 2), int(y1 + h / 2), z

    return center


def calculate_distance(centerlink_1, centerlink_2):
    return np.linalg.norm(centerlink_1 - centerlink_2)


def find_closest_center(z: int, center: CenterLink,
                        centers: list, searchrange: int = 3):
    '''
    Finds the next closest center in the slice after slice z.
    The found center must be within `searchrange` distance of `center`.

    Inputs:
        z: the current slice.
        center: the current CenterLink.
        centers: CenterLink of next slice (z+1).
        searchrange: the range of which to search for a
            candidate center, in (x,y) coordinates.

    Output:
        a center within `searchrange` distance, or None if none is found.
    '''
    candidates = []
    for candidate in centers:
        distance = calculate_distance(center, candidate)
        if distance <= searchrange:
            # There are no competitors for this candidate
            candidates.append((candidate, distance))

    if len(candidates) == 0:
        return None

    distances = [cand[1] for cand in candidates]
    argsort = np.argsort(distances)

    for i in argsort:
        best = candidates[i][0]
        # No competitor; we can return this candidate
        if not best.get_prev():
            break

        # There is a competitor for this candidate.
        # Need to find out which, center or candidate
        # previous link, has the shorter distance to
        # the candidate
        cand_prev = best.get_prev()
        cand_prev_dist = calculate_distance(cand_prev, best)
        center_dist = calculate_distance(center, best)
        if cand_prev_dist < center_dist:
            # Our center won against the competitor
            # Need to assign a new next-slice-center to
            # competitor
            new_centers = centers.copy()
            new_centers.remove(best)
            closest_second = find_closest_center(
                z, cand_prev, new_centers, searchrange)
            cand_prev.set_next(closest_second)
        else:
            # Our center lost to the competitor
            # nothing to be done except move to next
            # candidate
            pass

    return best


def get_chains(timepoint: np.array, preds: list, searchrange: int = 3):
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
        chains: A list for each slice.
            Each element in the list contains the centroids for that slice.
            The centroids are of type CenterLink.
            As before, the average is taken over all occurrences of the
            cell across all slices.
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

    image_iter = zip(timepoint.squeeze(1), masks_tot, bboxes_tot)
    for z, (z_slice, masks, bboxes) in enumerate(image_iter):  # each slice
        centers = []

        for mask, bbox in zip(masks, bboxes):  # each box in slice
            center = get_bbox_center(bbox, z)

            # Coordinates in original slice
            mask_nonzero = np.argwhere(mask.detach().cpu())

            # Get cell coordinates in original image (z_slice)
            region = z_slice[mask_nonzero].detach().cpu()

            # Calculate average pixel intensity
            intensity = np.mean(region.numpy())

            center_link = CenterLink(center, intensity)
            centers.append(center_link)

        centers_tot.append(centers)

    for i, centers in enumerate(centers_tot):  # each slice
        for j, center in enumerate(centers):  # each center in slice
            if i == len(centers_tot) - 1:
                continue
            # search in the direction of z-dimension
            # a total of `searchrange` pixels in all directions
            closest_center = find_closest_center(
                j, center, centers_tot[i + 1], searchrange)
            # maybe it was assigned before
            if closest_center is not None:
                closest_center.set_prev(center)

            center.set_next(closest_center)

    # Get only first link of each chain because that's all we need
    # (link is first occurring part of each cell)
    # link = cell occurrence in slice, chain = whole cell
    links = [
        center for centers in centers_tot for center in centers if center.get_prev() is None]
    # Get chains (= whole cells)
    # which contains average pixel intensity and average center
    # for the whole chain (i.e. over all links in the chain)
    chains = [CenterChain(link)
              for link in links if link.get_center() != np.nan]

    return chains
