import numpy as np
import math


class CenterLink():

    def __init__(self, center: tuple, intensity: float):
        self.chain = None
        self.next = None
        self.prev = None
        self.center = center
        self.intensity = intensity

    def __sub__(self, other):
        x, y, _ = self.get_center()
        x_other, y_other, _ = other.get_center()
        return (x - x_other, y - y_other)

    def set_next(self, center):
        # if type(self.next) == list:
        #     self.next.append(center)
        # elif type(self.next) == CenterLink:
        #     self.next = [self.next, center]
        # elif self.prev is None:
        self.next = center

    def set_prev(self, center):
        # if type(self.prev) == list:
        #     self.prev.append(center)
        # elif type(self.prev) == CenterLink:
        #     self.prev = [self.prev, center]
        # elif self.prev is None:
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
        # TODO: make CenterChain realize when it has been made
        # of two separate chains of links which were matched with
        # a common center.
        links = [center]
        next = center.get_next()

        while next:
            links.append(next)

            # cell has banana shape with
            # pointy ends facing up;
            # this is the end point
            if type(next.get_prev()) in [CenterLink, list]:
                # type is either CenterLink or list,
                # so `next` has one or more previous links
                pass

            # cell has banana shape with
            # pointy ends facing down;
            # this is the starting point
            if type(next) in [CenterLink, list]:
                # type is either CenterLink or list,
                # so current center (`center`) has one or more
                # next links
                pass

            next = next.get_next()
            if next in links:
                raise RuntimeError('Circular list of links detected')

        centers = [link.get_center()
                   for link in links]
        intensities = [link.get_intensity()
                       for link in links]

        center_mean = np.round(np.mean(centers, axis=0), 2)
        intensity = np.round(np.mean(intensities, axis=0), 10)

        owner = links[0].chain
        if owner not in [self, None]:  # links already have an owner chain
            chain = links[0].chain

            # need to use adjusted mean to take the mean correctly
            # there may have been a previous mean performed
            owner_nlinks = len(chain.get_links())
            new_center = (center_mean + chain.get_center() *
                          owner_nlinks) / (len(links) + owner_nlinks)

            # TODO: adjust mean of intensity...
            new_intensity = np.mean(intensity, chain.get_intensity())

            # add current link information to owner chain
            chain.set_center(new_center)
            chain.set_intensity(new_intensity)
            chain.set_links(chain.get_links() + links)

            # deactivate current chain (self);
            # will be filtered out later
            self.center = None
            self.intensity = None
            self.links = None

        self.center = center_mean
        for link in links:
            link.chain = self
        self.intensity = intensity
        self.links = links

    def __len__(self):
        return len(self.links)

    def __repr__(self):
        return self.get_center(), self.get_intensity()

    def get_center(self):
        x, y, z = self.center
        return x, y, z

    def get_intensity(self):
        return self.intensity

    def get_links(self):
        return self.links

    def get_state(self):
        x, y, z = self.get_center()
        intensity = self.get_intensity()
        return x, y, z, intensity

    def set_center(self, center):
        self.center = center

    def set_intensity(self, intensity):
        self.intensity = intensity


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
        centers: CenterLink[] of next slice (z+1).
        searchrange: the range of which to search for a
            candidate center, in (x,y) coordinates.

    Output:
        a center within `searchrange` distance, or None if none is found.
    '''
    candidates = []
    for candidate in centers:
        distance = calculate_distance(center, candidate)
        if distance <= searchrange:
            candidates.append((candidate, distance))

    if len(candidates) == 0:
        return None

    distances = [cand[1] for cand in candidates]
    argsort = np.argsort(distances)

    # for i in argsort:
    #     best = candidates[i][0]
    #     # No competitor; we can return this candidate
    #     if not best.get_prev():
    #         break

    #     # There is a competitor for this candidate.
    #     # Need to find out which, center or candidate
    #     # previous link, has the shorter distance to
    #     # the candidate
    #     cand_prev = best.get_prev()
    #     cand_prev_dist = calculate_distance(cand_prev, best)
    #     center_dist = calculate_distance(center, best)
    #     if cand_prev_dist < center_dist:
    #         # Our center won against the competitor
    #         # Need to assign a new next-slice-center to
    #         # competitor
    #         new_centers = centers.copy()
    #         new_centers.remove(best)
    #         closest_second = find_closest_center(
    #             z, cand_prev, new_centers, searchrange)
    #         cand_prev.set_next(closest_second)
    #     else:
    #         # Our center lost to the competitor
    #         # nothing to be done except move to next
    #         # candidate
    #         pass

    return candidates[argsort[0]][0]


def get_chains(timepoint: np.array, preds: list, searchrange: int = 10):
    '''
    Returns the chains of the input timepoint.
    Each chain represents a cell. Each chain is composed of connected links,
    with links to next and/or previous links in the chain.
    Beginning link has no previous link. Last link has no next link.

    Input arguments:
        timepoint: 3D Numpy array of dimensions
            (D, 1024, 1024) where D is the z-dimension length.
        preds: List of model outputs of type dictionary,
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

    # Total bounding boxes and masks (each element is for each slice)
    # Each element is also a list; but is now all elements inside the slice;
    # bounding boxes in bbox_tot, and masks in masks_tot.
    bboxes_tot = (pred['boxes'] for pred in preds)
    masks_tot = (pred['masks'] for pred in preds)
    centers_tot = []

    image_iter = zip(timepoint, masks_tot, bboxes_tot)
    for z, (z_slice, masks, bboxes) in enumerate(image_iter):  # each slice
        centers = []

        for mask, bbox in zip(masks, bboxes):  # each box in slice
            center = get_bbox_center(bbox, z)

            # Mask coordinates in original slice
            mask_nonzero = np.argwhere(mask.detach().cpu())

            # Get cell coordinates in original image (z_slice)
            region = z_slice[mask_nonzero].detach().cpu()

            # Calculate average pixel intensity
            # TODO: average intensity: take only pixels above
            # a certain threshold
            intensity = np.mean(region.numpy())
            # if intensity == np.nan or math.isnan(intensity):
            #     print('\n\n\nintensity nan')
            #     print('region shape', region.shape)
            #     print('unique region', np.unique(region, return_counts=True))
            #     print('masknonzero shape', mask_nonzero.shape)
            #     print('masknonzero unique', np.unique(
            #         mask_nonzero.cpu(), return_counts=True))

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

            if closest_center is not None:
                closest_center.set_prev(center)

            center.set_next(closest_center)

    # Get only first link of each chain because that's all we need
    # (link is first occurring part of each cell)
    # link = cell occurrence in slice, chain = whole cell
    # links = [
    #     center for centers in centers_tot for center in centers if center.get_prev() is None]

    # get first link of every chain
    # (chain = whole cell, link = slice of cell)
    chains = [
        CenterChain(link).get_state()
        for links in centers_tot
        for link in links
        if link.get_prev() is None and
        link.get_center() is not None
    ]
    # Get chains (= whole cells)
    # which contains average pixel intensity and average center
    # for the whole chain (i.e. over all links in the chain)
    # chains = [CenterChain(link).get_state()
    #           for link in links
    #           if not np.isnan(link.get_center()).any()]

    # chains = [[(center.get_center(), center.get_intensity())
    #            for center in centers]
    #           for centers in chains]

    # chains = [[(center[0], center[1],
    #             center[2], center.get_intensity())
    #            for center in centers]
    #           for centers in chains]

    return chains
