"""cutoff_args := (voxel_cumul_dists, occlusions, old_occ_voxels, voxel_counts, nr_pos)"""
from enum import Enum

import numpy as np


class CutoffState(Enum):
    CONTINUE = 1
    SKIP = 2
    STOP = 3


def all_voxels_under_thresh(
        cutoff_args: (list, {(float, float, float): [(int, int, int)]}, any, any, any)) -> CutoffState:
    voxel_cumul_dists, occlusions, _, __, ___ = cutoff_args

    # Init vars
    thresh = 0.3
    uncovered_voxels = 0

    occ_voxels = []
    for pos in occlusions.keys():
        occ_voxels.extend(occlusions[pos])

    # Occluded voxels will automatically have coverage of 0,
    # but we don't want to continue infinitely because of a occluded voxel.
    occluded_matrix = np.zeros(np.shape(voxel_cumul_dists))
    for occ_vox in occ_voxels:
        occluded_matrix[occ_vox[0]][occ_vox[1]][occ_vox[2]] = thresh

    dist_or_occluded = np.ravel(voxel_cumul_dists + occluded_matrix)

    for d in dist_or_occluded:
        if d < thresh:
            uncovered_voxels += 1
    if uncovered_voxels <= 0:
        return CutoffState.STOP
    return CutoffState.CONTINUE


def occluded_cutoff(cutoff_args: (any, any, {
    (float, float, float): [(int, int, int)]}, [int], int)) -> CutoffState:
    _, __, old_occlusions, voxel_counts, nr_pos = cutoff_args
    pos_thresh = 3

    # If we have executed more than double poses than occluding poses, we return
    # if nr_pos > len(old_occlusions.keys()) * 4:
    #     return True

    occ_voxels = []
    for pos in old_occlusions.keys():
        voxels = old_occlusions[pos]
        for vox in voxels:
            if not any(x == vox for x in occ_voxels):
                occ_voxels.append(vox)

    if nr_pos > len(occ_voxels) * pos_thresh:
        return CutoffState.STOP

    # Check that all ex-occluded voxels are now covered at least once
    uncovered_occ_voxels = 0
    overstep_thresh_voxels = 0
    for voxel in occ_voxels:
        v = voxel_counts[voxel[0]][voxel[1]][voxel[2]]
        if v < 2: #  Try it from at least 2 sides
            uncovered_occ_voxels += 1

        if v > pos_thresh:
            overstep_thresh_voxels += 1
    if uncovered_occ_voxels > 0:
        return CutoffState.CONTINUE
    elif uncovered_occ_voxels == 0:
        return CutoffState.STOP

    if overstep_thresh_voxels > 0:
        return CutoffState.SKIP
    return CutoffState.STOP
