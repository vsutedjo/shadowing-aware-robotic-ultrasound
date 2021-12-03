import quantumrandom
import random
from functions.helpers.helpers import calculate_possible_poses


def get_random_poses(nr_rows, nr_cols, angles, nr_poses=None):
    poses = []
    possible_poses = calculate_possible_poses(nr_rows, nr_cols, angles)
    if nr_poses is None:
        # if no limit has been handed over, we take as many poses as we have surface voxels
        nr_poses = nr_rows * nr_cols
    nr_possible_poses = len(possible_poses)

    random.shuffle(possible_poses)
    poses = possible_poses[0:nr_poses]

    # for i in range(nr_poses):
    #     index = int(quantumrandom.randint(0, nr_possible_poses))
    #     poses.append(possible_poses[index])
    return poses
