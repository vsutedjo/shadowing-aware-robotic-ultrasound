import unittest
import torch
import functions.helpers.torch_optimization as topt
from functions.scan_calculators import calculate_distances
import numpy as np
import time
from functions.cost_functions import coverage_loss, occlusion_loss_preparation, occlusion_loss
import matplotlib.pyplot as plt
import pickle


class TestOptimization(unittest.TestCase):

    @staticmethod
    def print_array(arr):

        assert len(arr.shape) == 1

        to_print = ""
        for i in range(arr.shape[0]):
            to_print += "{0:.2f}".format(arr[i])

        print(to_print)

    def test_volume_coverage_with_occlusions(self):
        nr_rows = 3
        nr_cols = 4
        nr_planes = 2

        angles = [-45, -15, 0, 15, 45] # Write the angles from smallest to largest
        transducer_width = 1
        focal_dist = 3.5
        dims = (nr_rows, nr_cols, nr_planes)

        with open('test_data/test_E_penalty_with_occlusions/w.pickle', 'rb') as handle:
            w = pickle.load(handle)

        with open('test_data/test_E_penalty_with_occlusions/d.pickle', 'rb') as handle:
            d = pickle.load(handle)

        with open('test_data/test_E_penalty_with_occlusions/occluded_voxels.pickle', 'rb') as handle:
            occluded_voxels = pickle.load(handle)

        d_array = np.array([d[key] for key in d.keys()])

        w_0 = np.ones(len(d)) * 0.5

        loss_np = E(w_0, d, (nr_rows, nr_cols, nr_planes), angles, occluded_voxels)
        print("loss_np: ", loss_np)


        #
        occluded_voxels = np.array(occluded_voxels)
        occluded_volume = np.zeros(dims)
        occluded_volume[occluded_voxels[:, 0], occluded_voxels[:, 1], occluded_voxels[:, 2]] = 1


        loss_np_optimized = coverage_loss(w_0, d_array, occluded_volume)
        print("loss_np_optimized: ", loss_np_optimized)


    def test_volume_coverage(self):
        """
        The function tests that the numpy and torch optimized versions of the loss function return consistent weights
        with the one obtained with the original functions
        :return:
        """
        losses_old_method = []
        losses_optimized_np = []
        losses_optimized_torch_cpu = []
        losses_optimized_torch_gpu = []

        times_old_method = []
        times_optimized_np = []
        times_optimized_torch_cpu = []
        times_optimized_torch_gpu = []
        for i in range(10):
            nr_rows = 3 + i
            nr_cols = 4 + i
            nr_planes = 2 + i

            angles = np.linspace(0, 45, 2 + i).tolist()  # Write the angles from smallest to largest
            transducer_width = 1
            focal_dist = 3.5
            dims = (nr_rows, nr_cols, nr_planes)

            d_dict = calculate_distances(angles=angles,
                                         nr_rows=nr_rows,
                                         nr_cols=nr_cols,
                                         nr_planes=nr_planes,
                                         transducer_width=transducer_width,
                                         focal_dist=focal_dist,
                                         plot=False)

            d_array = np.array([d_dict[key] for key in d_dict.keys()])

            w_0 = np.random.random(len(d_dict))
            w_0 = np.around(w_0, decimals=2)  # rounding to 2nd decimal to make numerical errors negligible

            # Reordering weights as the old function expects them as
            # w = [w(angle_0, row_0, col_0),
            #      w(angle_0, row_0, col_1),
            #      w(angle_0, row_0, col_2),
            #      w(angle_0, row_0, col_3),
            #      w(angle_0, row_1, col_0),
            #               ...
            # While the torch and numpy optimized functions expects them as
            # w = [w(angle_0, row_0, col_0),
            #      w(angle_1, row_0, col_0),
            #      w(angle_0, row_0, col_1)
            #      w(angle_1, row_0, col_1)
            #      w(angle_0, row_0, col_2),
            #               ...
            # For the optimization itself it doesn't matter, as the weights are anyways randomly initialized,
            # but we add this reshaping here to make sure the result is consistent

            w_0_rearranged = np.reshape(w_0, [len(angles), nr_cols, nr_rows])
            w_0_rearranged = np.transpose(w_0_rearranged, [1, 2, 0])
            w_0_rearranged = w_0_rearranged.flatten()

            # Initializing the model and assigning weights
            model = topt.CoverageModel(d=d_array)
            model.weights.data = torch.Tensor(w_0_rearranged)

            # 1. Computing the loss with the old, numpy original method
            t1 = time.time()
            loss_np = E(w_0, d_dict, nr_rows, nr_cols, nr_planes, angles, dims)
            times_old_method.append(time.time() - t1)
            print("old numpy computation", times_old_method[-1])
            print("old numpy loss: ", loss_np)

            # 2. Computing the loss with the optimized, numpy method
            t1 = time.time()
            loss_np_optimized = coverage_loss(w_0_rearranged, d_array)
            times_optimized_np.append(time.time() - t1)
            print("optimized numpy computation", times_optimized_np[-1])
            print("optimized numpy loss: ", loss_np_optimized)

            # 3. Computing the loss with the optimized, numpy method (on cpu)
            t1 = time.time()
            loss_torch_cpu = model()
            times_optimized_torch_cpu.append(time.time() - t1)
            print("torch computation - cpu", times_optimized_torch_cpu[-1])
            print("optimized numpy loss - cpu: ", loss_torch_cpu)

            # 4. Computing the loss with the optimized, numpy method (on gpu)
            model.to('cuda')
            t1 = time.time()
            loss_torch_gpu = model()
            times_optimized_torch_gpu.append(time.time() - t1)
            print("torch computation - cpu", times_optimized_torch_gpu[-1])
            print("optimized numpy loss - gpu: ", loss_torch_gpu)

            losses_old_method.append(loss_np)
            losses_optimized_np.append(loss_np_optimized)
            losses_optimized_torch_cpu.append(loss_torch_cpu.item())
            losses_optimized_torch_gpu.append(loss_torch_gpu.detach().item())

            # todo: uncomment this
            # self.assertAlmostEqual(loss_np, loss_torch, delta=0.1)

            print("---------\n")

        fig, ax = plt.subplots(1, 2, figsize=(15,10))
        line, = ax[0].plot(times_old_method)
        line.set_label('Old Implementation')

        line2, = ax[0].plot(times_optimized_np)
        line2.set_label('New np Implementation')

        line3, = ax[0].plot(times_optimized_torch_cpu)
        line3.set_label('New torch Implementation - cpu')

        line4, = ax[0].plot(times_optimized_torch_gpu)
        line4.set_label('New torch Implementation - gpu')

        ax[0].legend()
        ax[0].set_title("Computation Time")

        line, = ax[1].plot(losses_old_method)
        line.set_label('Old Implementation')

        line2, = ax[1].plot(losses_optimized_np)
        line2.set_label('New np Implementation')

        line3, = ax[1].plot(losses_optimized_torch_cpu)
        line3.set_label('New torch Implementation - cpu')

        line4, = ax[1].plot(losses_optimized_torch_gpu)
        line4.set_label('New torch Implementation - gpu')

        ax[1].legend()
        ax[1].set_title("Losses")

        plt.savefig("C:\\Users\\maria\\OneDrive\\Desktop\\fig.png")
        plt.show()

    def test_occlusion_prevention(self):

        data_path = "test_data\\test_E_penalty\\"

        with open(data_path + "w.pickle", 'rb') as handle:
            w = pickle.load(handle)

        with open(data_path + "d.pickle", 'rb') as handle:
            d = pickle.load(handle)

        with open(data_path + "args.pickle", 'rb') as handle:
            args = pickle.load(handle)

        nr_angles = 2
        transducer_width = 1
        focal_dist = 3.5
        nr_rows = 3
        nr_cols = 3
        nr_planes = 3
        angles = [0, 45]
        dims = (3, 3, 3)
        _, gauss, seen_voxels = args

        #todo:change this!
        rescanning_weights, gauss = occlusion_prevention_preparation(d=d,
                                                                     transducer_width=transducer_width,
                                                                     focal_dist=focal_dist,
                                                                     seen_voxels=seen_voxels,
                                                                     gauss=gauss,
                                                                     nr_angles=nr_angles)

        d_array = np.array([d[key] for key in d.keys()])

        # rearrange weights so to match d arrangement (not needed if weights are randomly initialized)
        w_torch = w.copy()
        w_torch = np.reshape(w_torch, [nr_angles, nr_rows, nr_cols])
        w_torch = np.transpose(w_torch, [1, 2, 0])
        w_flatten = w_torch.flatten()

        sim_penalty_term, seen_penalty_term, cost = E_penalty(w, d, nr_rows, nr_cols, nr_planes, angles, dims, transducer_width, focal_dist, args)

        sim_penalty_term_2, seen_penalty_term_2, cost2 = occlusion_loss(w_flatten, d_array, gauss, rescanning_weights)

        print(sim_penalty_term, "  ", sim_penalty_term_2)
        print(seen_penalty_term, "  ", seen_penalty_term_2)
        print(cost, "  ", cost2)

        occlusion_model = topt.OcclusionPrevention(d_array, gauss, rescanning_weights)
        occlusion_model.weights.data = torch.Tensor(w_flatten)

        loss = occlusion_model()
        print(loss.detach().item())


if __name__ == '__main__':
    unittest.main()
