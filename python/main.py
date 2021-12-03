from functions.optimizers import AcquisitionGeometryParams, OptimizersOptions, random_scan, \
    perpendicular_occlusion_prevention_optimizer, volume_coverage_occlusion_prevention_optimizer, \
    random_occlusion_prevention_optimizer, perpendicular_scan, volume_coverage_optimizer

from functions.cost_functions import occlusion_loss_preparation, \
    occlusion_loss, post_process_weights, coverage_loss_preparation, coverage_loss

if __name__ == '__main__':
    nr_rows = 8
    nr_cols = 10
    nr_planes = 6
    angles = [-35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]  # Write the angles from smallest to
    # largest
    transducer_width = 1
    occ_threshold = 0.4  # The threshold at which a confidence voxel counts as occluded
    focal_dist = 2.5
    visualize_scans = False
    model_name = "two_beams_nonparallel"
    material_type = "soft_tissue"  # "soft_tissue"
    dims = (nr_rows, nr_cols, nr_planes)
    show_occluded_voxels_after_simulation = True  # If you set this to true, you have to close the figure to continue the pipeline.

    acquisition_params = AcquisitionGeometryParams(angles=angles,
                                                   nr_rows=nr_rows,
                                                   nr_cols=nr_cols,
                                                   nr_planes=nr_planes,
                                                   focal_dist=focal_dist,
                                                   transducer_width=transducer_width,
                                                   dims=dims,
                                                   model_name=model_name,
                                                   occ_threshold=occ_threshold,
                                                   material_type=material_type)

    volume_coverage_optimizer_param = OptimizersOptions(pre_processing_function=coverage_loss_preparation,
                                                        post_processing_function=post_process_weights,
                                                        optimization_function=coverage_loss)
    # The optimized version.
    occlusion_prevention_optimizer_param = OptimizersOptions(pre_processing_function=occlusion_loss_preparation,
                                                             post_processing_function=post_process_weights,
                                                             optimization_function=occlusion_loss)

    ################### SINGLE SCANS #########################
    # random_scan(acquisition_params, show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation,
    #             visualize_scans=visualize_scans)

    perpendicular_scan(acquisition_params, synthetic=False,
                       show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation,
                       visualize_scans=visualize_scans)

    # volume_coverage_optimizer(acquisition_params=acquisition_params,
    #                           preparation_fun=volume_coverage_optimizer_param.pre_processing_function,
    #                           post_processing_function=volume_coverage_optimizer_param.post_processing_function,
    #                           optimization_function=volume_coverage_optimizer_param.optimization_function,
    #                           optimization_method=volume_coverage_optimizer_param.method,
    #                           optimization_framework=volume_coverage_optimizer_param.framework,
    #                           visualize_scans=visualize_scans,
    #                           show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation)

    ############### SCANS WITH TWO ITERATIONS ###############################

    # volume_coverage_occlusion_prevention_optimizer(acquisition_params=acquisition_params,
    #                                                volume_coverage_optimizer_options=volume_coverage_optimizer_param,
    #                                                occlusion_optimizer_options=occlusion_prevention_optimizer_param,
    #                                                visualize_scans=visualize_scans,
    #                                                show_occluded_voxels_after_simulation=True)

    # perpendicular_occlusion_prevention_optimizer(acquisition_params=acquisition_params,
    #                                              occlusion_optimizer_options=occlusion_prevention_optimizer_param,
    #                                              visualize_scans=visualize_scans,
    #                                              show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation)

    # random_occlusion_prevention_optimizer(acquisition_params=acquisition_params,
    #                                       occlusion_optimizer_options=occlusion_prevention_optimizer_param,
    #                                       visualize_scans=visualize_scans,
    #                                       show_occluded_voxels_after_simulation=show_occluded_voxels_after_simulation)
