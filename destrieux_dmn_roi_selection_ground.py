import numpy as np
from nilearn import surface
from nilearn import datasets
from nilearn import plotting


def euclidean(x, y):
    """Standard euclidean distance.
    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


def destrieux_roi_selection(dmn_coord, destrieux_labels, destrieux_coor, ridus):
    '''
    --- calculated the DMN ROIs according to the DMN coordinates
    :param dmn_coord: (M, 3)
    :param destrieux_labels: (N, )
    :param destrieux_coor:  (N, 3)
    :param ridus: (1, )
    :return:
    '''
    selected_rois_names = []
    selected_coords = []
    selected_rois_index = []

    for destrieux_index in range(destrieux_labels.shape[0]):

        for dmn_index in range(len(dmn_coord)):
            distance = euclidean(np.array(destrieux_coor[destrieux_index]), np.array(dmn_coord[dmn_index]))
            if distance < ridus:
                selected_rois_index.append(destrieux_index)
                selected_rois_names.append(destrieux_labels[destrieux_index])
                selected_coords.append(destrieux_coor[destrieux_index])
                break
    return selected_rois_index, selected_coords, selected_rois_names


if __name__ == "__main__":

    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    parcellation = destrieux_atlas['map_left']
    fsaverage = datasets.fetch_surf_fsaverage()

    atlas = destrieux_atlas
    coordinates = []
    labels = destrieux_atlas['labels']
    # print("labels.shape: ", len(labels))
    # print("labels: ", labels)

    selected_labels = []
    for hemi in ['left', 'right']:
        vert = destrieux_atlas['map_%s' % hemi]
        rr, _ = surface.load_surf_mesh(fsaverage['pial_%s' % hemi])
        for k, label in enumerate(labels):
            if "Unknown" not in str(label):
                if "Medial_wall" not in str(label):
                    selected_labels.append(label)
                    coordinates.append(np.mean(rr[vert == k], axis=0))

    selected_coordinates_array = np.array(coordinates)  # 3D coordinates of parcels
    print("selected_coordinates_array.shape: ", selected_coordinates_array.shape)

    selected_labels_array = np.array(selected_labels)
    print("selected_labels_array.shape: ", selected_labels_array.shape)

    # default mode network MNI coordinates
    dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]

    selected_rois_index_ret, selected_coords_ret, selected_rois_names_ret = destrieux_roi_selection(dmn_coords,
                                                                                                    selected_labels_array,
                                                                                                    selected_coordinates_array,
                                                                                                    39)

    print("len(selected_rois_index_ret): ", len(selected_rois_index_ret))
    print("selected_rois_index_ret: ", selected_rois_index_ret)
    print("selected_rois_names_ret: ", selected_rois_names_ret)

    # to make it easy to read
    index_list = [x for x in range(len(selected_rois_index_ret))]
    # print("index_list: ", index_list)

    selected_dmn_rois_index_list_unc_matrix = [x for x in zip(index_list, selected_rois_index_ret)]
    print("selected_dmn_rois_index_list_unc_matrix: ", selected_dmn_rois_index_list_unc_matrix)

    selected_dmn_rois_labels_list = [x for x in zip(index_list, selected_rois_names_ret)]
    print("selected_dmn_rois_labels_list: ", selected_dmn_rois_labels_list)

    fake_selected_destrieux = np.zeros((len(selected_coords_ret), len(selected_coords_ret)))
    plotting.plot_connectome(fake_selected_destrieux, selected_coords_ret,
                             output_file="data/destrieux_selected_coords_ret.pdf",
                             title="DMN destrieux_selected_coords")
