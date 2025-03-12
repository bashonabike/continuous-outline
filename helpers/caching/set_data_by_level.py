import numpy as np
import pandas as pd


def set_level_1_data(dataframes:dict, outer_contours:list, outer_edges:np.array, inner_contours:list,
                     inner_edges:np.array, focus_masks:list[np.array]):
    """
    Set the level 1 data for the contours and edges.
    :param outer_contours: dict of dataframes to set
    :param outer_contours: List of contours for the outer contour.
    :param outer_edges: List of edges for the outer contour.
    :param inner_contours: List of contours for the inner contour.
    :param inner_edges: List of edges for the inner contour.
    :param focus_masks: List of focus masks for the contours.
    :return: None
    """
    #Set contours headers
    contour_idx_offset = 0
    for outer, contours in zip([True, False], [outer_contours, inner_contours]):
        num_rows = len(contours)
        contour_values = pd.Series(range(1 + contour_idx_offset, num_rows + contour_idx_offset + 1))
        is_outer_values = pd.Series([1] if outer else [0] * num_rows)
        is_inner_values = pd.Series([0] if outer else [1]  * num_rows)
        partial_df = pd.DataFrame({'contour': contour_values, 'is_outer': is_outer_values, 'is_inner': is_inner_values})
        partial_df = partial_df[dataframes["Contours"].columns]
        dataframes["Contours"] = pd.concat([dataframes["Contours"], partial_df], ignore_index=True)
        contour_idx_offset = num_rows

    #Set contours details
    for i, contour in enumerate(outer_contours + inner_contours):
        contour = pd.Series([i + 1] * len(contour))
        points = pd.Series(range(1, len(contour) + 1))
        contour_nd = np.array(contour)
        y_values = pd.Series(contour_nd[:, 0])
        x_values = pd.Series(contour_nd[:, 1])
        partial_df = pd.DataFrame({'point_num': points, 'x': x_values, 'y': y_values, 'contour': contour})
        partial_df = partial_df[dataframes["Contour"].columns]
        dataframes["Contour"] = pd.concat([dataframes["Contour"], partial_df], ignore_index=True)

    #Set edges pixel maps
    if outer_edges.shape != inner_edges.shape:
        raise ValueError("Outer and inner edge arrays must have the same shape.")

    y_coords, x_coords = np.indices(outer_edges.shape)  # Generate y and x coordinates using indices

    # Flatten the arrays
    y_flat = y_coords.flatten()
    x_flat = x_coords.flatten()
    outer_flat = outer_edges.flatten()
    inner_flat = inner_edges.flatten()

    # Create is_inner and is_outer columns
    is_inner = np.where(inner_flat != 0, 1, 0)
    is_outer = np.where(outer_flat != 0, 1, 0)

    # Create the DataFrame
    partial_df = pd.DataFrame({
        'y': y_flat,
        'x': x_flat,
        'is_inner': is_inner,
        'is_outer': is_outer,
        'inner_edge_num': inner_flat,
        'outer_edge_num': outer_flat
    })[dataframes["EdgesPixelMap"].columns]

    dataframes["EdgesPixelMap"] = partial_df

    #Set focus masks header
    focus_masks = pd.Series(range(0, len(focus_masks)))
    partial_df = pd.DataFrame({'focus_mask': focus_masks})
    dataframes["FocusMasks"] = partial_df

    #Set focus masks details
    for i, focus_mask in enumerate(focus_masks):
        y_coords, x_coords = np.where(focus_mask)

        # Create the DataFrame
        partial_df = pd.DataFrame({
            'y': y_coords,
            'x': x_coords,
            'focus_mask': i
        })[dataframes["FocusMask"].columns]

        dataframes["FocusMask"] = pd.concat([dataframes["FocusMask"], partial_df], ignore_index=True)




