import os
import config
from src.cascade_model import make_list_of_dirs, prepare_data_for_plot, plot_all_data

degree_field = 'degree_variance'
degree_title = 'Degree variance'

VK_DATA_FOLDER = os.path.join(config.DATA_DIR, 'cascades')
THETA = [1.6, 1.8, 2, 2.2]

if __name__ == '__main__':
    field = degree_field
    title = degree_title
    dirs_list = make_list_of_dirs(THETA)
    prepared_data = prepare_data_for_plot(dirs_list, VK_DATA_FOLDER, field)
    plot_all_data(prepared_data, title)
