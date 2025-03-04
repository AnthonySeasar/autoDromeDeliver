import numpy as np
import matplotlib.pyplot as plt


def ripley_k(data, r):
    n = len(data)
    k = np.zeros(len(r))
    for i in range(n):
        for j, radius in enumerate(r):
            in_circle = np.linalg.norm(data[i] - data, axis=1) <= radius
            k[j] += np.sum(in_circle)
    return k / n


r = np.linspace(0, 35, 100)

hospitals = np.load('data_hospital.npy')
schools = np.load('data_school.npy')
fitting_rooms = np.load('data_fitting.npy')
malls = np.load('data_mall.npy')
combined = np.load('data_combined.npy')

# lat and lon of california
california_bounds = [32.5, 42, -124.5, -114]

# amount of sub-area
num_sub_areas_x = 30
num_sub_areas_y = 60

# calculate the size of sub areas
lat_step = (california_bounds[1] - california_bounds[0]) / num_sub_areas_y
lon_step = (california_bounds[3] - california_bounds[2]) / num_sub_areas_x


def calculate_sub_area_k(data, lat_min, lat_max, lon_min, lon_max, r):
    sub_area_data = data[(data[:, 0] >= lat_min) & (data[:, 0] <= lat_max) &
                         (data[:, 1] >= lon_min) & (data[:, 1] <= lon_max)]
    return ripley_k(sub_area_data, r)


k_matrix = np.zeros((num_sub_areas_y, num_sub_areas_x, 4, len(r)))

for i in range(num_sub_areas_y):
    for j in range(num_sub_areas_x):
        lat_min = california_bounds[0] + i * lat_step
        lat_max = lat_min + lat_step
        lon_min = california_bounds[2] + j * lon_step
        lon_max = lon_min + lon_step

        k_matrix[i, j, 0, :] = calculate_sub_area_k(hospitals, lat_min, lat_max, lon_min, lon_max, r)
        k_matrix[i, j, 1, :] = calculate_sub_area_k(schools, lat_min, lat_max, lon_min, lon_max, r)
        k_matrix[i, j, 2, :] = calculate_sub_area_k(fitting_rooms, lat_min, lat_max, lon_min, lon_max, r)
        k_matrix[i, j, 3, :] = calculate_sub_area_k(malls, lat_min, lat_max, lon_min, lon_max, r)

# generate the result and store in k_matrix
np.save('k_matrix.npy', k_matrix)

if __name__ == '__main__':

    radius_index = 50
    plt.figure(figsize=(10, 6))

    for func_type, label in enumerate(['Hospitals', 'Schools', 'Fitting Rooms', 'Malls']):
        plt.imshow(k_matrix[:, :, func_type, radius_index],
                   extent=[california_bounds[2], california_bounds[3], california_bounds[0], california_bounds[1]],
                   origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(label=f"{label} K Value")
        plt.title(f"{label} K Value Distribution (Radius {r[radius_index]})")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
