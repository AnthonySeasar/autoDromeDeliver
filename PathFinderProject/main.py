import requests
import folium
import numpy as np

# use data on opensteetmap
overpass_url = "http://overpass-api.de/api/interpreter"

# hospitals
overpass_query = """
[out:json];
area["ISO3166-2"="US-CA"];
(node["amenity"="hospital"](area);
 way["amenity"="hospital"](area);
 rel["amenity"="hospital"](area);
);
out center;
"""
response = requests.get(overpass_url,
                        params={'data': overpass_query})
data_hospital = response.json()

# map hospitals
m_h = folium.Map(location=[37.6, -120.9], zoom_start=6)
# Loop through the data and add each data point to the map
for element in data_hospital['elements']:
    if element['type'] == 'node':
        lat = element['lat']
        lon = element['lon']
        name = element['tags']['name'] if 'name' in element['tags'] else 'No Name'
        folium.Marker(location=[lat, lon],
                      popup=name,
                      icon=folium.Icon(color="green", icon="ok-sign"),
                      ).add_to(m_h)

# high school, college, university
overpass_query = """
[out:json];
area["ISO3166-2"="US-CA"];
(
  node["amenity"="school"]["name"~"High School"](area);
  way["amenity"="school"]["name"~"High School"](area);
  rel["amenity"="school"]["name"~"High School"](area);

  node["amenity"="college"](area);
  way["amenity"="college"](area);
  rel["amenity"="college"](area);

  node["amenity"="university"](area);
  way["amenity"="university"](area);
  rel["amenity"="university"](area);
);
out center;
"""
response = requests.get(overpass_url,
                        params={'data': overpass_query})
data_school = response.json()

# map schools
m_s = folium.Map(location=[37.6, -120.9], zoom_start=6)
# Loop through the data and add each data point to the map
for element in data_school['elements']:
    if element['type'] == 'node':
        lat = element['lat']
        lon = element['lon']
        name = element['tags']['name'] if 'name' in element['tags'] else 'No Name'
        folium.Marker(location=[lat, lon],
                      popup=name,
                      icon=folium.Icon(color="green", icon="ok-sign"),
                      ).add_to(m_s)
# Show the map
m_s

# malls
overpass_query = """
[out:json];
area["ISO3166-2"="US-CA"];
(
  node["shop"="mall"](area);
  way["shop"="mall"](area);
  rel["shop"="mall"](area);
);
out center;
"""
response = requests.get(overpass_url,
                        params={'data': overpass_query})
data_mall = response.json()

# map malls
m_m = folium.Map(location=[37.6, -120.9], zoom_start=6)
# Loop through the data and add each data point to the map
for element in data_mall['elements']:
    if element['type'] == 'node':
        lat = element['lat']
        lon = element['lon']
        name = element['tags']['name'] if 'name' in element['tags'] else 'No Name'
        folium.Marker(location=[lat, lon],
                      popup=name,
                      icon=folium.Icon(color="green", icon="ok-sign"),
                      ).add_to(m_m)
# Show the map


# fitting_room
overpass_query = """
[out:json];
area["ISO3166-2"="US-CA"];
(node["shop"="clothes"](area);
 way["shop"="clothes"](area);
 rel["shop"="clothes"](area);
);
out center;
"""
response = requests.get(overpass_url,
                        params={'data': overpass_query})
data_fitting = response.json()

# map fitting room
m_f = folium.Map(location=[37.6, -120.9], zoom_start=6)
# Loop through the data and add each data point to the map
for element in data_fitting['elements']:
    if element['type'] == 'node':
        lat = element['lat']
        lon = element['lon']
        name = element['tags']['name'] if 'name' in element['tags'] else 'No Name'
        folium.Marker(location=[lat, lon],
                      popup=name,
                      icon=folium.Icon(color="green", icon="ok-sign"),
                      ).add_to(m_f)
# Show the map

# Create a map for all
m = folium.Map(location=[36.7783, -119.4179], zoom_start=6)


# Function to add data to the map
def add_data_to_map(data, color, icon):
    for element in data['elements']:
        if element['type'] == 'node':
            lat = element['lat']
            lon = element['lon']
            folium.Marker([lat, lon], icon=folium.Icon(color=color, icon=icon)).add_to(m)


# Add schools, hospitals, malls, fitting rooms to the map
add_data_to_map(data_school, 'blue', 'graduation-cap')
add_data_to_map(data_hospital, 'red', 'plus')
add_data_to_map(data_mall, 'green', 'shopping-cart')
add_data_to_map(data_fitting, 'purple', 'asterisk')


def extract_coordinates(json_data):
    elements = json_data['elements']
    coordinates = np.array([[element['lat'], element['lon']] for element in elements])
    return coordinates


# 将数据保存为.npy文件的函数
def save_data(data, filename):
    np.save(filename, data)


hospital_coordinates = extract_coordinates(data_hospital)
save_data(hospital_coordinates, "data_hospital.npy")

school_coordinates = extract_coordinates(data_school)
save_data(school_coordinates, "data_school.npy")

mall_coordinates = extract_coordinates(data_mall)
save_data(mall_coordinates, "data_mall.npy")

fitting_coordinates = extract_coordinates(data_fitting)
save_data(fitting_coordinates, "data_fitting.npy")

data_school_1 = np.load("data_school.npy")
data_hospital_1 = np.load("data_hospital.npy")
data_mall_1 = np.load("data_mall.npy")
data_fitting_1 = np.load("data_fitting.npy")

# 合并数据集
data_combined = np.concatenate([data_school_1, data_hospital_1, data_mall_1, data_fitting_1], axis=0)

# 保存组合的数据备用
save_data(data_combined, "data_combined.npy")

if __name__ == '__main__':
    m.save("map_all_locations.html")