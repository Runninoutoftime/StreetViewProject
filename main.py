
from StreetView import StreetView
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Create a StreetView object
api_key = os.getenv('GOOGLE_API_KEY') + '-U' # Because it is unrestricted and .env can't have a dash?
streetview_folder = 'streetview_images'
panorama_folder = 'panorama_images'

streetview_obj = StreetView(api_key, streetview_folder, panorama_folder)

print(api_key)

list_of_coordinates = [[33.933208, -83.382564], [33.939153, -83.386615], [33.934710, -83.370456], [33.956412, -83.381247]]

for coordinate in list_of_coordinates:
    streetview_obj.get_images(coordinate[0], coordinate[1])
    coordinate_images_path = streetview_obj.streetview_folder + '/' + str(coordinate[0]).replace('.', '') + '_' + str(coordinate[1]).replace('.', '')
    # print(coordinate_images_path)
    streetview_obj.stitch_images(coordinate_images_path, coordinate[0], coordinate[1])
