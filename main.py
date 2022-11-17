
from StreetView import StreetView
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Create a StreetView object
api_key = os.getenv('GOOGLE_API_KEY') + '-U' # Because it is unrestricted and .env can't have a dash?
streetview_folder = 'streetview_images'
panorama_folder = 'panorama_images'
coordinate_file = 'test_data.csv'

streetview_obj = StreetView(api_key, streetview_folder, panorama_folder, coordinate_file)
# This one function will get all the images and stitch them together for a list of coordinates
streetview_obj.process_coordinates()

# print(api_key)

