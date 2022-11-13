import google_streetview.api
import google_streetview.helpers

# Define parameters for street view api
params = {
    'size': '640x640', # max 640x640 pixels
    'location': '33.933260, -83.382731',
    'heading': '0;90;180;270',
    'pitch': '-0.76',
    'fov': '90',
    'key': 'key_here',
    }

# Create a results object
# results = google_streetview.api.results(params)

api_list = google_streetview.helpers.api_list(params)

results = google_streetview.api.results(api_list)

results.preview()

results.download_links('images')

results.save_metadata('metadata')

# google_streetview.api.

# results.preview()

# Download images to directory 'downloads'
# results.download_links('downloads')


