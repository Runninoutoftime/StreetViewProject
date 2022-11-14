import google_streetview.api
import google_streetview.helpers
from dotenv import load_dotenv
from PIL import Image
from imutils import paths
import numpy as np
import cv2
import os

class StreetView:

    def __init__(self, api_key, streetview_folder, panorama_folder):
        self.api_key = api_key
        self.streetview_folder = streetview_folder
        self.panorama_folder = panorama_folder

    def get_images(self, lat, lon):
        """Given a latitude and longitude, get the street view images and save them to a folder

        Args:
            lat (float): Latitude of the location
            lon (float): Longitude of the location
        """
        
        # Loads the API key from the .env file
        load_dotenv()

    # Define parameters for street view api
        params = {
            'size': '640x640', # max 640x640 pixels
            'location': str(lat) + ', ' + str(lon),
            'heading': '0;45;90;135;180;225;270;315;', # gets 8 images from every 45 degrees
            'pitch': '0',
            'fov': '95', # May need to be adjusted 
            'key': str(self.api_key),
            }

        api_list = google_streetview.helpers.api_list(params)

        results = google_streetview.api.results(api_list)

        # To preview metadata of results
        # results.preview()

        # For each result in results, check if date is in range if we want to filter by date

        # Download images to directory with same name as location
        results.download_links(self.streetview_folder + '/' + str(lat).replace('.' ,'') + '_' + str(lon).replace('.', ''))

        # Save metadata of images to a file with same name as location
        results.save_metadata(self.streetview_folder + '/' + str(lat).replace('.' ,'') + '_' + str(lon).replace('.', '') + '/metadata')


    # Takes a folder of images and stitches them together
    def stitch_images(self, input_folder, lat, lon): 
        """Takes a folder of images and stitches them together into a panorama

        Args:
            input_folder (string): The folder containing images to stitch
            output_folder (string): The folder to save the stitched image to
            lat (float): Latitude of the location
            lon (float): Longitude of the location
        """

        imagePaths = sorted(list(paths.list_images(input_folder)))
        images = []

        # print(imagePaths)
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            images.append(image)

        stitcher = cv2.Stitcher_create()
        (status, stitched) = stitcher.stitch(images)

        if status == 0:

            # This set of code below is to remove the black borders from the panorama and make it rectangular
            # https://pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/
            stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

            # Find contours in thresholded image, then grab the largest one
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            c = max(cnts, key=cv2.contourArea)

            # allocate memory for the mask which will contain the
            # rectangular bounding box of the stitched image region
            mask = np.zeros(thresh.shape, dtype="uint8")
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            # create two copies of the mask: one to serve as our actual
            # minimum rectangular region and another to serve as a counter
            # for how many pixels need to be removed to form the minimum
            # rectangular region
            minRect = mask.copy()
            sub = mask.copy()

            # keep looping until there are no non-zero pixels left in the
            # subtracted image
            while cv2.countNonZero(sub) > 0:
                # erode the minimum rectangular mask and then subtract
                # the thresholded image from the minimum rectangular mask
                # so we can count if there are any non-zero pixels left
                minRect = cv2.erode(minRect, None)
                sub = cv2.subtract(minRect, thresh)

            # find contours in the minimum rectangular mask and then extract
            # the bounding box (x, y)-coordinates
            cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            c = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)

            # use the bounding box coordinates to extract the our final
            # stitched image
            stitched = stitched[y:y + h, x:x + w]

            output_path = self.panorama_folder + '/' + str(lat).replace('.' ,'') + '_' + str(lon).replace('.', '') + '.jpg'
            cv2.imwrite(output_path, stitched)

            # cv2.imshow("Stitched", stitched)
            # cv2.waitKey('q')
        else:
            print("error with stitching")
