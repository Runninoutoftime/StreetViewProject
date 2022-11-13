// require('dotenv').config()

// import { readFile, readFileSync } from 'fs'
// import pkg from 'papaparse'
// const {parse} = pkg

// // const csv_file = readFileSync('output.csv', { encoding: 'utf-8'})

// // let csv_results = parse(csv_file)

// // console.log(csv_results['data'][0][1])
// // console.log(csv_results['data'].length)


// class image_collection {
//     constructor(input_csv_file, output_folder) {
//         this.input_csv_file = input_csv_file // The path to the csv file containg the coordinates for streetview image collection
//         this.output_folder = output_folder // The path to the folder where the images will be saved
//         this.coordinates = this.get_coordinates()
//     }

//     // --------------------- Data Formatting and Processing Functions ---------------------

//     // From a csv file containing an array of [x, y] coordinates, return each [x, y] pair as an object in a 2d array
//     get_coordinates() {

//         // Read the csv file
//         let csv_data = readFileSync(this.input_csv_file, { encoding: 'utf-8' })
//         let csv_results = parse(csv_data)
//         let coordinates = [[]]

//         // For each coordinate pair, add it to the coordinates array
//         for (let i = 0; i < csv_results['data'].length; i++) {
//             coordinates[i] = csv_results['data'][i]
//         }

//         // Last value is null, so remove it
//         coordinates.pop(coordinates.length)

//         // Convert the coordinates to floats
//         for (let i = 0; i < coordinates.length; i++) {
//             coordinates[i][0] = parseFloat(coordinates[i][0])
//             coordinates[i][1] = parseFloat(coordinates[i][1])
//         }

//         return coordinates
//     }

//     // Helper function for get_images. From a single [x, y] coordinate, return a streetview image
//     get_image() {
//         let maps;
//         let basic_request = 'https://maps.googleapis.com/maps/api/streetview?'

//         for (coordinate in this.coordinates) {

//             // Parameters for the requests to the Google Maps API
//             let x = coordinate[0]
//             let y = coordinate[1]
//             let param_size = 'size=640x640'
//             let param_location = 'location=' + x + ',' + y
//             let param_key = 'key=' + process.env.GOOGLE_API_KEY
//             let param_source = 'source=outdoor'


//             let param_fov = 'fov=120'
//             let param_heading_1 = 'heading=0'
//             let param_heading_2 = 'heading=120'
//             let param_heading_3 = 'heading=240'
//             // let param_signature = pls no https://console.cloud.google.com/google/maps-apis/credentials?project=streetview-367300, https://developers.google.com/maps/documentation/streetview/digital-signature#node-js

//             // Construct the requests
//             let request1 = basic_request + param_size + '&' + param_location + '&' + param_fov + '&' + param_heading_1 + '&' + param_source + '&' + param_key
//             let request2 = basic_request + param_size + '&' + param_location + '&' + param_fov + '&' + param_heading_2 + '&' + param_source + '&' + param_key
//             let request3 = basic_request + param_size + '&' + param_location + '&' + param_fov + '&' + param_heading_3 + '&' + param_source + '&' + param_key


//         }

//         //https://stackoverflow.com/a/53264945
//     }

//     // From a 2d array of [x, y] coordinates, return an array of streetview images
//     get_images() {
//     }

//     // --------------------- Google Maps Functions ---------------------
// }

// let image_collection_obj = new image_collection('output.csv', 'images')
// console.log(image_collection_obj.get_coordinates())
