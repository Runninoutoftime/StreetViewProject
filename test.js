import { readFile, readFileSync } from 'fs'
import pkg from 'papaparse'

const {parse} = pkg
const csv_file = readFileSync('output.csv', { encoding: 'utf-8'})

let csv_results = parse(csv_file)

console.log(csv_results['data'][0][1])
console.log(csv_results['data'].length)


