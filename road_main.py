from CalcIntersections import CalcIntersections

ga = CalcIntersections(13)

counties = ["Fulton_county", "Bibb_county", "DeKalb_county", "Clarke_county", "Baker_county"]

county_geoms = ga.getCounties()
roads = ga.getRoads(1,322,2)
ga.sortRoads(counties)
ga.findIntersections(counties)
ga.samplePoints(counties)
ga.heatMap(counties)