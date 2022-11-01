import matplotlib.pyplot as plt

def plotPoints(point_list):
    xs = [point.x for point in point_list]
    ys = [point.y for point in point_list]



    plt.scatter(xs, ys)
    plt.show()