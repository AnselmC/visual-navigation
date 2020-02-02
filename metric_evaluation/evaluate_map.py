import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


COLORS = ["red", "lightgreen", "red"]


def read_file(fname):
    with open(fname, "r") as f:
        data = f.read()
        data = data.replace("\n", ",").replace("x,", "").replace(
            "y,", "").replace("z,", "").replace(" ", "").split(",")[:-1]

    points = [(float(x), float(y), float(z))
              for x, y, z in zip(data[::3], data[1::3], data[2::3])]
    return points


def vec_length(point):
    return np.sqrt(sum([p**2 for p in point]))


def num_outliers(points, median_factor=4):
    count = 0
    lengths = []
    for point in points:
        lengths.append(vec_length(point))
    median = np.median(lengths)
    var = np.var(lengths)
    std = np.std(lengths)
    print("Variance: {}".format(var))
    print("Std dev: {}".format(std))
    print("Median: {} ".format(median))
    return len(np.where(lengths > median_factor * median)[0])


def plot_pointcloud(points, ax, color):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    x_median = [np.median(x)]
    y_median = [np.median(y)]
    z_median = [np.median(z)]
    if ax is None:
        ax = plt.axes(projection="3d")
        ax.set_facecolor("black")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        # ax.scatter([0], [0], [0], ".", color="white", s=10) #origin
        plt.axis("off")
    ax.scatter(x, y, z, ".", color=color, alpha=0.5,  s=0.5)
    ax.scatter(x_median, y_median, z_median, "o", color=color, s=30)

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", dest="input", nargs="*", help="input csv file with mappoints. x,y,z per line", type=str, required=True)
    parser.add_argument("-o", dest="output",
                        help="dir where to save plots", type=str)

    args = parser.parse_args()

    all_points = []
    fig = plt.figure()
    fig.set_facecolor("black")
    ax = None
    for i, f in enumerate(args.input):
        print("Reading {}".format(f))
        all_points.append(read_file(f))
        print("Num outliers: {}".format(num_outliers(all_points[-1])))
        ax = plot_pointcloud(
            all_points[-1], ax=ax, color=COLORS[i % len(COLORS)])
    plt.tight_layout()
    plt.draw()
    plt.show()
