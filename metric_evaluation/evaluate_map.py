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
    plt.hist([i for i in lengths if i > 6.9], bins=20)
    plt.yscale('log')
    plt.show()
    median = np.median(lengths)
    var = np.var(lengths)
    std = np.std(lengths)
    print("Maximum: {:.2E}".format(max(lengths)))
    print("Variance: {:.2E}".format(var))
    print("Std dev: {:.2E}".format(std))
    print("Median: {} ".format(median))
    return (sum(i > 6.9 for i in lengths), sum(i > 50 for i in lengths))


def plot_pointcloud(points, ax, color, black=False, no_axes=False):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    x_median = [np.median(x)]
    y_median = [np.median(y)]
    z_median = [np.median(z)]
    if ax is None:
        ax = plt.axes(projection="3d")
        ax.set_facecolor("black" if black else "white")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        # ax.scatter([0], [0], [0], ".", color="white", s=10) #origin
        plt.axis("off" if no_axes else "on")
    x_o = []
    y_o = []
    z_o = []
    for x_i, y_i, z_i in zip(x, y, z):
        if vec_length([x_i, y_i, z_i]) > 6.9:
            x_o.append(x_i)
            y_o.append(y_i)
            z_o.append(z_i)
    
    ax.scatter(x_o, y_o, z_o, ".", color=color, alpha=0.75,  s=5.5)
    ax.scatter(x, y, z, ".", color=color, alpha=0.25,  s=0.5)
    ax.scatter(x_median, y_median, z_median, "o", color=color, s=30)

    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", dest="input", nargs="*", help="input csv file with mappoints. x,y,z per line", type=str, required=True)
    parser.add_argument("-b", action="store_true", help="whether to plot on black background")
    parser.add_argument("--no-axes", action="store_true", help="whether to turn axes off in plot")
    parser.add_argument("-c", type=str, nargs="*", help="The color(s) to use for the maps")

    args = parser.parse_args()

    if args.c:
        COLORS = args.c
    all_points = []
    fig = plt.figure()
    ax = None
    for i, f in enumerate(args.input):
        print("Reading {}".format(f))
        all_points.append(read_file(f))
        print("Num points: {}".format(len(all_points[-1])))
        outliers, extreme_outliers = num_outliers(all_points[-1])
        print("Num outliers: {}".format(outliers))
        print("Num extreme outliers: {}".format(extreme_outliers))
        ax = plot_pointcloud(
            all_points[-1], ax=ax, color=COLORS[i % len(COLORS)], black=args.b, no_axes=args.no_axes)
    plt.tight_layout()
    plt.draw()
    plt.show()
