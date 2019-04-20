import matplotlib.pyplot as plt


def show_predictions(test_set, predictions):
    print(test_set)
    print(predictions)
    years = [1983, 1984, 1985, 1986, 1987]
    total_populations = [8939007, 8954518, 8960387, 8956741, 8943721]

    plt.plot(years, total_populations)
    plt.title("Year vs Population in Bulgaria")
    plt.xlabel("Year")
    plt.ylabel("Total Population")
    plt.show()
