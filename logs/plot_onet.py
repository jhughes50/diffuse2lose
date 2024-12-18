"""
    Jason Hughes and Tenzi Zhuoga
    Decemeber 2024

    Plot the avg epoch loss for vae

"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    val_avg = list()
    train_avg = list()

    for i in range(40):
        val = np.loadtxt("./onet/epoch_%i_val_onet.txt" %i)
        train = np.loadtxt("./onet/epoch_%i_train_onet.txt" %i)

        val_avg.append(sum(val)/len(val))
        train_avg.append(sum(train)/len(train))

    plt.plot(train_avg, label="train")
    plt.plot(val_avg, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Occupancy Net Average Loss per Epoch")
    plt.legend()
    plt.savefig("onet.png")
