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

    for i in range(10):
        val = np.loadtxt("epoch_%i_eval.txt" %i)
        train = np.loadtxt("epoch_%i_train.txt" %i)

        val_avg.append(sum(val)/len(val))
        train_avg.append(sum(train)/len(train))

    plt.plot(train_avg)
    plt.plot(val_avg)
    plt.savefig("vae.png")
