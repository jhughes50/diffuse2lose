"""
    Jason Hughes
    November 2024

    Logging Object for losses
"""
from typing import List

class Logger:

    def __init__(self, path : str = "./logs/") -> None:
        self.path_ = path
        print(self.path_)
        with open(self.path_+"epoch_average.txt", "x") as f:
            pass

    def log_list(self, losses : List, epoch : int, train_val : str) -> None:
        file_name = self.path_+"epoch_%i_%s.txt" %(epoch, train_val)
        with open(file_name, "w") as file:
            for item in losses:
                file.write(f"{item}\n")

    def log_float(self, loss : float) -> None:
        with open(self.path_+"epoch_average.txt", "w") as f:
            f.write(f"{loss}\n")

