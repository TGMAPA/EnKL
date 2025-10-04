import numpy as np

# Lectura de archivo de entrada
def readFile(path):
    file = open(path, "r")
    matrix = []
    i = 0
    for line in file.readlines():
        row = line.split(",")
        if i != 0:
            row = [float(item) for item in row]
            matrix.append(row)
        else:
            headers = row
            headers[-1] = headers[-1][:-1]
            i+=1

    return np.array(matrix), headers

