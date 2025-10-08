from tools import *
from EnsambleKalmanFilter import *
import matplotlib.pyplot as plt


matrix, headers = readFile("/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/SignalAnalysis/activities/EnKL/data/User1_Pre2.csv")

xP = EnKL(matrix, 128)

print("Xp shape: ", xP.shape)


col = 2
limit = 3072
test_X1 = xP[:limit, 2]
print(test_X1.shape)

print(test_X1)
plt.plot(test_X1)
#plt.plot(matrix[:limit, 2])
plt.show()

