import os
import sys

SEED = [42, 85, 291, 572, 481, 8412, 8512, 8002, 1723, 4715]
SIZE = [10, 64]
NP = [8, 44]
C = [2, 4]
BTYPE = range(0, 10)

def intToStr(x, num):
	ret = ""
	while(x):
		ret += str(x % 10)
		x /= 10
	while(len(ret) < num):
		ret += "0"

	return ret[::-1]

for e in range(1, 4):
	for i in xrange(2):
		for j in xrange(10):
			command = "mpirun -np {} ./matrixmul -f ../exported_tests/sparse05_{}_{} -s {} -c {} -e {} -v -i > out 2> /dev/null".format(
					NP[i], intToStr(SIZE[i], 5), intToStr(BTYPE[j], 3), SEED[j], C[i], e)
			if(len(sys.argv) == 1):
				os.system(command)
			print(command)
			command2 = "./sprawdzaczka/compare ../exported_tests/result_{}_{}_{}_{} out".format(
				e, intToStr(SIZE[i], 5), intToStr(BTYPE[j], 3), intToStr(SEED[j], 5)
			)
			if(len(sys.argv) == 1):
				os.system(command2)
			print(command2)
