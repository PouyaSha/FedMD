# import pickle

# import numpy as np

# file = open("result_FEMNIST_imbalanced/col_performance.pkl","rb")

# data = pickle.load(file)

# plot_data = []

# for i in range(10):

#         plot_data.append(np.array(data[i]))

# print(np.ndarray.tolist(np.array(plot_data)))



import pickle

import numpy as np

from matplotlib import pyplot as plt




file = open("result_FEMNIST_imbalanced/col_performance.pkl","rb")

data = pickle.load(file)

fig = plt.figure()

ax = plt.subplot( 111 )

ax.minorticks_on()

ax.grid()

ax.set_xlabel( 'Epochs' )

ax.set_ylabel( 'Test Accuracy' )

ax.set_title( 'Collaboration Performance' )

epochs = []

for i in range(21):

  epochs.append(i+1)

#print(np.array(data).shape)


dat  = []

for erf in range(10):
  dat.append(np.array(data[erf]))
  #ax.plot(epochs,np.array(data[erf]).mean(axis = 0), '-o', label="model {0} ".format(erf))

ax.plot(epochs,(np.array(dat)).mean(axis = 0), '-o', label="Average Accuracy")
ax.legend(loc='lower right')



plt.savefig("Collaboration_Performance1.png")



fig.show()
