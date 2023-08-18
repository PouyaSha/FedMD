import numpy as np
import matplotlib.pyplot as plt

img = (np.random.standard_normal([28, 28]) )#.astype(np.uint32)
#img = (np.uint8([[ i+j for i in range(28)] for j in range(28)]))
# see the raw result (it is 'antialiased' by default)
_ = plt.imshow(img)#, interpolation='none')
# if you are not in a jupyter-notebook
plt.show()
plt.savefig("q.png")
print(img.max())