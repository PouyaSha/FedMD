import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


file = open("result_FEMNIST_imbalanced/attacker.pkl","rb")
model = pickle.load(file)


img = (np.random.standard_normal([28, 28]) )
im = img[np.newaxis , ...]
_ = plt.imshow(img)
plt.show()
plt.savefig("initial_input.png")
target_class = [13]
learning_rate = 10
pixel_step = 10                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

loss = tf.keras.losses.SparseCategoricalCrossentropy()

for k in range(10):
    for i in range(28):
        for j in range(28):
            y_pred = [model.predict(im, verbose=None)[0]]
            old_loss = loss(target_class, y_pred).numpy()
            print("old loss: ", old_loss)
            im[0][i][j] += pixel_step
            y_pred = [model.predict(im , verbose=None)[0]]
            new_loss = loss(target_class, y_pred).numpy()
            print("new loss: ", new_loss)
            im[0][i][j] -= ( pixel_step + (new_loss - old_loss) * learning_rate )
    print(new_loss)

img = im[0]
_ = plt.imshow(img)
plt.show()
plt.savefig("last_input.png")
