import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Neural_Networks import remove_last_layer
from PIL import Image
from data_utils import load_MNIST_data, load_EMNIST_data


X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST = load_MNIST_data(standarized = True, verbose = True)
img = X_train_MNIST[0]

file = open("result_FEMNIST_imbalanced/attacker.pkl","rb")
model = pickle.load(file)


#img = (np.random.standard_normal([28, 28]) )





im = img[np.newaxis , ...]



_ = plt.imshow(img)
plt.show()
plt.savefig("initial_input.png")
target_class = [13]
learning_rate = 1
pixel_step = 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
model_A = remove_last_layer(model, loss="mean_absolute_error")

print(np.mean(model_A.predict(im)))

loss = tf.keras.losses.SparseCategoricalCrossentropy() 
new_loss = 5
for k in range (10):
    for i in range(28):
        for j in range(28):
            y_pred = [model.predict(im, verbose=None)[0]]
            old_loss = loss(target_class, y_pred).numpy() - (np.mean(model_A.predict(im, verbose=None))/4)
            #print("old loss: ", old_loss)
            im[0][i][j] += pixel_step
            y_pred = [model.predict(im , verbose=None)[0]]
            new_loss = loss(target_class, y_pred).numpy() - (np.mean(model_A.predict(im, verbose=None))/4)
            print("new loss: ", new_loss)
            im[0][i][j] -= ( pixel_step + (new_loss - old_loss) * learning_rate )
    #print(new_loss)

    img = im[0]
    _ = plt.imshow(img)
    plt.show()
    plt.savefig("last_input.png")



e = []
for i in range(16):
  e.append(i+1)

fig = plt.figure()
ax1 = plt.subplot( 1,1,1 )
ax1.minorticks_on()
ax1.grid()
ax1.set_xlabel( 'classes' )
ax1.set_ylabel( 'logits' )
ax1.bar(e,model_A.predict(im)[0])
plt.savefig("logits.png")