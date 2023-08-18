import os
import errno
import argparse
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from PIL import Image
from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer
import numpy as np
from tensorflow.keras.models import load_model

from data_utils import load_MNIST_data, load_EMNIST_data, generate_EMNIST_writer_based_data, generate_partial_data
from FedMD import FedMD
from Neural_Networks import train_models, cnn_2layer_fc_model, cnn_3layer_fc_model

private_classes = [10, 11, 12, 13, 14, 15]
public_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST = load_MNIST_data(standarized = True, verbose = True)
    
public_dataset = {"X": X_train_MNIST, "y": y_train_MNIST}

sample = {"X": X_train_MNIST[0:1] , "y": y_train_MNIST[0:1]}
print(sample["X"].shape)    
    
X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST, \
writer_ids_train_EMNIST, writer_ids_test_EMNIST \
= load_EMNIST_data("./dataset/emnist-letters.mat",
                       standarized = True, verbose = True)
    
y_train_EMNIST += len(public_classes)
y_test_EMNIST += len(public_classes)
    
#generate private data
private_data, total_private_data\
=generate_EMNIST_writer_based_data(X_train_EMNIST, y_train_EMNIST,
                                       writer_ids_train_EMNIST,
                                       N_parties = 10, 
                                       classes_in_use = private_classes, 
                                       N_priv_data_min = 3 * len(private_classes)
                                      )
    
X_tmp, y_tmp = generate_partial_data(X = X_test_EMNIST, y= y_test_EMNIST, 
                                         class_in_use = private_classes, verbose = True)
private_test_data = {"X": X_tmp, "y": y_tmp}
del X_tmp, y_tmp
"""
CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layer_fc_model, 
                    "3_layer_CNN": cnn_3layer_fc_model} 

attack_model_name = "3_layer_CNN"
attacker_saved_names = ["Attacker Model"]
attack_model_params = {"n1": 128, "n2": 128, "n3": 128, "dropout_rate": 0.3}
attack_model = CANDIDATE_MODELS[attack_model_name](n_classes=16,
                                                        input_shape=(28,28),
                                                            **attack_model_params)
"""
file = open("result_FEMNIST_imbalanced/attacker.pkl","rb")

model = pickle.load(file)






model_A_twin = None
model_A_twin = clone_model(model)
model_A_twin.set_weights(model.get_weights())
model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3), 
                            loss = "sparse_categorical_crossentropy",
                            metrics = ["accuracy"])
model_A = remove_last_layer(model_A_twin, loss="mean_absolute_error")

d = {"model_logits": model_A, 
        "model_classifier": model_A_twin,
        "model_weights": model_A_twin.get_weights()}



print("attacker test performance ... ")
            
y_pred = d["model_classifier"].predict(private_test_data["X"], verbose = 0).argmax(axis = 1)            
print(np.mean(private_test_data["y"] == y_pred))
del y_pred

n = 500

plt.imshow(private_test_data["X"][n])
plt.savefig("m.png")

print(d["model_classifier"].predict(private_test_data["X"])[n])

e = []

for i in range(16):

  e.append(i+1)


print(d["model_logits"].predict(private_test_data["X"])[n])





fig = plt.figure()

ax1 = plt.subplot( 2,1,1 )

ax1.minorticks_on()

ax1.grid()

ax1.set_xlabel( 'classes' )

ax1.set_ylabel( 'lables' )

img = (np.random.standard_normal([28, 28]) * 255 )

ax1.bar(e,d["model_classifier"].predict(private_test_data["X"])[n])
"""
im = []
im.append(img)
im.append(img)
print(np.array(im).shape)
"""
im = img[np.newaxis , ...]
print(im.shape)
print(img.shape)
ax1.bar(e,d["model_classifier"].predict(im))



ax2 = plt.subplot( 2,1,2 )

ax2.minorticks_on()

ax2.grid()

ax2.set_xlabel( 'classes' )

ax2.set_ylabel( 'logits' )



ax2.bar(e,-d["model_logits"].predict(private_test_data["X"])[n])



plt.savefig("C.png")

y_true = [10]
y_pred = [d["model_classifier"].predict(private_test_data["X"])[n]]
scce = tf.keras.losses.SparseCategoricalCrossentropy()
print("loss: ",scce(y_true, y_pred).numpy())

print(type(private_test_data["X"][0][0][0]))
print(type(im[0][0][0]))
"""
print(type(d["model_classifier"].predict(private_test_data["X"])))
print(type(img))
"""
