import numpy as np
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer


def attacker_logit_update(averaged_logit, model, logits_matching_batchsize, N_logits_matching_round, alignment_data, private_test_data)

    
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
    print(np.mean(self.private_test_data["y"] == y_pred))
    del y_pred

    print("updates attacker model ...")
    weights_to_use = None
    weights_to_use = d["model_weights"]
    d["model_logits"].set_weights(weights_to_use)
    d["model_logits"].fit(alignment_data, averaged_logit, 
                                      batch_size = logits_matching_batchsize,  
                                      epochs = N_logits_matching_round, 
                                      shuffle=True, verbose = 0)
    d["model_weights"] = d["model_logits"].get_weights()

    return None
