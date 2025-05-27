from tensorflow import keras
import numpy as np
import pandas as pd
import dask.dataframe as dk
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from datetime import datetime, timedelta
import tenseal as ts
# Định nghĩa mô hình CNN
# VGG, ...
# Conv2D, tabular, ...
# HE, tính tương thích của HE với CNN
# Tính chất data in, out; Học tăng cường

model = models.load_model("/mnt/c/Users/hoang/D/Code/CoCheMalware/federated_learning_log/Month 05 -- Day 22___22h 04p/client_0_log/model_1.keras")

weights_list = []
biases_list =  []
local_weights_shape=[]


for index, layer in enumerate(model.layers):
    if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
        weights_list.append(layer.get_weights()[0])
        biases_list.append(layer.get_weights()[1])
        local_weights_shape.append(layer.get_weights()[0].shape)   
        for s in layer.get_weights()[1].shape:
            print ("shape: ",s)

print(local_weights_shape)
# print(weights_list)
from numbers import Real
import random
import copy

# i=0
# for index, layer in enumerate(model.layers):
#     if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
#         layer.set_weights([weights_list[i], biases_list[i]])
#         i+=1

# arr = [w.flatten() for w in weights_list]
# arr= np.concatenate(arr).ravel()
# print (arr)
# context = ts.context(
#     ts.SCHEME_TYPE.CKKS, # ckks cho số thực, bfv cho int
#     poly_modulus_degree=32768,
#     coeff_mod_bit_sizes=[60, 40,40, 40, 60]
# )
# context.generate_galois_keys()
# context.global_scale = 2**40

# vector = ts.ckks_vector(context, arr).serialize()
# encrypted_weights = ts.lazy_ckks_vector_from(vector)

# # context = ts.context_from(message_0.body['context'])
# encrypted_weights.link_context(context)

# encrypted_weights +=encrypted_weights
# encrypted_weights = encrypted_weights*(1/2)
# encrypted_weights= encrypted_weights.serialize()


# encrypted_weights = ts.lazy_ckks_vector_from(encrypted_weights)
# encrypted_weights.link_context(context)
 
# decrypted_weights = np.array(encrypted_weights.decrypt())
# print(decrypted_weights)
# def de_flatten_weights(flattened_weights):
#     weights=[]
#     right_pointer=0
#     for shape in local_weights_shape:
#         delta = 1
#         for i in shape:
#             delta*=i
#         weights.append(np.array(flattened_weights[right_pointer*1:right_pointer+delta].reshape(shape))) 
#         right_pointer +=delta

#     return weights

# deflatten_weights  = de_flatten_weights(decrypted_weights)


import pandas as pd
import matplotlib.pyplot as plt

# Đọc log từ file CSV
log_name = "/mnt/c/Users/hoang/D/Code/CoCheMalware/Centralized_Log/Month05Day26__10h43p.csv"
log_df = pd.read_csv(log_name)  # Đổi tên file nếu cần

epochs = range(1, len(log_df) + 1)

plt.figure(figsize=(16, 10))

# Accuracy
plt.subplot(2, 2, 1)
plt.plot(epochs, log_df['accuracy'], 'b-', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.ylim(0.9, 1.0)

# Loss
plt.subplot(2, 2, 2)
plt.plot(epochs, log_df['loss'], 'r-', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Validation Accuracy
plt.subplot(2, 2, 3)
plt.plot(epochs, log_df['val_accuracy'], 'g-', label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.ylim(0.95, 1.0)

# Validation Loss
plt.subplot(2, 2, 4)
plt.plot(epochs, log_df['val_loss'], 'm-', label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()




# def laplace(mean, sensitivity, epsilon): # mean : value to be randomized (mean)
#         scale = sensitivity / epsilon
#         rand = random.uniform(0,1) - 0.5 # rand : uniform random variable
#         return mean - scale * np.sign(rand) * np.log(1 - 2 * np.abs(rand))

# mean = 0
# sensitivity =  2 / (3*50000*1.0)
# epsilon = 0.01
# # print("Sensivity", sensi)
# # for index in range(20):
# #     print(laplace(0, sensi, 0.01))

# local_weights_shape = []
# weights = model.get_weights()
# local_weights_noise= {}
# local_weights_noise[1] = []
# weights_with_noise= []
# for index, weights_layer in enumerate(weights):
#     weights_shape = weights_layer.shape
#     weights_dp_noise = np.zeros(weights_layer.shape)
#     local_weights_shape.append(weights_layer.shape)
#     if np.all(abs(weights_layer) > 1e-15):
#         print(weights_shape)
#         if len(weights_shape) ==1:
#             for i_shape in range(weights_shape[0]):
#                 weights_dp_noise[i_shape] = laplace(mean=mean, 
#                             sensitivity=sensitivity,
#                             epsilon=epsilon)
#         elif len(weights_shape) ==2:
#             for i_shape in range(weights_shape[0]):
#                 for j_shape in range(weights_shape[1]):
#                     weights_dp_noise[i_shape][j_shape]=laplace(mean=mean, 
#                             sensitivity=sensitivity,
#                         epsilon=epsilon)
#         elif len(weights_shape) ==3 :
#             for i_shape in range(weights_shape[0]):
#                 for j_shape in range(weights_shape[1]):
#                     for z_shape in range(weights_shape[2]):
#                         weights_dp_noise[i_shape][j_shape][z_shape]=laplace(mean=mean, 
#                             sensitivity=sensitivity,
#                             epsilon=epsilon)
#     weights_with_noise.append(weights_layer+weights_dp_noise)
#     local_weights_noise[1].append(weights_dp_noise)
# # print(weights[7])
# # print(local_weights_noise)
# for index, weights_layer in enumerate(weights_with_noise):
#     weights_shape = weights_layer.shape
#     weights_dp_noise = np.zeros(weights_layer.shape)
    
# # print(weights_with_noise[7])
# # print(weights_with_noise)

# # model.set_weights(weights_with_noise)


# arr = [weight.flatten() for weight in weights ]

# arr = np.concatenate(arr).ravel()
# print("ARR", ts.ckks_vector(context, arr))
# a=[]
# right=0

# def de_flatten_weights(flattened_weights):
#     print("Local w shape", local_weights_shape)
#     right_pointer=0
#     weights=[]
#     for shape in local_weights_shape:
#         delta = 1
#         for i in shape:
#             delta*=i
#         weights.append(np.array(flattened_weights[right_pointer*1:right_pointer+delta].reshape(shape))) 
#         right_pointer +=delta
#     return weights

# print(de_flatten_weights(arr))

# for f in a:
#     print(f.shape)
# print(local_weights_shape)



# encrypted_weights = []
# for weight_layer in weights:
#     weights_flat = (weight_layer.flatten())
#     encrypted_weights.append(ts.ckks_vector(context, weights_flat).serialize())

# print(encrypted_weights)

# encrypted_biases = ts.ckks_vector(context, biases)

# import numpy as np

# nested_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
# array = np.array(nested_list, dtype=object)  # dtype=object để xử lý các sublist có độ dài khác nhau
# flattened_array = np.concatenate(array).ravel()
# flattened_list = list(flattened_array)
# print(flattened_list)
# # Kết quả: [1, 2, 3, 4, 5, 6, 7, 8, 9]

# shape =[(1,4), (1,3), (1,2)]
# a= []
# right=0
# for index, s in enumerate(shape):
#     a.append(np.array(flattened_list[1*right:right+s[0]*s[1]]).reshape(s))
#     right += s[0]*s[1]
# print(a)


# def save_shape(self, iteration):
#         if iteration <2:
#             for layer in self.model.layers:
#                 if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
#                     self.local_weights_shape.append(weights.shape)
                
#             for layer in self.model.layers:
#                 if layer.name.startswith('conv1d') or layer.name.startswith('dense'):
#                     self.local_biases_shape.append(layer.get_weights()[1].shape)
#         return None