# This vanilla NN finds the smallest numbers position in the initial Array 

import numpy as np 

initial = np.array([[1, 2, 3, 4]]) # NOTE: USE np.array([]) FOR MULTIDIMENSIONAL TENSORS

hidden = np.array([[1, 5.433333, 15, 7.88888]])

output = np.array([[15, 9.222222122222222222222222222222222, 10, 11.6777]])
print(output)
print(output.reshape(4, 1))

bias = 7

def activation_function(value):
    value = value/15 # This is the only makshift way I could think of to deal with every output being 1, so i normalised it to 15 because the distribution is kinda like that facepalm
    output = float((2.0/(1.0 + np.exp(-value))) - 1)
    # print("exponent", np.exp(-value))
    return output

# Please ignore the atrocious naming conventions :sob: this is just meant to be pesudo code for C :skull:
# Ill make it more efficient later with numpy or something wah wah
weight1o = initial[0, 0] * hidden[0, 0] + initial[0, 1] * hidden[0, 0] + initial[0, 2] * hidden[0, 0] + initial[0, 3] * hidden[0, 0] + bias
weight2o = initial[0, 0] * hidden[0, 1] + initial[0, 1] * hidden[0, 1] + initial[0, 2] * hidden[0, 1] + initial[0, 3] * hidden[0, 1] + bias
weight3o = initial[0, 0] * hidden[0, 2] + initial[0, 1] * hidden[0, 2] + initial[0, 2] * hidden[0, 2] + initial[0, 3] * hidden[0, 2] + bias
weight4o = initial[0, 0] * hidden[0, 3] + initial[0, 1] * hidden[0, 3] + initial[0, 2] * hidden[0, 3] + initial[0, 3] * hidden[0, 3] + bias

# print(weight1o)

weight1o = activation_function(weight1o)

# print(weight1o)


weight2o = activation_function(weight2o)
weight3o = activation_function(weight3o)
weight4o = activation_function(weight4o)

weight_sequence = [weight1o, weight2o, weight3o, weight4o]

# print(weight_sequence)

# below line doesnt work because when multiplying matrices with the dimensions 1x4 & 4x1 it results in just adding up everything. im so rusty that i forgot the fundamentals
# mb chat
# ol_output =  np.matmul(weight_sequence, output.reshape(4, 1))

ol_output = weight_sequence * output

print(ol_output)

# u gotta nest a for loop cus an numpy array has two dimensions, first it iterates thru the total number of arrays, then the individual array's elemnts
# cant believe this loop took me 20 mins extremely rusty SMHFKHEAD
# for i, o in enumerate(ol_output):
for x in ol_output:
    for index, y in enumerate(x):
        ol_output[0, index] = ol_output[0, index] + bias
        ol_output[0, index] = activation_function(ol_output[0, index])
    

final_answer = float('inf')

for x in ol_output:
    for index, y in enumerate(x):
        # WHY NOT U STUPID BASTARD
        valuer = ol_output[0, index]
        if valuer < final_answer:
            print("cool")
            final_answer = index  

# its hilarious that without backpropagation implemented yet and with just randomly initialized numbers it somehow finds the right position by default
# i think i should start calling myself a prophet LOL
print(f"The smallest number is in the {final_answer}th position of the array")        











