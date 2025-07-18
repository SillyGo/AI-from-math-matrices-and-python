
from math import sqrt
from numpy.random import randn

#-------------------------------------------------------------------------------------------------------------------------------------------------------#
#FUNÇÕES BASICAS:

def random_matrix_init(linhas:int, colunas:int, scope:tuple):
    retorno = []
    for i in range(linhas):
        std = sqrt(2.0 / colunas)
        numbers = randn(colunas)
        scaled = numbers * std
        scaled = list(scaled)
        retorno.append(scaled)
    return retorno

def null_matrix_init(linhas:int, colunas:int, scope:tuple):
    retorno = []
    for i in range(linhas):
        retorno.append([])
        for j in range(colunas):
            retorno[i].append(0)
    return retorno

def unit_matrix_init(linhas:int, colunas:int, scope:tuple):
    retorno = []
    for i in range(linhas):
        retorno.append([])
        for j in range(colunas):
            retorno[i].append(1)
    return retorno

def matrix_sum(matrix1:list, matrix2:list):
    linhas, colunas = len(matrix1), len(matrix1[0])
    placeholder = null_matrix_init(linhas, colunas, (0,0))
    for i in range(linhas):
        for j in range(colunas):
            placeholder[i][j] = matrix1[i][j] + matrix2[i][j]
    return placeholder

def dot_product(vec1:list, vec2:list):
    sum = 0
    for i in range(len(vec1)):
        sum = sum + vec1[i]*vec2[i]
    return sum

def relu(a):
    if (a > 0):
        return a
    else:
        return 0

def matrix_mult(matrix1:list, matrix2:list):
    linhas1, colunas1 = len(matrix1), len(matrix1[0])
    colunas2 = len(matrix2[0])
    placeholder = null_matrix_init(linhas1, colunas2, (0,0))
    for i in range(linhas1):
        for j in range(colunas2):
            sum = 0
            for k in range(colunas1):
                sum = sum + matrix1[i][k]*matrix2[k][j]
            placeholder[i][j] = sum
    return placeholder

def scalar_mult(k:float, matriz:list): 
    placeholder = deep_copy(matriz)
    for i in range(len(placeholder)):
        for j in range(len(placeholder[i])):
            placeholder[i][j] = k*placeholder[i][j]
    return placeholder

def apply_act_function(activation_function:str, matrix:list):
    if activation_function == "relu":
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = relu(matrix[i][j])
    return matrix

def print_matrix(matriz:list):
    print("-----------")
    for i in range(len(matriz)):
        out = ""
        for j in range(len(matriz[i])):
            out = out + str(matriz[i][j]) + " "
        print(out)
    print("-----------")

def print_network(NN:list):
    for i in range(len(NN)):
        print("------------------------------------")
        print("LAYER", i + 1)
        print_matrix(NN[i][0])
        print_matrix(NN[i][1])
        print("------------------------------------")

def deep_copy(matriz2:list):
    placeholder = []
    for i in range(len(matriz2)):
        placeholder.append([])
        for j in range(len(matriz2[0])):
            placeholder[i].append(matriz2[i][j])
    return placeholder

#def unison_shuffled_copies(a, b):   #this code was taken from stack overflow
#    assert len(a) == len(b)
#    p = np.random.permutation(len(a))
#    return a[p], b[p]

def relu_derivative(a:float):
    if a > 0:
        return 1
    elif a == 0:
        return 0
    
def transposta(matriz:list):
    placeholder = null_matrix_init(len(matriz[0]), len(matriz), (0,0))
    for i in range(len(placeholder)):
        for j in range(len(placeholder[0])):
            placeholder[i][j] = matriz[j][i]
    return placeholder

def triangulate(c_matrix:list):
    linhas = len(c_matrix)
    placeholder = null_matrix_init(linhas, linhas, (0,0))
    for i in range(linhas):
        placeholder[i][i] = c_matrix[i][0]
    return placeholder


#-------------------------------------------------------------------------------------------------------------------------------------------------------#
#A PARTE ENVOLVENDO REDES NEURAIS:  

def network(sizes:list): #layers e sizes devem ser listas com o mesmo tamanho, logo que sizes representa o numero de neuronios por layer.
    layers = len(sizes)
    NN = []
    for i in range(layers):
        if i == 0:
            NN.append([])
            NN[i].append(null_matrix_init(sizes[i], 1, (0,0))) #cria a matriz coluna dos viézes
            NN[i].append(unit_matrix_init(sizes[i], 1, (1,1))) #cria a matriz dos pesos
        elif i == layers - 1:
            NN.append([])
            NN[i].append(null_matrix_init(sizes[i], 1, (38.5,38.5)))
            NN[i][0][0][0] = 0
            NN[i].append(random_matrix_init(sizes[i], sizes[i-1], (1,10)))
        else:
            NN.append([])
            NN[i].append(null_matrix_init(sizes[i], 1, (0,0)))
            NN[i].append(random_matrix_init(sizes[i], sizes[i-1], (1,10)))
    return NN

def feed_forward(input_list:list, NN:list, stop:int):
    last_output = deep_copy(input_list)
    for i in range(1,stop):
        if len(last_output) == 1 and len(last_output[0]) == 1:
            last_output = apply_act_function("relu", matrix_sum(scalar_mult(last_output[0][0], NN[i][1]), NN[i][0]))
        else:
            last_output = apply_act_function("relu", matrix_sum(matrix_mult(NN[i][1], last_output), NN[i][0]))
    return last_output  

def ff_list(input_list:list, NN:list, stop:int):
    out = []
    last_output = deep_copy(input_list)
    out.append(last_output)
    for i in range(1,stop):
        if len(last_output) == 1 and len(last_output[0]) == 1:
            last_output = apply_act_function("relu", matrix_sum(scalar_mult(last_output[0][0], NN[i][1]), NN[i][0]))
        else:
            last_output = apply_act_function("relu", matrix_sum(matrix_mult(NN[i][1], last_output), NN[i][0]))
        out.append(last_output)
    return out  

def w_gradient(outputs:list, sum_matrix:list, I:int):
    for j in range(len(outputs)):
        new_copy = deep_copy(sum_matrix)
        for k in range(len(new_copy)):
            new_copy[k][0] = new_copy[k][0]*relu_derivative(outputs[j][0][I][k][0])
        new_copy = triangulate(new_copy)
        p_output_matrix = null_matrix_init(len(new_copy[0]), len(outputs[j][0][I - 1]), (0,0))
        for k1 in range(len(p_output_matrix)):
            for k2 in range(len(p_output_matrix[0])):
                p_output_matrix[k1][k2] = outputs[j][0][I - 1][k2][0]
        if j == 0:
            weight_gradient = matrix_mult(new_copy, p_output_matrix)
        else:
            weight_gradient = matrix_sum(weight_gradient, matrix_mult(new_copy, p_output_matrix))
        j = len(outputs)
    return weight_gradient

def backprop(X_train:list, Y_train:list, NN:list, error_function:str, learning_rate:float):
    sum = 0
    NN_copy = deep_copy(NN)
    if error_function == "MSE":
        outputs = [] #serão organizados da seguinte forma: linhas:= batch; colunas:= outputs por layer.
        for i in range(len(X_train)):
            outputs.append([])
            outputs[i].append(ff_list(X_train[i], NN, len(NN)))
        for i in range(0, len(NN) - 1):
            I = len(NN) - i - 1
            if I == len(NN) - 1:
                for j in range(len(X_train)):
                    if j == 0:
                        sum_matrix = matrix_sum(Y_train[j],scalar_mult((-1),outputs[j][0][len(NN) - 1]))   #matrix_sum = Y_train[j] - outputs[j][len(NN)]
                    else:
                        sum_matrix = matrix_sum(sum_matrix, matrix_sum(Y_train[j],scalar_mult((-1),outputs[j][0][len(NN) - 1])))   #matrix_sum = matrix_sum + (Y_train[j] - outputs[j][len(NN)])
                sum_matrix = scalar_mult(-2/len(X_train), sum_matrix) #aqui temos as derivadas dos últimos outputs, em forma de matriz coluna.
                previous = deep_copy(sum_matrix)
                weight_gradient = w_gradient(outputs, sum_matrix, I)
                for k1 in range(len(weight_gradient)):
                    for k2 in range(len(weight_gradient[k1])):
                        sum = sum + weight_gradient[k1][k2]
                NN[I][1] = matrix_sum(NN[I][1], scalar_mult(-learning_rate, weight_gradient))
                NN[I][0] = matrix_sum(NN[I][0], scalar_mult(100*learning_rate, previous))
            else:
                sum_matrix = deep_copy(previous)
                sum_matrix = matrix_mult(transposta(NN_copy[I + 1][1]), sum_matrix)
                previous = deep_copy(sum_matrix)
                weight_gradient = w_gradient(outputs, sum_matrix, I)
                for k1 in range(len(weight_gradient)):
                    for k2 in range(len(weight_gradient[k1])):
                        sum = sum + weight_gradient[k1][k2]
                NN[I][1] = matrix_sum(NN[I][1], scalar_mult(-learning_rate, weight_gradient))
                NN[I][0] = matrix_sum(NN[I][0], scalar_mult(learning_rate, previous))
    return sum

def MSE(NN:list, X_train:list, Y_train:list):
    error = 0
    for i in range(len(X_train)):
        prediction = feed_forward(X_train[i], NN, len(NN))
        for j in range(len(prediction)):
            error = error + (Y_train[i][j] - prediction[j])**2
    return error/len(X_train)

NN = network([]) 
print_network(NN)
epochs = 2000
X_train1 = [[[0]], [[1]], [[2]], [[3]], [[4]], [[5]]]
Y_train1 = [[[0]], [[1]], [[4]], [[9]], [[16]], [[25]]]
X_train2 = [[[6]], [[7]], [[8]], [[9]], [[10]]]
Y_train2 = [[[36]], [[49]], [[64]], [[81]], [[100]]]
