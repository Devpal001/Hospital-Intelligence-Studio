import math, copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("hospital_bed_occupancy_los_10000_clean.csv")

#in this dataset, we have two columns: Condition and Length_of_Stay. 
# We will use the Condition as the input feature and 
# Length_of_Stay as the target variable for our linear regression model.


'''x_train = df["Comorbidities_Count"].values
y_train = df["Length_of_Stay_Days"].values

print("x_train:", x_train)
print("y_train:", y_train)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_train type:", type(x_train))
print("y_train type:", type(y_train))

#m is the number of training examples
m= x_train.shape[0]
print("Number of training examples:", m)

# i is the index of the training example we want to access
i=0
x_i = x_train[i]
y_i = y_train[i]
print("x_i:", x_i)
print("y_i:", y_i)

#visualize the data
plt.scatter(x_train, y_train,)
plt.xticks(rotation=45)
plt.xlabel("Comorbidities Count")
plt.ylabel("Length of Stay")
plt.title("Comorbidities Count vs Length of Stay")
plt.show()

w=2
b=5
print("Initial weight:", w)
print("Initial bias:", b)


def compute_model_output(x, w, b):
    """
    Compute the output of the linear regression model for a given input x, weight w, and bias b.
    
    Parameters:
    x (float): The input feature value.
    w (float): The weight parameter of the model.
    b (float): The bias parameter of the model.
    
    Returns:
    float: The predicted output of the model.
    """
    m = x.shape[0]  # Number of training examples
    f_wb = np.zeros(m)  # Initialize the output array with zeros
    for i in range(m):
        f_wb[i] = w * x[i] + b  # Compute the model output for each training example
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)

plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.plot(x_train, tmp_f_wb, color='red', label='Model Output') 
plt.title("Model Output vs Training Data")
plt.xlabel("Comorbidities Count")
plt.ylabel("Length of Stay")
plt.legend()
plt.show()



def compute_cost(x, y, w, b):
    """
    Compute the cost function for linear regression.
    
    Parameters:
    x (numpy.ndarray): The input feature values.
    y (numpy.ndarray): The target variable values.
    w (float): The weight parameter of the model.
    b (float): The bias parameter of the model.
    
    Returns:
    float: The computed cost value.
    """
    m = x.shape[0]  # Number of training examples
    total_cost = 0.0  # Initialize total cost to zero
    
    for i in range(m):
        f_wb_i = w * x[i] + b  # Compute the model output for the i-th training example
        cost_i = (f_wb_i - y[i]) ** 2  # Compute the squared error for the i-th training example
        total_cost += cost_i  # Accumulate the total cost
    
    cost = total_cost / (2 * m)  # Compute the average cost
    return cost
print("Cost at w = 2, b = 5:", compute_cost(x_train, y_train, w, b))
  

def compute_gradient(x, y, w, b):
    """
    Compute the gradient of the cost function with respect to the weight and bias parameters.
    
    Parameters:
    x (numpy.ndarray): The input feature values.
    y (numpy.ndarray): The target variable values.
    w (float): The weight parameter of the model.
    b (float): The bias parameter of the model.
    
    Returns:
    tuple: A tuple containing the gradients with respect to weight and bias (dj_dw, dj_db).
    """
    m = x.shape[0]  # Number of training examples
    dj_dw = 0.0  # Initialize gradient with respect to weight
    dj_db = 0.0  # Initialize gradient with respect to bias
    
    for i in range(m):
        f_wb_i = w * x[i] + b  # Compute the model output for the i-th training example
        error_i = f_wb_i - y[i]  # Compute the error for the i-th training example
        dj_dw += error_i * x[i]  # Accumulate the gradient with respect to weight
        dj_db += error_i  # Accumulate the gradient with respect to bias
    
    dj_dw /= m  # Average the gradient with respect to weight
    dj_db /= m  # Average the gradient with respect to bias
    
    return dj_dw, dj_db  # Return the gradients as a tuple


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Perform gradient descent to optimize the weight and bias parameters.
    
    Parameters:
    x (numpy.ndarray): The input feature values.
    y (numpy.ndarray): The target variable values.
    w_in (float): Initial weight parameter.
    b_in (float): Initial bias parameter.
    alpha (float): Learning rate.
    num_iters (int): Number of iterations for gradient descent.
    
    Returns:
    tuple: A tuple containing the optimized weight and bias parameters (w_out, b_out).
    """
    w = copy.deepcopy(w_in)  # Create a copy of the initial weight
    b = copy.deepcopy(b_in)  # Create a copy of the initial bias
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)  # Compute gradients
        w -= alpha * dj_dw  # Update weight
        b -= alpha * dj_db  # Update bias
        
        if i % 100 == 0:  # Print cost every 100 iterations
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i}: Cost {cost:.4f}, w {w:.4f}, b {b:.4f}")
    
    return w, b

w_init = 0.0  # Initial weight
b_init = 0.0  # Initial bias

iterations = 1000  # Number of iterations for gradient descent
tmp_alpha= 1.0e-2  # Learning rate
w_final, b_final = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations)
print(f"(w,b) found by gradient descent: ({w_final:.4f},{b_final:.4f})")

w = w_final                      
b = b_final
x_i = 5  # Example input feature value (Comorbidities Count)
length_of_stay_days = w * x_i + b    

print(f"{length_of_stay_days:.0f} Days of Length of Stay for a patient with {x_i} Comorbidities Count")'''


x_train = df[["Age","Comorbidities_Count","Surgery_Flag","ICU_Flag","Hospital_Acquired_Infection","Discharge_Delay_Days","Social_Support"]].values
y_train = df["Length_of_Stay_Days"].values

b_init = 2  # Initial bias
w_init = np.array([2,3,4,5,4,1,3])  # Initial weight
print(f"Initial weight: {w_init}, Initial bias: {b_init}")

'''def predict_single_loop(x, w, b):
    """
    Predict the output for a single input feature value using the linear regression model.
    
    Parameters:
    x (float): The input feature value.
    w (float): The weight parameter of the model.
    b (float): The bias parameter of the model.
    
    Returns:
    float: The predicted output of the model.
    """
    n = x.shape[0]  # Number of training examples
    p=0  # Initialize predictions array with zeros
    for i in range(n):
        p_i=x[i]*w[i] # Compute the predicted output for the i-th training example
        p=p+p_i  # Accumulate the predicted outputs
    p = p + b  # Add the bias term to the accumulated predictions
    return p  # Return the predicted output'''


#multivariate linear regression model for predicting length of stay based on multiple features

'''def predict(x,w,b):
    """
    Predict the output for a single input feature value using the linear regression model.
    
    Parameters:
    x (numpy.ndarray): The input feature values.
    w (numpy.ndarray): The weight parameters of the model.
    b (float): The bias parameter of the model.
    
    Returns:
    float: The predicted output of the model.
    """
    p = np.dot(x, w) + b  # Compute the dot product of input features and weights, then add bias
    return p  # Return the predicted output

x_vec = x_train[0]

print("x_vec:", x_vec)
print("x_vec length:", len(x_vec))
print("w_init:", w_init)
print("w_init length:", len(w_init))


f_wb_vec = predict(x_vec, w_init, b_init)  # Predict outputs for x_vec
print("f_wb_vec:", f_wb_vec)  # Print the predicted outputs

i have to remove this hash
#f_wb_vec = predict_single_loop(x_vec, w_init, b_init)  # Predict outputs for x_vec
#print("f_wb_vec:", f_wb_vec)  # Print the predicted outputs

def compute_cost_multi(x, y, w, b):
    """
    Compute the cost function for linear regression with multiple features.
    
    Parameters:
    x (numpy.ndarray): The input feature values (2D array).
    y (numpy.ndarray): The target variable values.
    w (numpy.ndarray): The weight parameters of the model (1D array).
    b (float): The bias parameter of the model.
    
    Returns:
    float: The computed cost value.
    """
    m = x.shape[0]  # Number of training examples
    total_cost = 0.0  # Initialize total cost to zero
    
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b  # Compute the model output for the i-th training example
        cost_i = (f_wb_i - y[i]) ** 2  # Compute the squared error for the i-th training example
        total_cost += cost_i  # Accumulate the total cost
    
    cost = total_cost / (2 * m)  # Compute the average cost
    return cost

compute_cost_multi(x_train, y_train, w_init, b_init)  # Compute the cost for the initial weights and bias
print("Cost at initial w and b:", compute_cost_multi(x_train, y_train, w_init, b_init))  # Print the computed cost

def compute_gradient_multi(x, y, w, b):
    """
    Compute the gradient of the cost function with respect to the weight and bias parameters for multiple features.
    
    Parameters:
    x (numpy.ndarray): The input feature values (2D array).
    y (numpy.ndarray): The target variable values.
    w (numpy.ndarray): The weight parameters of the model (1D array).
    b (float): The bias parameter of the model.
    
    Returns:
    tuple: A tuple containing the gradients with respect to weight and bias (dj_dw, dj_db).
    """
    m, n = x.shape  # Number of training examples and number of features
    dj_dw = np.zeros(n)  # Initialize gradient with respect to weight as a zero vector
    dj_db = 0.0  # Initialize gradient with respect to bias
    
    for i in range(m):
        f_wb_i = np.dot(x[i], w) + b  # Compute the model output for the i-th training example
        error_i = f_wb_i - y[i]  # Compute the error for the i-th training example
        dj_dw += error_i * x[i]  # Accumulate the gradient with respect to weight
        dj_db += error_i  # Accumulate the gradient with respect to bias
    
    dj_dw /= m  # Average the gradient with respect to weight
    dj_db /= m  # Average the gradient with respect to bias
    
    return dj_dw, dj_db  # Return the gradients as a tuple

tmp_dj_dw, tmp_dj_db = compute_gradient_multi(x_train, y_train, w_init, b_init)  # Compute gradients for initial weights and bias
print("Gradient at initial w (dj_dw):",(tmp_dj_dw))  # Print the computed gradients
print("Gradient at initial b (dj_db):",(tmp_dj_db))  # Print the computed gradients


def gradient_descent_multi(x, y, w_in, b_in, alpha, num_iters):
    """
    Perform gradient descent to optimize the weight and bias parameters for multiple features.
    
    Parameters:
    x (numpy.ndarray): The input feature values (2D array).
    y (numpy.ndarray): The target variable values.
    w_in (numpy.ndarray): Initial weight parameters (1D array).
    b_in (float): Initial bias parameter.
    alpha (float): Learning rate.
    num_iters (int): Number of iterations for gradient descent.
    
    Returns:
    tuple: A tuple containing the optimized weight and bias parameters (w_out, b_out).
    """
    w = copy.deepcopy(w_in)  # Create a copy of the initial weights
    b = copy.deepcopy(b_in)  # Create a copy of the initial bias
    
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_multi(x, y, w, b)  # Compute gradients
        w -= alpha * dj_dw  # Update weights
        b -= alpha * dj_db  # Update bias
        
        if i % 100 == 0:  # Print cost every 100 iterations
            cost = compute_cost_multi(x, y, w, b)
            print(f"Iteration {i}: Cost {cost:.4f}, w {w}, b {b:.4f}")
    
    return w, b  # Return the optimized weights and bias


initial_w = np.zeros(x_train.shape[1])  # Initialize weights as a zero vector
initial_b = 0.0  # Initialize bias as zero

iterations=10000 # Number of iterations for gradient descent
tmp_alpha= 1.0e-4  # Learning rate

w_final, b_final = gradient_descent_multi(x_train, y_train, initial_w, initial_b, tmp_alpha, iterations)  # Perform gradient descent
print(f"(w,b) found by gradient descent: ({w_final},{b_final:.4f})")  # Print the optimized weights and bias'''


#Prediction for a specific patient based on their Patient ID

w = np.array([0.09271369, 1.07433568, 0.68028327, 0.55618327, 0.33216646, 1.00949448, 0.24837546])  # Convert the optimized weights to a NumPy array                     

b = 0.6651

patient_id = int(input("Enter Patient ID: "))

patient = df[df["Patient_ID"] == patient_id]

if patient.empty:
    print("Patient ID not found.")

else:
    feature_names = [
        "Age",
        "Comorbidities_Count",
        "Surgery_Flag",
        "ICU_Flag",
        "Hospital_Acquired_Infection",
        "Discharge_Delay_Days",
        "Social_Support"
    ]

    x_patient = patient[feature_names].values[0]

    prediction = np.dot(x_patient, w) + b

    print("\nPatient Details")
    print("----------------------------")

    for name, value in zip(feature_names, x_patient):
        print(f"{name}: {value}")

    print("----------------------------")

    print(
        f"Predicted Length of Stay: {prediction:.2f} days"
    )

    print(
        f"Actual Length of Stay: "
        f"{patient['Length_of_Stay_Days'].iloc[0]} days"
    )


'''w = np.array(w_final)  # Convert the optimized weights to a NumPy array                     
b = b_final  # Assign the optimized bias
x_i = np.array([65, 5, 1, 0, 0, 2, 1])  # Example input feature values (Age, Comorbidities Count, Surgery Flag, ICU Flag, Hospital Acquired Infection, Discharge Delay Days, Social Support)
length_of_stay_days = np.dot(w, x_i) + b  # Compute the predicted length of stay using the optimized weights and bias 

print(f"{length_of_stay_days:.0f} Days of Length of Stay for a patient with {x_i} Comorbidities Count")'''



