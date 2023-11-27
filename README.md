# Linear Regression problem with one variable

# Problem Description
Consider the attached file dataset1.txt. The first column of the data file shows the input data (x), and
the second column shows each samples’ output value (y).
1. What is the cost function 𝐽(θ) equation for linear regression?
2. Fit a linear regression model on your data using:
    a. Closed-form solution calculated by MSE method
    b. Gradient descent method in online (stochastic) mode (1500 iterations)
    c. Gradient descent method in batch mode (1500 iterations)
3. Plot the dataset and superimpose the fitted models using the three above methods.
4. Use each estimated parameter θ (for each method) to predict the output for x=6.2, 12.8,
22.1, 30.
5. Compare the parameter θ estimated by each method by plotting them in one figure.
6. Plot the cost function 𝐽(θ) along the epochs (plot both online & batch methods on one
figure using hold on command).

# Performance Report
This code is a small project on linear regression. It involves several functions related to linear modeling and three different methods of training the model: close-form, online-mode, and batch-mode.

1. **generate_data()**: Loads a text file ("dataset1.txt") and reads the data in it into X and Y columns.

2. **close_form()**: Utilizes the close-form method to determine the parameters of the linear model. It calculates the model parameters directly using mathematical equations.

3. **online_mode()**: Updates the linear model using the online-mode training method, iterating to update model parameters.

4. **batch_mode()**: Updates the linear model using the batch-mode training method, also implemented with an iterative loop.

5. **predict()**: A helper function that returns model predictions.

6. **cost_online()** and **cost_batch()**: These functions calculate the cost function for online and batch training methods, respectively.

7. Then, for each training method, there's a plot showing the predicted model line based on the trained model. Finally, a plot combining all three model lines is shown.
