<br/>
<p align="center">
  <h3 align="center">OncoDetect</h3>

  <p align="center">
     Unleashing the Power of Smart Diagnosis
    <br/>
    <br/>
  </p>
</p>



## About The Project

Objective:
The main goal of the project is to develop a binary classification model capable of predicting whether a tumor is cancerous or not based on input features such as tumor size and the number of nodes.

Components:

Logistic Regression Model:
The project implements a logistic regression model, a popular algorithm for binary classification tasks.
It utilizes the sigmoid activation function to transform input features into probabilities.

Regularized Logistic Regression:
The code includes a class for regularized logistic regression, which incorporates L2 regularization to prevent overfitting.

Training Data Generation:
Synthetic data is generated to simulate two classes of tumors: cancerous and non-cancerous.
The data includes features like tumor size and the number of nodes.

Gradient Descent Training:
The logistic regression model is trained using gradient descent.
The weights and bias are updated iteratively to minimize the logistic regression cost function.

User Interface:
A simple command-line interface is provided for user interaction.
Users can input tumor details, and the trained model predicts whether the tumor is cancerous or not.

Potential Enhancements:

Model Persistence:
Implement functionality to save and load trained model parameters, allowing reuse without retraining.

Real-world Validation:
Adapt the model for real-world medical data, considering ethical and regulatory standards.

Conclusion:
While the current implementation serves as a starting point for binary classification, further refinement, validation, and potentially the incorporation of more advanced machine learning techniques would be necessary for a robust and reliable cancer detection system in real-world scenarios.


## Built With

The project is built with the use of below technologies

* [Python](https://www.python.org/)
* [NumPy Library](https://numpy.org/)
* [Matplotlib Library](https://matplotlib.org/)
* [Supervised Machine Learning](https://www.javatpoint.com/supervised-machine-learning)
* [VS Code IDE](https://code.visualstudio.com/)

## Getting Started

To get started with the provided code for cancer detection using logistic regression, follow these steps:

### Prerequisites

1. Python Installation:
   Make sure you have Python installed on your system. You can download it from the official [Python website](https://www.python.org/downloads/).

2. Library Installation:
   Install the required libraries (NumPy and Matplotlib) by running the following command in your terminal or command prompt:
     ```bash
     pip install numpy matplotlib
     ```

### Installation

1. Download the Code:
   Copy and paste the provided code into a Python script or an integrated development environment (IDE) of your choice. Save the file with a `.py` extension.

2. Run the Code:
   Execute the script by running the following command in your terminal or command prompt:
     ```bash
     python your_script_name.py
     ```
     Replace `your_script_name.py` with the actual name of your Python script.

3. Interact with the User Interface:
•	The code will provide a simple command-line interface.
•	Follow the prompts to input tumor details (size and number of nodes).
•	The model will predict whether the tumor is cancerous or not.



## Usage


1. Run the Code:
•	Execute the Python script containing the provided code by running the following command in your terminal or command prompt:
   	  ```bash
    	 python your_script_name.py
     	```
•	Replace `your_script_name.py` with the actual name of your Python script.

2. Training:
•	The script generates synthetic training and testing data for cancer detection.
•	The logistic regression model is trained using gradient descent.

3. User Interface:
•	After training, the script provides a simple command-line interface for interactive testing.
•	Enter the details of a tumor when prompted:
•	Input the size of the tumor.
•	Input the number of nodes associated with the tumor.

4. Prediction:
•	The trained model predicts whether the given tumor is cancerous or not.
•	The result is displayed in the console.

5. Repeat or Exit:
•	After each prediction, you can choose to predict again or exit.
•	Enter `1` to predict again or `0` to exit.

6. Experimentation:
   Feel free to experiment with different tumor sizes and node numbers to observe how the model predictions change.

 Example Interaction:

```bash
-------------------CANCER DETECTION--------------------
ENTER 1 TO PREDICT CANCER AND 0 TO EXIT : 1
ENTER THE DETAILS OF TUMOR
ENTER SIZE OF TUMOR : 2.5
ENTER NUMBER OF NODES : 3.0
RESULT : TUMOR IS CANCEROUS
-------------------------------------------------------
ENTER 1 TO PREDICT CANCER AND 0 TO EXIT : 1
ENTER THE DETAILS OF TUMOR
ENTER SIZE OF TUMOR : 1.0
ENTER NUMBER OF NODES : 0.5
RESULT : TUMOR IS NOT CANCEROUS
-------------------------------------------------------
ENTER 1 TO PREDICT CANCER AND 0 TO EXIT : 0
```

By following these usage instructions, you can interact with the cancer detection model and observe its predictions based on the provided synthetic data.


## Contributing



### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Authors

* **Vinodraj V K** - *Computer Science Student* - [Vinodraj V K](https://github.com/VinodrajVK) - *Built the Complete Project*

## Acknowledgements

* [Vinodraj VK](https://github.com/VinodrajVK)

