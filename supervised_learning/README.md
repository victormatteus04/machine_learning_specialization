# Supervised Learning

Supervised Learning: The model is trained on a labeled dataset. The model learns to map the input to the output. It learns from being given the correct answer. Some examples would be linear regression, logistic regression, and neural networks. Spam filtering, image recognition, and speech recognition are some examples of supervised learning.

## Regression Problem

A regression problem in machine learning is a type of supervised learning task where the goal is to predict a continuous numeric output based on input features. Here’s a breakdown to help you understand it better:

### Key Concepts:
1. **Input Variables (Features):** These are the variables you use to predict the output. For example, in predicting house prices, features could include the size of the house, the number of bedrooms, and the location.

2. **Output Variable (Target):** This is the value you're trying to predict. In regression, it's always a continuous value (like predicting a price, temperature, or age).

3. **Model:** A regression model is a mathematical representation that maps input features to the output. Common regression models include Linear Regression, Polynomial Regression, Ridge Regression, and more.

4. **Loss Function:** The model's predictions are compared against the actual values using a loss function (like Mean Squared Error or Mean Absolute Error). The model's goal is to minimize this loss, making the predictions as close as possible to the actual values.

5. **Training:** During training, the model learns the relationship between the features and the target by adjusting its parameters to minimize the loss function.

6. **Testing/Validation:** After training, the model's performance is evaluated on a separate set of data (testing or validation data) to see how well it generalizes to unseen data.

### Example:
Imagine you're trying to predict the price of a house based on its size. The dataset might look like this:

| Size (sq ft) | Price ($) |
|--------------|-----------|
| 1500         | 300,000   |
| 2000         | 400,000   |
| 2500         | 500,000   |

A linear regression model would try to find the best line (a function) that predicts the price based on the size. The function might look like:

$$
\text{Price} = \text{Intercept} + \text{Slope} \times \text{Size}
$$

### Types of Regression:
- **Linear Regression:** Assumes a straight-line relationship between the input variables and the output.
- **Polynomial Regression:** Fits a polynomial equation to the data, useful when the relationship is non-linear.
- **Ridge/Lasso Regression:** These are linear models that include regularization to prevent overfitting by penalizing large coefficients.

### Metrics to Evaluate Regression:
- **Mean Squared Error (MSE):** The average of the squared differences between predicted and actual values.
- **R-squared (R²):** Indicates how well the model explains the variability of the output. A value close to 1 means the model explains the data well.

### Application:
Regression is widely used in forecasting, financial modeling, risk assessment, and any scenario where you need to predict a continuous outcome.

## Classification Problem

Classification is another fundamental type of supervised learning task in machine learning, where the goal is to predict a discrete label or category for a given set of input features. Here’s how it differs from regression and what it involves:

### Key Concepts:
1. **Input Variables (Features):** Like in regression, these are the variables used to predict the output. For instance, in an email spam detection system, features might include the frequency of certain words, the length of the email, or the presence of links.

2. **Output Variable (Target):** In classification, the target is a discrete label or category. For example, in a binary classification problem, the output could be "spam" or "not spam." In a multi-class classification, the output could be more than two categories (e.g., types of flowers: "setosa," "versicolor," "virginica").

3. **Model:** A classification model learns to map input features to one of the possible categories. Common models include Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVM), and Neural Networks.

4. **Loss Function:** The loss function in classification measures how far the model’s predicted class probabilities are from the actual classes. Common loss functions include Cross-Entropy Loss for multi-class problems and Log Loss for binary classification.

5. **Training:** During training, the model learns to assign the correct class to the input data by adjusting its parameters to minimize the loss function.

6. **Testing/Validation:** The model's performance is evaluated on a separate dataset to check how well it predicts the correct classes on new, unseen data.

### Example:
Imagine you're trying to classify emails as "spam" or "not spam." The dataset might look like this:

| Email Features            | Label     |
|----------------------------|-----------|
| Contains "free", >1000 words | Spam      |
| Contains "meeting", <500 words | Not Spam |

A Logistic Regression model might be used to predict the probability that an email belongs to the "spam" class. Based on the probability, the email is classified as "spam" or "not spam."

### Types of Classification:
- **Binary Classification:** Only two classes. For example, classifying emails as "spam" or "not spam."
- **Multi-Class Classification:** More than two classes. For example, classifying types of animals as "cat," "dog," "bird," etc.
- **Multi-Label Classification:** Each instance can belong to multiple classes simultaneously. For example, tagging a blog post with multiple categories like "technology," "AI," and "business."

### Metrics to Evaluate Classification:
- **Accuracy:** The proportion of correctly classified instances out of the total instances.
- **Precision:** The proportion of true positive predictions out of all positive predictions. Useful when the cost of false positives is high.
- **Recall (Sensitivity):** The proportion of true positive predictions out of all actual positives. Useful when the cost of false negatives is high.
- **F1-Score:** The harmonic mean of precision and recall, providing a balance between them.
- **Confusion Matrix:** A table showing the performance of the model by displaying the true positives, false positives, true negatives, and false negatives.

### Application:
Classification is used in various real-world applications, such as spam detection, disease diagnosis, image recognition, sentiment analysis, and credit scoring.

### Comparison with Regression:
- **Output:** Classification predicts discrete categories, while regression predicts continuous values.
- **Models:** Some algorithms can be adapted for both tasks, like Decision Trees, but their objectives and loss functions differ.
- **Metrics:** The evaluation metrics in classification (like accuracy, precision) differ from those in regression (like MSE, R²).
