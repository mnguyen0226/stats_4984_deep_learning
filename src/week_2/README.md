# Lecture
- Objective:
    - Intro to PyTorch with simple linear regression model.
    - Limitation of linear model.
    - Multi-Layer Perception.

## PyTorch
- PyTorch is an open-source deep learning framework that's known for its flexibility and ease-of-use. This is enabled in part by its compatibility with the popular Python high-level programming language favored by machine learning developers and data scientists.
- A torch.Tensor is a multi-dimensional array containing elements of a single data type.
- Use cuda.is_available() to find out if you have a GPU at your disposal and set your device accordingly
- Autograd:
    - A gradient is a partial derivative because one computes it with respect to w.r.t a single parameters.
    - Autograd is PyTorch's automatic differentiation package. Use backward() method for gradient calculation.
- Activation Function:
    - ReLU, eLU, Sigmoid, Tanh, Leaky ReLU, Maxout, ELU,...
- Loss Function: A loss function tells how good our current classifier is.

# Deep Learning Q & A
- `Q1:` Can hidden layer in linear fully connect layer can be other number besides 64? Yes.
- `Q2:` Why the last layer of classifier is a log-softmax function? Log-softmax function is used to do softmax transformation so that we won't get negative number (log probability). We assume that the labels is 1 hot encoder.
- `Q3:` Where is the required_grad function? During training, we can call net.parameters() to indicate we want the optimizer to do back-prop and update the parameters.
- `Q4:` Why the loss is the log-likelihood? Likelihood allows you to measures how accuracy the labels vs the prediction is. "Log" makes the likelihood to be calculated easier and still get valid results since we are trying to max (or min for negative likelihood).
    - More on this: https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/maximum-likelihood.html