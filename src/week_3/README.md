# Lecture
- Objective:
    - Backpropagation.
    - Regulations.

## Loss Function:
- A Loss function tells how good our current classifier is.

## Stochastic Gradient Descent with Mini-batch
- In large-scale applications, the training data can have on order of millions of examples. Hence, it seems wasteful to compute the full loss function over the entire training set in order to perform only a single parameter update.

## Step Size:
- Effect of step size: The gradient tells us the direction in which the function has the steepest rate of increase, but it does not tell us how far along this direction we should step.
- Choosing the step size (also called the learning rate) will become one of the most important hyperparameter settings in training a neural network

## Global vs Local Optimum
- Global Optimum: is the optimal solution among all possible solutions.
- Local Optimum: is the solution that is optimal (either maximal or minimal) within a neighboring set of candidate solutions.
- Saddle Point: a point on the surface of the graph of a function where the slopes (derivatives) in orthogonal directions are all zero but is not a local optimum.

## Convexity
- A function is convex if a line segment between any two points lies entirely "above" the graph.
- If loss function is convex, simple algorithms like gradie strong guarantee to converge to the global optimum

## SGD: 
- Gradient estimates can be very noisy.
- Use larger mini-batches to reduce stochastics.
- Computation per update is efficient and does not depend on the number of training samples. This allows convergence on extremely large datasets.

## Optimizers:
- SGD + Momentum:
    - Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations.
- RMSprop
- Adam:
    - Adam can be looked at as a combination of RMSprop and SGD with momentum. It uses a squared gradients to scale the learning rate like RMSprop and it takes advantage of momentum by using moving average of the gradient instead of gradient itself like SGD with momentum.

## Regularization:
- L1 & L2 Regularization: We add a component that will penalize large weights where lambda is the regularization parameter.
- Dropout: 
    - The probability of keeping each node is set at random. You only decide the threshold: a value that will determine if the node is keep or not.
    - Why does it works?
        - Dropout means that the neural network cannot rely on any input node, since each have a random probability of being removed. Therefore, the neural network will be reluctant to give high weights to certain features, because they might disappear. 
        - Consequently, the weights are spread across all features, making them smaller. This effectly shrinks the model and regularizes it.