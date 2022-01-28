# Deep Learning Tips
- `Q1:` Can hidden layer in linear fully connect layer can be other number besides 64? Yes
- `Q2:` Why the last layer of classifier is a log-softmax function? Log-softmax function is used to do softmax transformation so that we won't get negative number (log probability). We assume that the labels is 1 hot encoder.
- `Q3:` Where is the required_grad function? During training, we can call net.parameters() to indicate we want the optimizer to do back-prop and update the parameters.
