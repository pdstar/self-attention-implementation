# Self Attention Implementation
Implement the scaled multiplicative attention calculation used by the multi-head attention class used in GPT models.

Here we implementing the _forward()_ function of the MultiHeadSelfAttention module in a minified GPT implementation. GPT refers to the "Generative Pre-trained Transformers" from OpenAI, originally described in "Improving language understanding with unsupervised learning". This specific GPT implementation is heavily inspired by the minGPT implementation provided by Andrej Karpathy.

## Key steps
* Multiple Q and K^T matrices, then divide by the square root of dK
* Set the mask fill value to negative infinity
* Multiply the intermediate matrix by V
* Run the checks successfully
