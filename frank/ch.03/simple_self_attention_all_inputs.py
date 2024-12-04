import torch

# We are going to use the following sentence for our example:
#
# Your journey starts with one step
#
# Sentence has been tokenized for us to save a step
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55],
    ]  # step     (x^6)
)

###############################
# Calculate Attention Scores  #
###############################
# size (6,6) comes from matrix multiplication
# input has shape (6,3)
# input.T has shape (3,6)
# resulting shape is rows of input and columns of input.T == (6, 6)
attention_scores = torch.empty(6, 6)
attention_scores = inputs @ inputs.T
print(f"Attention Scores: {attention_scores}")

###############################
# Calculate Attention Weights #
###############################
attention_weights = torch.softmax(attention_scores, dim=-1)
print(f"Attention Weights: {attention_weights}")

###############################
## Calculate Context Vectors ##
###############################
context_vectors = torch.zeros(6, 3)
context_vectors = attention_weights @ inputs

print(f"Context Vectors: {context_vectors}")
