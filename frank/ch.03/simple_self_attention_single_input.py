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

# 2nd input token is the query
# "journey"
query = inputs[1]

##############################
# Calculate Attention Scores #
##############################


# torch.empty creates a tensor with uninitialized data
# memory is allocated for the tensor but values are not set which makes it faster than initializing with specific values
# inputs.shape == torch.Size[6, 3], shape[0] == 6
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    # calculate dot product for every element
    attn_scores_2[i] = torch.dot(x_i, query)

print(f"Attention Scores: {attn_scores_2}")


##############################
# Normalize Attention Scores #
##############################

# dim = 0  means each column will sum to 1
# dim = 1  means each row will sum to 1
# dim = -1 means the last dimension will sum to 1
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(f"Attention Weights: {attn_weights_2}")

##############################
### Compute Context Vector ###
##############################
context_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(f"Context Vector {context_vec_2}")
