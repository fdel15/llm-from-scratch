import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2   = inputs[1]       # second input element
d_in  = inputs.shape[1] # input embedding size
d_out = 2               # output embedding size

##############################
# Initialize weight matrixes #
##############################
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(f"W_query shape: {W_query.shape}")

##############################
## Compute weight matrixes  ##
##############################

query_2 = x_2 @ W_query
key_2   = x_2 @ W_key
value_2 = x_2 @ W_value

print(f"query_2: {query_2}")

#################################
# Input from 3D to 2D embedding #
#################################
# [6,3] @ [3,2] == 6,2
keys   = inputs @ W_key
values = inputs @ W_value

print(f"Keys Shape: {keys.shape}")

############################
# Compute Attention Scores # 
############################

attn_score_2 = query_2 @ keys.T
print(f"Attention Score: {attn_score_2}")

#############################
# Compute Attention Weights # 
#############################

d_k = keys.shape[1] # == 2
attn_weights_2 = torch.softmax(attn_score_2 / (d_k**0.5), dim=-1)
print(f"Attention Weights: {attn_weights_2}")

#############################
# Compute Context Vector    # 
#############################
# [1,6] @ [6,2] == [1,2]
context_vec_2 = attn_weights_2 @ values
print(f"Context Vector: {context_vec_2}")