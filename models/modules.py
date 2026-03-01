import torch
import torch.nn as nn
from torch.nn import LayerNorm
from typing import Tuple

class SoftMaxOpPyTorch:
    @staticmethod
    def max(x: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable soft-max and soft-argmax operations along a given dimension.

        :param x: torch.Tensor
            Input tensor of scores.
        :param dim: int
            The dimension along which to compute the soft-max/argmax.
        :return: Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - soft_max_value (logsumexp)
            - soft_argmax_value (softmax)
        """
        # Subtract the max for numerical stability
        max_x = torch.max(x, dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - max_x)
        Z = torch.sum(exp_x, dim=dim, keepdim=True)
        
        # 1. The soft-max value (logsumexp)
        soft_max_value = torch.log(Z) + max_x
        
        # 2. The soft-argmax value (softmax)
        soft_argmax_value = exp_x / Z
        
        return soft_max_value.squeeze(dim), soft_argmax_value
    
class StructuredDecoding(nn.Module):
    def __init__(self):
        super().__init__()
    
    def DVL(self, transition, log_emission, prior=None, return_log_likelihood=False):
        """
        Fully optimized, batched Differentiable Viterbi.
        """
        device, dtype = transition.device, transition.dtype
        eps = 1e-10
        transition = transition.clamp(min=eps)
        
        B, T, N = log_emission.shape

        # Normalize and convert to log-space (operations done once)
        transition = transition.clamp(min=eps) / transition.sum(dim=1, keepdim=True)
        log_transition_T = torch.log(transition).T # Transpose ONCE, outside the loop
        
        if prior is None:
            prior = torch.ones(N, device=device, dtype=dtype) / N
        else:
            prior = prior.clamp(min=eps) / prior.sum()
        log_prior = torch.log(prior)

        log_trellis = torch.zeros((B, T, N), device=device, dtype=dtype)
        soft_backpointers = torch.zeros((B, T - 1, N, N), device=device, dtype=dtype)

        log_trellis[:, 0] = log_prior[None, :] + log_emission[:, 0]

        for t in range(1, T):
            prev_log = log_trellis[:, t - 1] # Shape: [B, N]

            # Calculate scores with the pre-transposed transition matrix
            # This creates a score matrix of shape [B, N_to, N_from] directly
            scores = prev_log[:, None, :] + log_transition_T[None, :, :]
            # Maximize over the 'N_from' dimension (the last dimension)
            soft_max_vals, soft_argmax_vals = SoftMaxOpPyTorch.max(scores, dim=-1)

            # Update trellis (no transpose needed)
            log_trellis[:, t] = log_emission[:, t] + soft_max_vals
            # Store backpointers directly (no transpose needed)
            soft_backpointers[:, t - 1] = soft_argmax_vals

        final_probs = SoftMaxOpPyTorch.max(log_trellis[:, -1], dim=1)[1]
        soft_path = [final_probs]

        for t in reversed(range(T - 1)):
            # Backpointers are already in the correct [B, N_to, N_from] shape for bmm
            current = torch.bmm(soft_path[0].unsqueeze(1), soft_backpointers[:, t])
            soft_path.insert(0, current.squeeze(1))

        soft_path = torch.stack(soft_path, dim=1)

        if return_log_likelihood:
            log_likelihood = SoftMaxOpPyTorch.max(log_trellis[:, -1], dim=1)[0]
            return soft_path, log_likelihood
        return soft_path
    
    def forward(self, transition, log_emission, prior=None, return_log_likelihood=False):
        """
        Forward pass for the Structured Decoding module.
        
        Args:
            transition: Transition matrix of shape (action_dim, action_dim).
            log_emission: Log emission probabilities of shape (batch_size, time_horz, action_dim).
            prior: Prior probabilities of shape (action_dim,). If None, uniform distribution is used.
            return_log_likelihood: If True, returns the log likelihood of the path.
        
        Returns:
            soft_path: The most likely sequence of actions.
            log_likelihood: Log likelihood of the path if return_log_likelihood is True.
        """
        
        if return_log_likelihood:
            soft_path, log_likelihood = self.DVL(transition, log_emission, prior, return_log_likelihood)
            return soft_path, log_likelihood
        
        soft_path = self.DVL(transition, log_emission, prior, return_log_likelihood)        
        return soft_path

class ViterbiPlanNet(nn.Module):
    def __init__(self, hidden_size, dropout, mlp_ratio, action_dim, time_horizon=3):
        '''ViterbiPlanNet model initialization
        
        This class defines the ViterbiPlanNet. It consists of a transformer encoder
        to process the start and end states, and a projection MLP to generate log-emission
        probabilities for each time step. The structured decoding block then predicts the
        most likely sequence of actions using a differentiable Viterbi algorithm.
        
        Args:
            hidden_size:   Dimension of the hidden state.
            dropout:       Dropout rate for regularization.
            mlp_ratio:     Ratio for the MLP hidden layer size.
            action_dim:    Number of action classes.
            time_horizon:  Number of time steps in the procedure.
        '''

        super().__init__()
        
        # This is used for the START and END states
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=1,
                dim_feedforward=hidden_size * mlp_ratio,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            enable_nested_tensor=False,
            num_layers=1,
            norm=LayerNorm(hidden_size)
        )
        
        # This is used to project [start, end] to [start (t0), t1, t2, ..., end (tn)]
        self.projection_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, mlp_ratio * hidden_size),            # Project concatenated start/end
            nn.Sigmoid(),                                                   # Activation function
            nn.Dropout(dropout),                                            # Dropout for regularization
            nn.Linear(mlp_ratio * hidden_size, time_horizon * action_dim)   # Output for all time steps
        )
        self.log_sigmoid = nn.LogSigmoid()
        self.structured_decoding_block = StructuredDecoding()
        self.time_horizon = time_horizon

    def forward(self, x, transition, time_horz_test=None):
        '''
        x: [batch_size, 2, hidden_size]
        '''
        
        # x is [batch_size, time_horz, hidden_size]
        batch_size, _, _ = x.shape
        
        # [bathch_size, 2, hidden_size] -> [batch_size, 2, hidden_size]
        start_end_features = self.encoder(x)
        
        # [batch_size, 2, hidden_size] -> [batch_size, time_horz, action_dim]
        x = self.projection_mlp(start_end_features.reshape(batch_size, -1)).reshape(batch_size, self.time_horizon, -1)
        x = self.log_sigmoid(x)  # Log-sigmoid to get log-emission probabilities
        
        if time_horz_test is not None and self.time_horizon > time_horz_test:
            pad_length = self.time_horizon - time_horz_test + 1
            pad_sequence = x[:, -1:, :].repeat(1, pad_length, 1)  # Repeat the last state
            x = torch.cat([x[:, :1, :], x[:, 1:time_horz_test-1, :], pad_sequence], dim=1)
        # [batch_size, time_horz, action_dim] -> [batch_size, time_horz, action_dim]
        out_viterbi = self.structured_decoding_block(transition, x, prior=None, return_log_likelihood=False)
        
        return out_viterbi
