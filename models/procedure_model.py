import torch
import torch.nn as nn
import torch.nn.functional as F

from models.state_encoder import StateEncoder
from models.modules import ViterbiPlanNet, SoftMaxOpPyTorch
from models.utils import viterbi_path

class ProcedureModel(nn.Module):
    def __init__(
            self, 
            vis_input_dim,
            lang_input_dim,
            embed_dim, 
            time_horz, 
            num_classes,
            args
        ):
        '''ProcedureModel initialization

        This class defines the ProcedureModel. It consists of a state encoder and a ViterbiPlanNet for action prediction.
        The state encoder encodes the visual and language features, while the ViterbiPlanNet predicts the actions based on the encoded states.

        Args:
            vis_input_dim:  dimension of visual features.
            lang_input_dim: dimension of language features.
            embed_dim:      dimension of embedding features.
            time_horz:      time horizon.
            num_classes:    number of action classes.
            args:           arguments from parser.
        '''
        super().__init__()

        self.mlp_ratio = args.mlp_ratio
        self.dropout = args.dropout
        self.dataset = args.dataset
        self.time_horz = time_horz
        self.embed_dim = embed_dim

        self.state_encoder = StateEncoder(
            vis_input_dim,
            lang_input_dim,
            embed_dim, 
            dropout = 0.4
        )
        
        self.task_decoder = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(embed_dim, args.num_tasks)
        )

        self.viterbi_plan = ViterbiPlanNet(
            embed_dim,
            dropout = self.dropout,
            mlp_ratio = self.mlp_ratio,
            action_dim = num_classes,
            time_horizon = self.time_horz)

        self.dropout = nn.Dropout(self.dropout)

        self.loss_action = nn.MSELoss()
        self.loss_state = nn.CrossEntropyLoss()
        self.loss_task = nn.MSELoss()
        
    def forward(
            self, 
            visual_features, 
            state_prompt_features, 
            actions, 
            transition_matrix_torch,
            tasks,
            transition_matrix=None,
            time_horz_test=None
        ):
        '''Forward pass and loss calculation

        This function calls forward_once() to get the outputs, and then calls 
        forward_loss() to get processed labels and losses.

        Args:
            visual_features:            Visual observations of procedures.  [batch_size, time_horizon, 2, vis_input_dim]
            state_prompt_features:      Descriptions of before and after state of all actions. [num_action, num_prompts, lang_input_dim]
            actions:                    Ground truth actions.     [batch_size, time_horizon]
            transition_matrix_torch:    Transition matrix for Viterbi decoding.  [num_action, num_action]
            tasks:                      Ground truth tasks.       [batch_size]
            transition_matrix:          Transition matrix for Viterbi decoding.  [num_action, num_action]
            time_horz_test:             Time horizon for testing. If None, use training time horizon.

        Returns:
            outputs: Dictionary of outputs.
            labels:  Dictionary of labels.
            losses:  Dictionary of losses.
        '''

        # forward network
        outputs = self.forward_once(
            visual_features, 
            state_prompt_features, 
            transition_matrix_torch,
            time_horz_test
        )

        batch_size, T = actions.shape
        action_logits = outputs["action"].reshape(batch_size, self.time_horz, -1)

        # viterbi decoding
        if transition_matrix is not None:
            pred_viterbi = []
            for i in range(batch_size):
                viterbi_rst = viterbi_path(transition_matrix, action_logits[i].permute(1, 0).detach().cpu().numpy())
                pred_viterbi.append(torch.from_numpy(viterbi_rst))
            pred_viterbi = torch.stack(pred_viterbi).cuda()
        else:
            pred_viterbi = None
        outputs["pred_viterbi"] = pred_viterbi

        # loss calculation
        labels, losses = self.forward_loss(outputs, actions, tasks)

        return outputs, labels, losses
    
    def forward_once(
            self, 
            visual_features, 
            state_prompt_features, 
            transition_matrix_torch,
            time_horz_test=None
        ):
        '''Forward pass

        This function calls the state encoder and DVL to get the outputs.

        Args:
            visual_features:            Visual observations of procedures.  [batch_size, time_horizon, 2, vis_input_dim]
            state_prompt_features:      Descriptions of before and after state of all actions.     [num_action, num_prompts, lang_input_dim]
            transition_matrix_torch:    Transition matrix for Viterbi decoding.  [num_action, num_action]
        
        Returns:
            outputs:                    Dictionary of outputs.
        '''
        
        # Step 1: state encoding (SCHEMA implementation)
        state_feat_encode, _, state_logits, _ = self.state_encoder(visual_features, state_prompt_features)
        
        # Step 1.2: task prediction
        task_logits = self.task_decoder(state_feat_encode.view(-1, 2 * self.embed_dim))

        # Step 2: action prediction
        action_logits = self.viterbi_plan(
            state_feat_encode,
            transition_matrix_torch,
            time_horz_test=time_horz_test
        )
        
        # Step 2.2: Calculate the action logits based only on DVL
        viterbi_output = self.differentiable_viterbi(transition_matrix_torch, action_logits, return_log_likelihood=False)

        # Collect outputs
        outputs = self.process_outputs(state_logits, action_logits, viterbi_output, task_logits)

        return outputs

    def forward_loss(self, outputs, actions, tasks):
        '''Loss calculation

        This function calculates the losses for state encoding and action prediction.

        Args:
            outputs:    Dictionary of outputs.
            actions:    Ground truth actions.
        
        Returns:
            labels:     Dictionary of processed labels.
            losses:     Dictionary of losses.
        '''

        _, num_action = outputs["action"].shape

        labels = self.process_labels(outputs, actions, tasks)
        
        losses = {}
        losses["state_encode"] = self.loss_state(
            outputs["state_encode"].reshape(-1, num_action), 
            labels["state"]
        )
        losses["action"] = self.loss_action(outputs["action"].reshape(-1, num_action), labels["action_one_hot"].reshape(-1, num_action))
        losses["task"] = self.loss_task(outputs["task"], labels["task_one_hot"])
        
        return labels, losses


    def process_outputs(self, 
            state_logits, 
            action_logits,
            viterbi_output,
            task_logits,
            pred_viterbi = None,
        ):
        '''Process outputs

        This function processes the outputs from the forward pass.

        Args:
            state_logits:          Similarity between visual and linguistic features for start and end states.  [batch_size, 2, num_action]
            action_logits:         Predicted action logits.  [batch_size, time_horizon, num_action]
            viterbi_output:        Viterbi output logits.    [batch_size, time_horizon, num_action]
            task_logits:           Task prediction logits.   [batch_size, num_task]
            pred_viterbi:          Predicted actions using viterbi decoding.    [batch_size, time_horizon]

        Returns:
            outputs: Dictionary of processed outputs.
        '''

        _, _, num_action = state_logits.shape

        outputs = {}
        outputs["state_encode"] = state_logits.reshape(-1, num_action)
        outputs["action"] = action_logits.reshape(-1, num_action)
        outputs["viterbi_logits"] = viterbi_output
        outputs["pred_viterbi"] = pred_viterbi
        outputs["task"] = task_logits

        return outputs


    def process_labels(self, outputs, actions, tasks):
        labels = {}
        labels["state"] = actions[:, [0, -1]].reshape(-1)
        labels["action"] = actions.reshape(-1)
        labels["action_one_hot"] = F.one_hot(actions, num_classes=outputs["action"].shape[1]).float()
        labels["task"] = tasks.reshape(-1)
        labels["task_one_hot"] = F.one_hot(tasks, num_classes=outputs["task"].shape[1]).float()
        return labels
    
    
    def differentiable_viterbi(
            self, 
            transition, 
            emission, 
            prior=None, 
            return_log_likelihood=False
        ):
        """Differentiable Viterbi algorithm using PyTorch.
        
        This function computes the soft Viterbi path through a sequence of emissions.
        
        Args:
            transition: Transition probabilities of shape [N_from, N_to].
            emission: Emission probabilities of shape [B, T, N].
            prior: Prior probabilities of shape [N], optional.
            return_log_likelihood: If True, returns the log likelihood of the final state.
        Returns:
            soft_path: Soft Viterbi path of shape [B, T, N].
            log_likelihood: Log likelihood of the final state if return_log_likelihood is True.
        """
        device, dtype = transition.device, transition.dtype
        eps = 1e-10
        transition = transition.clamp(min=eps)
        
        B, T, N = emission.shape

        # Normalize and convert to log-space (operations done once)
        transition = transition.clamp(min=eps) / transition.sum(dim=1, keepdim=True)
        log_transition_T = torch.log(transition).T # Transpose ONCE, outside the loop
        log_emission = torch.log(emission.clamp(min=eps))
        
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

        # --- BACKWARD PASS ---
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

