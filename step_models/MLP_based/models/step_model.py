import torch
import torch.nn as nn
import torch.nn.functional as F

from models.state_encoder import StateEncoder

class StepModel(nn.Module):
    def __init__(
            self, 
            vis_input_dim,
            lang_input_dim,
            embed_dim, 
            time_horz, 
            args
        ):
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

        self.dropout = nn.Dropout(self.dropout)

        self.loss_state = nn.CrossEntropyLoss()
        self.loss_task = nn.MSELoss()
        
    def forward(
            self, 
            visual_features, 
            state_prompt_features, 
            actions, 
            tasks
        ):
        # forward network
        outputs = self.forward_once(
            visual_features, 
            state_prompt_features
        )

        # loss calculation
        labels, losses = self.forward_loss(outputs, actions, tasks)

        return outputs, labels, losses
    
    def forward_once(
            self, 
            visual_features, 
            state_prompt_features
        ):
        # Step 1: state encoding
        state_feat_encode, _, state_logits, _ = self.state_encoder(visual_features, state_prompt_features)
        
        # Step 1.2: task prediction
        task_logits = self.task_decoder(state_feat_encode.view(-1, 2 * self.embed_dim))

        # Collect outputs
        outputs = self.process_outputs(state_logits, task_logits)

        return outputs

    def forward_loss(self, outputs, actions, tasks):
        _, num_action = outputs["state_encode"].shape

        labels = self.process_labels(outputs, actions, tasks)
        
        losses = {}
        losses["state_encode"] = self.loss_state(
            outputs["state_encode"].reshape(-1, num_action), 
            labels["state"]
        )
        losses["task"] = self.loss_task(outputs["task"], labels["task_one_hot"])
        
        return labels, losses

    def process_outputs(self, 
            state_logits, 
            task_logits
        ):
        _, _, num_action = state_logits.shape

        outputs = {}
        outputs["state_encode"] = state_logits.reshape(-1, num_action)
        outputs["action"] = state_logits  # Output is just start and end action logits
        outputs["task"] = task_logits

        return outputs


    def process_labels(self, outputs, actions, tasks):
        labels = {}
        labels["state"] = actions[:, [0, -1]].reshape(-1)
        labels["action"] = actions[:, [0, -1]]  # Reference start and end actions
        labels["action_one_hot"] = F.one_hot(labels["action"], num_classes=outputs["state_encode"].shape[1]).float()
        labels["task"] = tasks.reshape(-1)
        labels["task_one_hot"] = F.one_hot(tasks, num_classes=outputs["task"].shape[1]).float()
        return labels

