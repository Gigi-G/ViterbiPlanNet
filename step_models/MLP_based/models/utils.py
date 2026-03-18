import numpy as np

def img_text_similarlity(state_features, prompt_features, scale):
        ''' Compute the similarity between visual and linguistic features

        Args:
            state_features:     Input visual feature.   (batch, length, embedding_dim)
            prompt_features:    Input language feature. (batch, length, embedding_dim)
            scale:              Scale parameter.

        Returns:
            logits:             Similarity matrix.      (batch, length, length)
        '''

        embedding_dim = state_features.shape[-1]
        
        # flatten features
        state_features = state_features.reshape(-1, embedding_dim)
        prompt_features = prompt_features.reshape(-1, embedding_dim)

        # normalized features
        image_features = state_features / state_features.norm(dim=1, keepdim=True)
        text_features = prompt_features / prompt_features.norm(dim=1, keepdim=True)

        # similarity as logits
        logits = scale * image_features @ text_features.t()
        return logits


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count