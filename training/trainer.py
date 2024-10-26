from transformers import Trainer
from training.losses import RankingLoss

class ATrainer(Trainer):
    def __init__(self, model, loss_kwargs, **kwargs):
        super(ATrainer, self).__init__(model, **kwargs)

        self.loss_fn = RankingLoss(**loss_kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        representations, descriptions, labels = inputs
        reps_vecs = model(**representations)
        desc_vecs = model(**descriptions)
        loss = self.loss_fn(reps_vecs, desc_vecs, labels)

        outputs = (reps_vecs, desc_vecs)

        return (loss, outputs) if return_outputs else loss
    
