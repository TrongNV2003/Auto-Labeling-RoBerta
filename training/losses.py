import torch
from torch import nn
from torch.nn import functional as F


class RankingLoss(nn.Module):
    def __init__(self, tau: float = 20):
        super(RankingLoss, self).__init__()

        self.tau = tau

# "qi,pi->qp" có nghĩa là nhân từng vector của query_vecs (kích thước q, d) với từng vector của doc_vecs (kích thước p, d) theo chiều d, để tạo ra ma trận cosine similarity kích thước (q, p).

    def forward(self, query_vecs: torch.Tensor, doc_vecs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine_similarity = torch.einsum("qi,pi->qp", 
                                         F.normalize(query_vecs, dim=-1, p=2), 
                                         F.normalize(doc_vecs, dim=-1, p=2))
        
# nhân với "tau" để tăng độ sắc nét của hàm cosine_similarity, giá trị nào lớn thì sẽ càng gần 1 hơn, giá trị nào nhỏ thì càng gần 0 hơn
# giúp tăng độ nhạy trong việc phân biệt các giá trị similarity nhỏ.        

        scores = cosine_similarity * self.tau
        
        scores = F.cross_entropy(scores, labels)

        return scores