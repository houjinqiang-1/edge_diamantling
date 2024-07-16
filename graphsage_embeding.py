from torch import nn
import torch
class SupervisedGraphSage(nn.Module):

    def __init__(self, embed_dim, lay_num, device):
        super(SupervisedGraphSage, self).__init__()
        #        self.enc = [enc1,enc2]#,enc3]
        self.embed_dim = embed_dim
        self.lay_num = lay_num
        self.device = device

    def forward(self, nodes, features, adj_lists, Encoder1, Encoder2, layerNodeAttention_weight,
                MeanAggregator):
        embeds1 = Encoder2(nodes, Encoder1(nodes, features[0], adj_lists[0], MeanAggregator).t(), adj_lists[0],
                           MeanAggregator)
        embeds2 = Encoder2(nodes, Encoder1(nodes, features[1], adj_lists[1], MeanAggregator).t(), adj_lists[1],
                           MeanAggregator)
        embeds = [embeds1.t(), embeds2.t()]
        result = torch.zeros(self.lay_num, nodes.size, self.embed_dim)
        for l in range(self.lay_num):
            result_temp = layerNodeAttention_weight(embeds, nodes, l)
            result[l] = result_temp
        # result = embeds1.t()
        return result
