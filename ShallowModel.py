from Global import *


#@title Shallow Model Class

class ShallowModel(Module):
  def __init__(self, x_feature:str, dropout: dict, embedding_dims: tuple=None)-> NoReturn:
    super(ShallowModel, self).__init__()
    # self.conv_dropout = Dropout(p=dropout["conv"])
    self.emb_dropout = Dropout(p=dropout["emb"]) 
    # self.fc_dropout = Dropout(p=dropout["fc"])
    self.x_feature = x_feature
    self.num_relations = Global.NUM_RELATIONS.value
    self.relation_embedder = Embedding(self.num_relations, embedding_dims[-1])
    if x_feature == 'one-hot':
      self.entity_embedder = Embedding(embedding_dims[0], embedding_dims[1])

  def reset_parameters(self):
    self.entity_embedder.reset_parameters()
    self.relation_embedder.reset_parameters()


  def forward(self, data: Data) -> torch.Tensor:
    edge_index = data.edge_index_messaging
    x = data.x
    edge_type = data.edge_type_messaging

    ####################################### One-Hot Entity Encoder #######################################
    if self.x_feature == 'one-hot':
      x = self.entity_embedder(x).reshape(self.entity_embedder.weight.shape[0], -1)
    ############################################## Identity Encoder ################################################
    elif self.x_feature == 'identity':
      x = torch.ones(x.shape[0], 1).to(Global.DEVICE.value)
      
    if self.training:
      x = self.emb_dropout(x)
    
    ####################################### Decoder #######################################
    positive_heads = x[data.edge_index_supervision[0]]
    positive_tails = x[data.edge_index_supervision[1]]
    positive_relations = data.edge_type_supervision

    negative_heads = x[data.edge_index_negative[0]]
    negative_tails = x[data.edge_index_negative[1]]
    negative_relations = data.edge_type_negative

    heads = torch.cat((positive_heads, negative_heads))
    relations = torch.cat((positive_relations, negative_relations))
    relations = self.relation_embedder(relations)
    tails = torch.cat((positive_tails, negative_tails))

    perm_index = torch.randperm(heads.shape[0])
    heads = heads[perm_index]
    tails = tails[perm_index]
    relations = relations[perm_index]

    if self.training:
      relations = self.emb_dropout(relations)

    scores = (heads * relations * tails).sum(dim=1)
    original_index = torch.sort(perm_index)[1]
    return scores[original_index[:positive_heads.shape[0]]], scores[original_index[positive_heads.shape[0]:]]
