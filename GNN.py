from Global import *


class GNN(Module):
  def __init__(self, conv_dims: list, fully_connected_for_head_dims: list, fully_connected_for_tail_dims: list, x_feature:str, dropout: dict, embedding_dims: tuple=None)-> NoReturn:
    super(GNN, self).__init__()
    self.conv_dropout = Dropout(p=dropout["conv"])
    self.emb_dropout = Dropout(p=dropout["emb"]) 
    self.fc_dropout = Dropout(p=dropout["fc"])
    self.x_feature = x_feature
    self.num_relations = Global.NUM_RELATIONS.value
    self.relation_embedder = Embedding(self.num_relations, fully_connected_for_tail_dims[-1])
    if x_feature == 'one-hot':
      self.entity_embedder = Embedding(embedding_dims[0], embedding_dims[1])
      first_conv_layer = [RGCNConv(embedding_dims[1], conv_dims[0], self.num_relations)]
    elif x_feature == 'identity':
      first_conv_layer = [RGCNConv(1, conv_dims[0], self.num_relations)]
    conv_list = first_conv_layer + \
                                [
                                  RGCNConv(conv_dims[i], conv_dims[i+1], self.num_relations)
                                  for i in range(len(conv_dims[:-1]))
                                ]
  
    fully_connected_list_for_tail = [
                                      Linear(conv_dims[-1], fully_connected_for_tail_dims[0]) 
                                    ] +\
                                    [
                                     Linear(fully_connected_for_tail_dims[i], fully_connected_for_tail_dims[i + 1])
                                     for i in range(len(fully_connected_for_tail_dims[:-1]))
                                    ]

    fully_connected_list_for_head = [
                                      Linear(conv_dims[-1], fully_connected_for_head_dims[0]) 
                                    ] +\
                                    [
                                     Linear(fully_connected_for_head_dims[i], fully_connected_for_head_dims[i + 1])
                                     for i in range(len(fully_connected_for_head_dims[:-1]))
                                    ]
    

    #graph conv layers
    self.conv_layers = ModuleList(conv_list)

    #fully connected dense layers
    self.head_fc_layers = ModuleList(fully_connected_list_for_head)
    self.tail_fc_layers = ModuleList(fully_connected_list_for_tail)
    

  def reset_parameters(self):
    self.entity_embedder.reset_parameters()
    self.relation_embedder.reset_parameters()
    for conv in self.conv_layers:
        conv.reset_parameters()
    for fc in self.head_fc_layers:
        fc.reset_parameters()
    for fc in self.tail_fc_layers:
        fc.reset_parameters()


  def forward(self, data: Data) -> torch.Tensor:
    edge_index = data.edge_index_messaging
    x = data.x
    edge_type = data.edge_type_messaging

    ####################################### One-Hot Entity Encoder #######################################
    if self.x_feature == 'one-hot':
      x = self.entity_embedder(x).reshape(self.entity_embedder.weight.shape[0], -1)
      
    ############################################## Identity Encoder ################################################
    elif self.x_feature == 'identity':
      x = torch.ones(x.shape[0], 1)
      if self.training:
        x = self.emb_dropout(x)
    ####################################### RGCN Encoder #######################################
    for conv in self.conv_layers[:-1]:
      x = conv(x, edge_index=edge_index, edge_type=edge_type)
      x = F.relu(x)
      if self.training:
        x = self.conv_dropout(x)
    x = self.conv_layers[-1](x, edge_index, edge_type)
    if self.training:
      x = self.conv_dropout(x)
    
    ####################################### Decoder #######################################
    positive_heads = x[data.edge_index_supervision[0]]
    positive_tails = x[data.edge_index_supervision[1]]
    positive_relations = self.relation_embedder(data.edge_type_supervision) # relation encoder

    negative_heads = x[data.edge_index_negative[0]]
    negative_tails = x[data.edge_index_negative[1]]
    negative_relations = self.relation_embedder(data.edge_type_negative)

    heads = torch.cat((positive_heads, negative_heads))
    relations = torch.cat((positive_relations, negative_relations))
    tails = torch.cat((positive_tails, negative_tails))
    perm_index = torch.randperm(heads.shape[0])
    heads = heads[perm_index]
    tails = tails[perm_index]
    relations = relations[perm_index]

    if self.training:
      relations = self.emb_dropout(relations)

    for head_fc, tail_fc in zip(self.head_fc_layers[:-1], self.tail_fc_layers[:-1]):
      heads = head_fc(heads)
      heads = F.relu(heads)
      tails = tail_fc(tails)
      tails = F.relu(tails)

      if self.training:
        heads = self.conv_dropout(heads)
        tails = self.conv_dropout(tails)

    heads = self.head_fc_layers[-1](heads)
    tails = self.tail_fc_layers[-1](tails)

    if self.training:
      heads = self.conv_dropout(heads)
      tails = self.conv_dropout(tails)
    # let f be the fully connected transform function for heads, and 
    # let g be the fully connected transform function for tails:
        # prior to the fully connected transforms:
        #     score(head, relation, tail) = <head, relation, tail> = head . relation . tail
        #     score(tail, relation, head) = <tail, relation, head> = tail . relation . head
        #     ===> score(head, relation, tail) = score(tail, relation, head) essentially holds
        # now with the fully connected transforms:
        #     score(head, relation, tail) = <f(head), relation, g(tail)> = f(head) . relation . g(tail)
        #     score(tail, relation, head) = <f(tail), relation, g(head)> = f(tail) . relation . g(head)
        #     ===> score(head, relation, tail) = score(tail, relation, head) does not necessarily holds

    scores = (heads * relations * tails).sum(dim=1)
    original_index = torch.sort(perm_index)[1]
    return scores[original_index[:positive_heads.shape[0]]], scores[original_index[positive_heads.shape[0]:]]
