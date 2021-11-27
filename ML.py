from Global import *
from GNN import GNN
from Graph import get_psuedo_negative_entities, graph_data_maker


def train(model: GNN, graph: Data, optimizer: torch.optim, loss_fn:torch.nn.modules.loss) -> tuple((torch.float, torch.Tensor, torch.Tensor)):
  optimizer.zero_grad()
  model.train()
  positive, negative = model(graph)
  scores = torch.cat((positive, negative))
  labels = torch.ones(graph.edge_index_supervision.shape[1]).to(Global.DEVICE.value)
  loss = loss_fn(positive, negative, labels)
  loss.backward()
  optimizer.step()
  return loss.item(), positive, negative


@torch.no_grad()
def evaluation_rank(model, eval_set, messaging_set):
  h_pred_top10 = list()
  t_pred_top10 = list()

  for eval_triplet in tqdm(eval_set):
    #use tqdm
    head = eval_triplet[Global.HEAD_INDEX.value].item()
    relation = eval_triplet[Global.RELATION_INDEX.value].item()
    tail = eval_triplet[Global.TAIL_INDEX.value].item()

    corrupted_tails = get_psuedo_negative_entities(
        entity=head, 
        corrupt_at=Global.TAIL_INDEX.value, 
        kg_sorted=kg_sorted_heads
    )
    
    num_psuedo_negative_triples = corrupted_tails.shape[0]
    psuedo_negative_triplets_corrupted_tail = torch.vstack(
        (
            torch.ones(num_psuedo_negative_triples).to(Global.DEVICE.value).type(torch.long) * head,
            torch.ones(num_psuedo_negative_triples).to(Global.DEVICE.value).type(torch.long) * relation,
            corrupted_tails
        )
    ).t().to(Global.DEVICE.value)

    corrupted_heads = get_psuedo_negative_entities(
        entity=tail, 
        corrupt_at=Global.HEAD_INDEX.value, 
        kg_sorted=kg_sorted_tails
    )

    num_psuedo_negative_triples = corrupted_heads.shape[0]
    psuedo_negative_triplets_corrupted_head = torch.vstack(
        (
            corrupted_heads,
            torch.ones(num_psuedo_negative_triples).to(Global.DEVICE.value).type(torch.long) * relation,
            torch.ones(num_psuedo_negative_triples).to(Global.DEVICE.value).type(torch.long) * tail
        )
    ).t().to(Global.DEVICE.value)

    # eval_triplet: (h, r, t)
    # psuedo_negative_triplets_head: (h′, r, t) for all h′
    # psuedo_negative_triplets_tail: (h, r, t′) for all t′

    # train_set being the messaging graph, calculate the score for (h, r, t)
    # train_set being the messaging graph, calculate the scores for all (h′, r, t)
    # train_set being the messaging graph, calculate the scores for all (h, r, t′)

    graph_data_for_object_tail = graph_data_maker(
      messaging=messaging_set,
      supervision=eval_triplet.reshape(1, 3),
      negative_samples=psuedo_negative_triplets_corrupted_head,
      x=features.to(Global.DEVICE.value),
      x_feature='one-hot'
    )

    graph_data_for_object_head = graph_data_maker(
            messaging=messaging_set,
            supervision=eval_triplet.reshape(1, 3),
            negative_samples=psuedo_negative_triplets_corrupted_tail,
            x=features.to(Global.DEVICE.value),
            x_feature='one-hot'
    )

    model.eval()
    scores_for_object_tail = torch.cat(model(graph_data_for_object_tail))

    sorted_by_scores_for_object_tail_indcs = torch.sort(scores_for_object_tail, descending=True)[1]

    head_objects = torch.cat((graph_data_for_object_tail.edge_index_supervision[0], graph_data_for_object_tail.edge_index_negative[0]))
    top10_heads = head_objects[sorted_by_scores_for_object_tail_indcs[:10]]
    h_pred_top10.append(top10_heads)

    model.eval()
    scores_for_object_head = torch.cat(model(graph_data_for_object_head))

    sorted_by_scores_for_object_head_indcs = torch.sort(scores_for_object_head, descending=True)[1]

    tail_objects = torch.cat((graph_data_for_object_head.edge_index_supervision[1], graph_data_for_object_head.edge_index_negative[1]))
    top10_tails = tail_objects[sorted_by_scores_for_object_head_indcs[:10]]
    t_pred_top10.append(top10_tails)

    # print('')
    ############################################################################## 1
    # if False:
    #   hit_index = (
    #             tail_objects[sorted_by_scores_for_object_head_indcs[:500]] == eval_triplet[2]
    #           ).nonzero(as_tuple=True)[0] + 1
    #   if hit_index.shape[0]:
    #     print(f'Tail ranked at {hit_index.item()}')
    #     print('-' * 30)
    # ############################################################################### 2

    # ############################################################################### 3
    # hit_index = (
    #             head_objects[sorted_by_scores_for_object_tail_indcs[:500]] == eval_triplet[0]
    #           ).nonzero(as_tuple=True)[0] + 1
    # if hit_index.shape[0]:
    #   print(f'Head ranked at {hit_index.item()}')
    #   print('-' * 30)
    ############################################################################## 4
    
  model.train()
  return torch.stack(h_pred_top10), torch.stack(t_pred_top10)


def evaluate_hits_at_10(model: GNN, mode:str) -> torch.float:
  if mode == 'validation':
    head_objects_top10, tail_objects_top10 = evaluation_rank(model, val_set, train_set)
    return evaluator.eval(
            head_objects_top10,
            tail_objects_top10,
            val_set,
            False
          )
  elif mode == 'testing':
    head_objects_top10, tail_objects_top10 = evaluation_rank(model, test_set, torch.cat((train_set, val_set), dim=0))
    return evaluator.eval(
            head_objects_top10,
            tail_objects_top10,
            test_set,
            False
          )
