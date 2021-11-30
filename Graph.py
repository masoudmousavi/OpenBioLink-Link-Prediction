from Global import *
from Utils import find_start_and_end_indcs, check_negative_samples


def get_psuedo_negative_entities(entity, corrupt_at, kg_sorted):
  if corrupt_at == Global.HEAD_INDEX.value:
    entity_type = Global.TAIL_INDEX.value 
  elif corrupt_at == Global.TAIL_INDEX.value:
    entity_type = Global.HEAD_INDEX.value

  indcs = find_start_and_end_indcs(entity, entity_type, kg_sorted=kg_sorted)
  
  if indcs is not None:
    fact_triplets_entities = kg_sorted[indcs][:, corrupt_at]
    features_copy = features.detach().clone()
    features_copy[fact_triplets_entities] = -1
    non_negative_mask = features_copy >= 0
    ret = torch.nonzero(non_negative_mask).reshape(-1)
    return ret
  else:
    return features.to(Global.DEVICE.value)


def graph_data_maker(messaging: torch.Tensor, supervision: torch.Tensor, negative_samples: torch.Tensor, x:torch.Tensor, x_feature: str=Global.FEATURE_ENG.value, check_for_correctness: bool=False) -> Data:
  relation_idx = Global.RELATION_INDEX.value
  head_idx = Global.HEAD_INDEX.value
  tail_idx = Global.TAIL_INDEX.value
  graph_data = Data(
        x=x.reshape(-1, 1),
        edge_index_messaging=messaging[:, (head_idx, tail_idx)].t().contiguous(),
        edge_type_messaging=messaging[:, relation_idx],
        edge_index_supervision=supervision[:, (head_idx, tail_idx)].t().contiguous(),
        edge_type_supervision=supervision[:, relation_idx],
        edge_index_negative=negative_samples[:, (head_idx, tail_idx)].t().contiguous(),
        edge_type_negative=negative_samples[:, (relation_idx)]
    )
  return graph_data


def mini_batch_maker(messaging, supervision, candidates, x_feature='one-hot'):
  heads = supervision[:, Global.HEAD_INDEX.value]
  relations = supervision[:, Global.RELATION_INDEX.value]
  tails = supervision[:, Global.TAIL_INDEX.value]

  ct_size = supervision.shape[0] // 2
  ch_size = supervision.shape[0] - ct_size
  while 1:
    while 1:
      c_tails = torch.unique((candidates.shape[0] * torch.rand(ct_size)).type(torch.int32))
      if c_tails.shape[0] == ct_size:
        break
    negative_samples_corrupted_tails = torch.vstack(
        (
            heads[: ct_size], 
            relations[: ct_size], 
            c_tails.to(Global.DEVICE.value)
            # torch.multinomial(
            #     candidates.type(torch.float).to(Global.DEVICE.value), 
            #     ct_size
            # )
        )
    ).t().contiguous()
    while 1:
      c_heads = torch.unique((candidates.shape[0] * torch.rand(ch_size)).type(torch.int32))
      if c_heads.shape[0] == ch_size:
          break
    negative_samples_corrupted_heads = torch.vstack(
        ( 
            # torch.multinomial(
            #     candidates.type(torch.float).to(Global.DEVICE.value), 
            #     ch_size
            # ), 
            c_heads.to(Global.DEVICE.value),
            relations[ch_size:], 
            tails[ch_size:]
        )
    ).t().contiguous()

    negative_samples = torch.cat(
        (negative_samples_corrupted_heads, negative_samples_corrupted_tails),
        dim=0
    )
    ####################
    break
    # odds of picking a valid training triplet is low, even if a valid triplet is selected as a negative triplet,
    # the same triplet will also be selected as a valid positive triplet at exactly #epochs times
    # #(h, r, t) as a positive triplet >> #(h, r, t) as a negative triplet
    ####################
    if check_negative_samples(negative_samples, sorted_train_set):
      break


  # relations = supervision[:, Global.RELATION_INDEX.value]
  # while True:
  #   batch_size = supervision.shape[0]
  #   negative_samples = torch.vstack(
  #       (
  #           torch.multinomial(candidates.type(torch.float).to(Global.DEVICE.value), batch_size),
  #           relations,
  #           torch.multinomial(candidates.type(torch.float).to(Global.DEVICE.value), batch_size)
  #       )
  #   ).t().contiguous()
  #   if check_negative_samples(negative_samples, sorted_train_set):
  #     break

  graph = graph_data_maker(
      messaging=messaging,
      supervision=supervision,
      negative_samples=negative_samples,
      x=candidates.to(Global.DEVICE.value),
      x_feature='one-hot'
  )

  return graph


def enrich_messaging(messaging):
  heads = messaging[:, Global.HEAD_INDEX.value]
  tails = messaging[:, Global.TAIL_INDEX.value]
  relations = messaging[:, Global.RELATION_INDEX.value]
  
  tail_to_head = torch.vstack(
      (tails, relations + Global.NUM_RELATIONS.value, heads)
  ).t()

  return torch.cat(
      (messaging, tail_to_head),
      dim=0
  )