from Global import *


def find_start_and_end_indcs(entity, entity_type, kg_sorted):
  up = kg_sorted.shape[0]
  down = 0
  indx = kg_sorted.shape[0] // 2
  found = False
  while up - down > 1:
    if kg_sorted[indx][entity_type].item() == entity:
      found = True
      break 
    elif kg_sorted[indx][entity_type].item() >= entity: 
      up = indx
      indx = (up + down) // 2
    else:
      down = indx 
      indx = (up + down) // 2
  if not found:
    return None
  while 1:
    indx += 1
    try:
      if not kg_sorted[indx][entity_type].item() == entity:
        indx -= 1
        end_indx = indx
        break
    except:
      end_indx = indx - 1
      break
  

  while 1:
    indx -= 1
    try:
      if not kg_sorted[indx][entity_type].item() == entity:
        indx += 1
        start_indx = indx
        break
    except:
      start_indx = indx + 1
      break
  return torch.tensor(range(start_indx, end_indx + 1))


def check_negative_samples(negative_samples, sorted_train_set):
  for triple in negative_samples:
    indcs = find_start_and_end_indcs(triple[0], 0, sorted_train_set)
    if isinstance(indcs, torch.Tensor):
      facts = sorted_train_set[indcs]
      are_actually_corrupt = ((triple == facts).type(torch.int8).sum(dim=-1) < 3).sum()
      if not are_actually_corrupt.type(torch.int8):
        return False 
  return True


def save_execution(model, iteration, epoch, epoch_loss):
  torch.save(model, f'model{iteration}-{epoch}.pth')
  with open(f'info{iteration}-{epoch}.txt', 'w') as info_file:
    info_file.write(f'model: {str(model)}')
    info_file.write('\n')
    info_file.write(f'iteration: {iteration}\n')
    info_file.write(f'epoch: {epoch + 1}\n')
    info_file.write(f'epoch loss: {epoch_loss}')

