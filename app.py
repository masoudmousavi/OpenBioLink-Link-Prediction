from Global import *
from GNN import GNN
from Graph import mini_batch_maker
from ML import train, evaluate_hits_at_10, evaluation_rank
from Utils import save_execution


model = GNN(
    x_feature='one-hot',
    conv_dims=[128, 128, 128],
    embedding_dims=(features.max() + 1, 128),
    fully_connected_for_head_dims=[128, 84], 
    fully_connected_for_tail_dims=[128, 84],
    dropout={
        "emb": 0.2,
        "conv": 0.2,
        "fc": 0.45
    }
).to(Global.DEVICE.value)
print(model)
margin_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=Global.MARGIN.value, reduction=Global.REDUCTION.value)
opt = Adam(model.parameters()) 
# model.reset_parameters()

######################################## Train - Dev ########################################
try:
  for epoch in range(15):
    iteration = 0
    epoch_loss = 0
  ############################## TRAIN Graph Maker #####################################
    for i in tqdm(range(0, train_set.shape[0] - Global.MINI_BATCH_SIZE.value, Global.MINI_BATCH_SIZE.value)):
      iteration += 1
      train_supervision = train_set[i: i + Global.MINI_BATCH_SIZE.value, :]
      train_messaging = torch.cat(
          (train_set[: i, :], train_set[i + Global.MINI_BATCH_SIZE.value: , :]),
          dim=0
      )
      train_graph = mini_batch_maker(
          train_messaging,
          train_supervision,
          features,
          'one-hot'
      )
      loss, positive_score, negative_score = train(model, train_graph, opt, margin_ranking_loss_fn)
      epoch_loss += loss
      if iteration % 5000 == 0 or iteration == 1:
        print('')
        print('-' * 50)
        agg_p_score = positive_score.sum().item()
        agg_n_score = negative_score.sum().item()
        print(f'Train Batch {iteration}:')
        print('-' * 50)
        print(f'Batch Loss:    {loss: .4f}')
        print('-' * 50)
        print(f'Average Loss:         <{epoch_loss / iteration: .4f} >')
        print('-' * 50)
        print(f'Agg P-Score:   {agg_p_score: .4f}')
        print('-' * 50)
        print(f'Agg N-Score:   {agg_n_score: .4f}')
        print(f'===' * 25)

      if iteration % 25000 == 0:
        print('')
        print('Partial Validation: ')
        save_execution(model, iteration, epoch, epoch_loss)
        a, b = evaluation_rank(model, val_set[:25000], train_set)
        evaluator.eval(
                    a,
                    b,
                    val_set[:25000],
                    False
                  )
        print(f'***' * 25)
    evaluate_hits_at_10(model, mode='validation')
    
except :
    print('', 'Interrupted')
    save_execution(model, iteration, epoch, epoch_loss)

######################################## Test ########################################
try:
  evaluate_hits_at_10(model, 'testing')
except :
  pass