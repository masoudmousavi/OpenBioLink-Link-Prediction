from Global import *
from GNN import GNN
from Graph import mini_batch_maker
from ML import train, evaluate_hits_at_10, evaluation_rank
from ShallowModel import ShallowModel
from Utils import save_execution


model = GNN(
    x_feature='one-hot',
    conv_dims=[32, 32, 32],
    embedding_dims=(features.max() + 1, 48),
    fully_connected_for_head_dims=[32, 24], 
    fully_connected_for_tail_dims=[32, 24],
    dropout={
        "emb": 0.3,
        "conv": 0.2,
        "fc": 0.45
    }
).to(Global.DEVICE.value)
# model = torch.load('model25000-1.pth').to(Global.DEVICE.value)
# shallow_model = ShallowModel('one-hot', {'emb':0.3}, (features.max() + 1, 72))
# opt_shallow = Adam(shallow_model.parameters())
print(model)
margin_ranking_loss_fn = torch.nn.MarginRankingLoss(margin=Global.MARGIN.value, reduction=Global.REDUCTION.value)
opt = Adam(model.parameters()) 
######################################## Train - Dev ########################################
try:
  for epoch in range(0, 25):
    # if epoch == 1:  
    #   iteration = 25000
    #   epoch_loss = 15663269.89113617
    # else:
    iteration = 0
    epoch_loss = 0
  ############################## TRAIN Graph Maker #####################################
    for i in tqdm(range(iteration * Global.MINI_BATCH_SIZE.value, train_set.shape[0] - Global.MINI_BATCH_SIZE.value, Global.MINI_BATCH_SIZE.value)):
      iteration += 1
      train_supervision = train_set[i: i + Global.MINI_BATCH_SIZE.value, :]
      train_messaging = torch.cat(
          (train_set[: i, :], train_set[i + Global.MINI_BATCH_SIZE.value:, :]),
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
        # print('')
        # print('-' * 50)
        agg_p_score = positive_score.sum().item()
        agg_n_score = negative_score.sum().item()
        with open('logs.txt', 'a') as f:
          f.write(f'Train Batch {iteration}:')
          f.write('\n')
          f.write('-' * 50)
          f.write('\n')
          f.write(f'Batch Loss:    {loss: .4f}')
          f.write('\n')
          f.write('-' * 50)
          f.write('\n')
          f.write(f'Average Loss:         <{epoch_loss / iteration: .4f} >')
          f.write('\n')
          f.write('-' * 50)
          f.write('\n')
          f.write(f'Total Loss:           <{epoch_loss: .4f} >')
          f.write('\n')
          f.write('-' * 50)
          f.write('\n')
          f.write(f'Agg P-Score:   {agg_p_score: .4f}')
          f.write('\n')
          f.write('-' * 50)
          f.write('\n')
          f.write(f'Agg N-Score:   {agg_n_score: .4f}')
          f.write('\n')
          f.write(f'===' * 25)
          f.write('\n')

      if iteration % 25000 == 0:
        # print('')
        # print('Partial Validation: ')
        save_execution(model, iteration, epoch, epoch_loss)
        # a, b = evaluation_rank(model, val_set[:25000], train_set)
        # evaluator.eval(
        #             a,
        #             b,
        #             val_set[:25000],
        #             False
        #           )
        # print(f'***' * 25)
    if epoch % 3 == 0:
      evaluate_hits_at_10(model, mode='validation')
    
except :
    print('', 'Interrupted')
    save_execution(model, iteration, epoch, epoch_loss)

######################################## Test ########################################
try:
  evaluate_hits_at_10(model, 'testing')
except :
  pass