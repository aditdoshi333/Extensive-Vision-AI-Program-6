import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_incorrect_preds(model, test_loader, device):
  model.eval()
  correct = 0
  incorrect_preds = {'incorrect': []}
  i = 1
  inference_data = dict()
  with torch.no_grad():
      for data, target in test_loader:
          if (len(incorrect_preds['incorrect'])<12):
              data, target = data.to(device), target.to(device)
              output = model(data)
              pred = output.argmax(dim=1, keepdim=True)  
              correct += pred.eq(target.view_as(pred)).sum().item()
              i+=1
              inference_data['Input'], inference_data['target'], inference_data['pred'] = data.to('cpu'), target.to('cpu'), pred.to('cpu').view(-1,)

              for id in range(len(data)):
                  if inference_data['target'][id] != inference_data['pred'][id]:
                      incorrect_preds['incorrect'] = incorrect_preds['incorrect']+ [{'Image':data[id],'pred':pred[id],'Target' : target[id]}]
  
  plt.figure(figsize=(15,7)) 
  for i in range(10):
      plt.subplot(2,5,i+1)  
      pixels = np.array(incorrect_preds['incorrect'][i]['Image'].cpu() , dtype='uint8')

      pixels = pixels.reshape((28, 28))

      # Plot
      plt.title('Target Value is {label}\n Predicted Value is {pred}'.format(label=incorrect_preds['incorrect'][i]['Target'].cpu(), pred =incorrect_preds['incorrect'][i]['pred'].cpu()[0]), color='r')
      plt.imshow(pixels, cmap='gray')

  plt.show()
