import torch
from basic_model import NeuralNetwork

print('running')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = NeuralNetwork(number_of_classes = 10).to(device)
print(model)

batch_size = 4
X = torch.rand(batch_size, 10, 100, device=device)
pred_probab = model(X)
print(f"Predicted probabilities: {pred_probab}")
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

print('done')