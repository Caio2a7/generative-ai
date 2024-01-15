# Uma lib direcionada para redes neurais
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from math import sin
import matplotlib.pyplot as plt

# Geração de amostras uniformes
tensor = torch.FloatTensor(3, 3)
urand = tensor.uniform_().to("cuda" if torch.cuda.is_available() else "cpu")

# Aqui vem a definição do dataset para essa rede
# Nesse caso esse dataset cria um intervalo entre os números -10 a 10 e cada elemento dele ele grava o dado no dataset
class AlgebraicDataset(Dataset):
  
  def __init__(self, f, interval, nsamples):
      X = torch.rand((nsamples, 1)).to(urand.device) * (interval[1] - interval[0]) + interval[0]
      self.data = [(x, f(x)) for x in X]
      
  def __len__(self):
      return len(self.data)
    
  def __getitem__(self, idx):
        return self.data[idx]

# INICIALIZAÇÃO DA REDE, aqui você constroi a rede no torch
#   exp: nn.RelU() que realiza operações não lineares
class MultiLayerNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(   # O nn.Sequential constroi a rede
        nn.Linear(1, 128),         # O Liner, Relu etc são os tipos de modelos de rede(as camadas)
        nn.ReLU(),
        nn.Linear(128, 64),        # O 128 é o número de neurônios e o 64 é o número de funções phi
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
# Pega o dado e joga na rede
  def forward(self, x):
    return self.layers(x)

# Vai testar se o seu pc possui placas da nvidia, melhores para realizar esse tipo de operação
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando na {device}")
# Vai colocar para esse modelo ser rodado na GPU
multimodel = MultiLayerNetwork().to(device)

# Aqui são os valores utilizados no dataset
# O intervalo de número
interval = (-10, 10)
train_nsamples = 1000  # Quantos números no intervalo vai pegar para treinar
test_nsamples = 100    # Quantos números no intervalo vai pegar para testar
f = lambda x: sin(x/2) # A função, ou seja de que forma vai por esse números

# Pega toda a estrutura e os dados postos até agora e cria o objeto do treinamento e teste
train_dataset = AlgebraicDataset(f, interval, train_nsamples)
test_dataset = AlgebraicDataset(f, interval, test_nsamples)
# Carrega tudo no dataloader
train_dataloader = DataLoader(train_dataset, train_nsamples, shuffle=True)
test_dataloader = DataLoader(test_dataset, test_nsamples, shuffle=True)

# Define a função de perda, MSE o qual contabiliza o erro da rede
lossfunc = nn.MSELoss()
# Define o optimizador, o qual faz a rede descer até o vale da verdade
optimizer = torch.optim.SGD(multimodel.parameters(), lr=1e-3)


# AQUI EXECUTA O TREINAMENTO MERMO
def train(model, dataloader, lossfunc, optimizer):
    model.train()
    cumloss = 0.0
    for output, verdade in dataloader:
        x = output.unsqueeze(1).float().to(device)
        y = verdade.unsqueeze(1).float().to(device)

        pred = model(x)
        loss = lossfunc(pred, y)

        # zera os gradientes acumulados
        optimizer.zero_grad()
        # computa os gradientes
        loss.backward()
        # anda, de fato, na direção que reduz o erro local
        optimizer.step()

        # loss é um tensor; item pra obter o float
        cumloss += loss.item()

    return cumloss / len(dataloader)


def test(model, dataloader, lossfunc):
    model.eval()

    cumloss = 0.0
    with torch.no_grad():
        for output, verdade in dataloader:
            x = output.unsqueeze(1).float().to(device)
            y = verdade.unsqueeze(1).float().to(device)

            pred = model(x)
            loss = lossfunc(pred, y)
            cumloss += loss.item()

    return cumloss / len(dataloader)


def plot_comparinson(f, model, interval=(-10, 10), nsamples=10):
  fig, ax = plt.subplots(1, 1, figsize=(10, 10))

  ax.grid(True, which='both')
  ax.spines['left'].set_position('zero')
  ax.spines['right'].set_color('none')
  ax.spines['bottom'].set_position('zero')
  ax.spines['top'].set_color('none')

  samples = np.linspace(interval[0], interval[1], nsamples)
  model.eval()
  with torch.no_grad():
    pred = model(torch.tensor(samples).unsqueeze(1).float().to(device))

  ax.plot(samples, list(map(f, samples)), "o", label="ground truth")
  ax.plot(samples, pred.cpu(), label="model")
  plt.legend()
  plt.show()

# Vai definir quantas épocas(rodadas) de treinamento a rede irá realizar
epochs = 5001
multimodel = torch.load('model.pt')
for t in range(epochs):
  train_loss = train(multimodel, train_dataloader, lossfunc, optimizer)
  if t % 500 == 0:
    print(f"Epoch: {t}; Train Loss: {train_loss}")
    plot_comparinson(f, multimodel, nsamples=40)

test_loss = test(multimodel, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss}")

torch.save(multimodel, 'model.pt')
'''
-> torch.save(modelo, 'D:...')  # Salva a rede neural para ser usada
-> model = torch.load('D:...')  # Carrega a rede neural para ser usada

    
-> funcao_linha = lambda x: 2*x + 3  # Define a função/operação realizada pela rede
-> intervalo = (-10, 10)             # Define o intervalo de valores utilizados por essa rede que usa uma função
-> treinamento_amostras = 1000       # Define a quantidade de amostras
-> teste_amostras = 100              #

-> treinamento_dataset = DatasetAlgebrico(funcao_linha, intervalo, treinamento_amostras)  # Faz a operação no conjunto de dados
-> teste_dataset = DatasetAlgebrico(funcao_linha, intervalo, teste_amostras)              #
-> treinamento_dataloader = DataLoader(treinamento_dataset, batch_size=treinamento_amostras, shuffle=True)   
# vai realizar a operação em toda a rede, 
    batch_size determina quantas operações você vai fazer por vez, shuffle é sobre embaralhar ou não os dados
-> teste_dataloader = DataLoader(teste_dataset, batch_size=teste_amostras, shuffle=True)    


- device = "cuda" if torch.cuda.is_available() else "cpu"    # Se tiver gpu vai utilizar ela
print(f"Rodando na {device}")
- modelo = RedeNeuralLinha().to(device)  # Vai jogar a rede pra ser rodada na gpu
- funcao_perda = nn.MSELoss()  # Função perda que determina se a IA tá acertando ou não nas suas operações
- optimizador = torch.optim.SGD(modelo.parameters(), lr=1e-3)  # O optimizador é o que faz os ajustes da Ia, ela
descer o morro né, ir descendo a parábula para ajustar os valores


- epochs = 201  # Determina a quantidade de vezes que você vai rodar de novo os testes
* for t in range(epochs):
    treinamento_perda = treinamento(modelo, treinamento_dataloader, funcao_perda, optimizador)
    if t % 10 == 0:
        print(f"Epoch: {t}; Train Loss: {treinamento_perda}")
        graficos_comparados(funcao_linha, modelo)
        
        
# Essas duas funç~eos gigantescas irão fazer as operações de função perda e do optimizador na rede neural, o treinamento
fato
*def treinamento(modelo, dataloader, funcao_perda, optimizador):
    ....

    return perda_acumulada / len(dataloader)


* def teste(modelo, dataloader, funcao_perda):
    ....
'''