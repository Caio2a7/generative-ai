# Modelo IA

Este código implementa uma rede neural simples usando a biblioteca PyTorch para realizar uma tarefa de regressão. Vamos dividir o código em seções para entender melhor o que está acontecendo:
Geração de Dados:
    Amostras uniformemente distribuídas no intervalo (-10, 10) são geradas.

Definição da Rede Neural:
    Uma rede neural feedforward é definida.
    A arquitetura possui camadas lineares intercaladas com funções de ativação ReLU.

Treinamento e Avaliação:
    Verifica se a GPU está disponível e configura o dispositivo de execução.
    Instancia o modelo na GPU (se disponível).
    Divide os dados em conjuntos de treinamento e teste.
    Define a função de perda (Erro Quadrático Médio) e o otimizador (SGD).
    Implementa funções para treinar e avaliar o modelo.

Visualização:
    Uma função é definida para visualizar a comparação entre os resultados previstos e os dados reais.

Treinamento e Salvamento do Modelo:
    O modelo é treinado por um número específico de épocas.
    A cada 500 épocas, a perda de treinamento é impressa e uma visualização é gerada.
    O modelo treinado é salvo no arquivo 'model.pt'.

Execução:
    O código executa o treinamento e a avaliação do modelo.

Observações:

    O código carrega um modelo pré-treinado (se disponível) e continua o treinamento.
    A visualização compara a função alvo com os resultados previstos pela rede neural.
    O código assume que as bibliotecas necessárias (PyTorch, NumPy, Matplotlib) estão instaladas.
