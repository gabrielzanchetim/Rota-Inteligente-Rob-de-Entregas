# **Entrega em Escritório com Algoritmo A***

## **Descrição**
Este projeto simula a otimização de rotas para um robô de entregas em um ambiente de escritório, utilizando o algoritmo A* para encontrar o menor caminho entre um ponto inicial e um destino. O objetivo é garantir a entrega eficiente de itens, evitando obstáculos e utilizando funções heurísticas para calcular os custos do caminho.

A aplicação foi inspirada na notícia sobre a **ADA**, a primeira robô de delivery a chamar elevador no Brasil, desenvolvida em parceria entre o iFood e a Synkar. A ADA foi projetada para realizar entregas autônomas em ambientes complexos, como escritórios e prédios residenciais, utilizando inteligência artificial e tecnologias de IoT. A notícia completa pode ser lida em [A ADA é a primeira robô de delivery a chamar elevador no Brasil](https://institucional.ifood.com.br/inovacao/a-ada-e-a-primeira-robo-de-delivery-a-chamar-elevador-no-brasil/).

---

## **Funcionalidades**
1. **Seleção de Ponto Inicial e Destino**:
   - Escolha do local de partida e do destino dentro do ambiente do escritório.
   
2. **Seleção de Itens para Transporte**:
   - O robô pode carregar itens como **água**, **café** ou **chocolate**.

3. **Otimização de Rotas**:
   - Implementação do algoritmo A* para encontrar o menor caminho.
   - Suporte a funções heurísticas admissíveis (distância de Manhattan) e não admissíveis.

4. **Visualizações Gráficas**:
   - **Grid do Escritório**: Representação do ambiente com obstáculos, pontos de interesse e caminhos percorridos.
   - **Árvore de Busca**: Mostra os nós abertos e fechados durante a execução do algoritmo.
   - **Grafo dos Caminhos**: Exibe todas as conexões possíveis no ambiente.
   - **Animação**: Simula o robô percorrendo o caminho encontrado.

5. **Interface Amigável**:
   - Utilização de comboboxes para seleção de opções.
   - Botões intuitivos para iniciar o processo e visualizar os resultados.

---

## **Tecnologias Utilizadas**
- **Linguagem**: Python
- **Interface Gráfica**: Tkinter
- **Algoritmo de Busca**: A*
- **Bibliotecas**:
  - `numpy`: Manipulação de matrizes (grid do ambiente).
  - `networkx`: Geração de grafos.
  - `matplotlib`: Visualizações e animações.
  - `graphviz`: Renderização de árvores de busca.
  - `Pillow (PIL)`: Processamento de imagens.

---

## **Como Funciona**
1. O escritório é representado por uma grade (matriz) onde:
   - Células brancas: Caminhos transitáveis.
   - Células pretas: Obstáculos.
   - Células azuis: Departamentos do escritório (destinos possíveis).
   
2. O algoritmo A* é executado para encontrar o menor caminho, considerando:
   - **Custo G**: Distância acumulada até o nó atual.
   - **Custo H**: Heurística que estima a distância restante até o destino.

3. O sistema exibe:
   - O caminho percorrido pelo robô.
   - Lista de nós abertos e fechados.
   - Árvore de busca e grafo do ambiente.
   - Animação do robô realizando a entrega.

---

## **Resultados Esperados**
- Caminhos otimizados para entregas no escritório.
- Visualização clara do processo de busca, incluindo:
  - Nós abertos e fechados.
  - Árvore de busca.
  - Grafo dos caminhos possíveis.
  - Animação do robô em ação.

## **Exemplo de Execução**
<img width="958" alt="escritorio" src="https://github.com/user-attachments/assets/04b25741-7725-4eb3-b733-32de8e8e4c36">
---
