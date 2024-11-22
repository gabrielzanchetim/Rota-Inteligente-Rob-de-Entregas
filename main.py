import tkinter as tk
from tkinter import ttk
from queue import PriorityQueue
import graphviz
import numpy as np
import networkx as nx
from PIL import Image

import matplotlib.animation as animacao
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Classe que representa um nó em um grafo ou árvore
class No:
    def __init__(self, valor, no_anterior=None, custo=0, altura=0):
        self.valor = valor
        self.no_anterior = no_anterior
        self.custo = custo
        self.altura = altura

    # Define o nó anterior
    def set_no_anterior(self, no_anterior):
        self.no_anterior = no_anterior

    # Retorna o nó anterior
    def get_no_anterior(self):
        return self.no_anterior

    # Retorna o valor do nó
    def get_valor(self):
        return self.valor

    # Comparação de custo para a fila de prioridade
    def __lt__(self, outro):
        return self.custo < outro.custo


# Classe que representa uma árvore binária
class ArvoreBinaria:
    def __init__(self, chave, pai=None, custo=0, altura=0):
        self.esquerda = None
        self.direita = None
        self.pai = pai
        self.valor = chave
        self.custo = custo
        self.altura = altura

    # Insere um nó na árvore
    def insere(self, chave, pai, custo, altura):
        if self.esquerda is None:
            self.esquerda = ArvoreBinaria(chave, pai=pai, custo=custo, altura=altura)
        elif self.direita is None:
            self.direita = ArvoreBinaria(chave, pai=pai, custo=custo, altura=altura)
        else:
            self.esquerda.insere(chave, pai, custo, altura)

    # Encontra um nó na árvore pelo valor
    def buscar(self, chave):
        if chave == self.valor:
            return self
        if self.esquerda:
            encontrado = self.esquerda.buscar(chave)
            if encontrado:
                return encontrado
        if self.direita:
            return self.direita.buscar(chave)
        return None

    # Imprime a árvore no console
    def imprime_arvore(self, altura=0, rotulos={}):
        if self.direita:
            self.direita.imprime_arvore(altura + 1, rotulos)
        pai_valor = rotulos[self.pai.valor] if self.pai else None
        atual_valor = rotulos[self.valor]
        print(' ' * 4 * altura + '->', f"[{atual_valor}, {pai_valor}, {self.custo}]")
        if self.esquerda:
            self.esquerda.imprime_arvore(altura + 1, rotulos)


# Calcula a heurística (distância de Manhattan) entre dois pontos
def heuristicas(a, b):
    if heuristica_nao_admissivel.get():
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) ** 2
    else:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) 

def imprime_no_console(no_atual, custo_g, destino, custo_f, fechado):
    if fechado: 
        print("No atual (Fechado):", no_atual.get_valor())
    else:
        print("No atual (Aberto):", no_atual.get_valor())
    print("No anterior:", no_atual.get_no_anterior().get_valor() if no_atual.get_no_anterior() else None)
    print("Custo G (Caminho):", custo_g[no_atual.get_valor()])
    print("Custo H (Heuristica):", heuristicas(no_atual.get_valor(), destino.get_valor()))
    print("Custo F (Total):", custo_f[no_atual.get_valor()], "\n")

def a_estrela(planta, comeco, destino_coords):
    linhas, colunas = planta.shape
    comeco = No((comeco[0], comeco[1]), custo=0, altura=0)
    destino = No((destino_coords[0], destino_coords[1]))

    fila = PriorityQueue()
    fila.put((0, comeco))

    custo_g = {comeco.get_valor(): 0}
    custo_f = {comeco.get_valor(): heuristicas(comeco.get_valor(), destino.get_valor())}
    comeco.custo = custo_f[comeco.get_valor()]

    lista_abertos = [comeco.get_valor()]
    lista_fechados = []

    passos = [([], lista_abertos.copy(), {comeco.get_valor(): custo_f[comeco.get_valor()]})]

    arvore_binaria = ArvoreBinaria(comeco.get_valor(), custo=custo_g[comeco.get_valor()], altura=0)

    while not fila.empty():
        no_atual = fila.get()[1]
        atual = no_atual.get_valor()

        imprime_no_console(no_atual, custo_g, destino, custo_f, 0)

        lista_abertos.remove(atual)
        lista_fechados.append(atual)

        if atual == destino.get_valor():
            caminho = []
            while no_atual:
                caminho.append(no_atual.get_valor())
                no_atual = no_atual.get_no_anterior()
            passos.append((lista_fechados.copy(), lista_abertos.copy(), custo_f.copy()))    
            return caminho[::-1], passos, arvore_binaria

        for d in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            vizinho_valor = (atual[0] + d[0], atual[1] + d[1])
            if 0 <= vizinho_valor[0] < linhas and 0 <= vizinho_valor[1] < colunas and planta[vizinho_valor] != 3:
                temp_custo_g = custo_g[atual] + 1
                if vizinho_valor not in custo_g or temp_custo_g < custo_g[vizinho_valor]:
                    no_vizinho = No(vizinho_valor, custo=heuristicas(vizinho_valor, destino.get_valor()), altura=no_atual.altura + 1)
                    no_vizinho.set_no_anterior(no_atual)
                    custo_g[vizinho_valor] = temp_custo_g
                    custo_f[vizinho_valor] = temp_custo_g + heuristicas(vizinho_valor, destino.get_valor())
                    if vizinho_valor not in lista_abertos:
                        fila.put((custo_f[vizinho_valor], no_vizinho))
                        lista_abertos.append(vizinho_valor)

                        atual_arvore_nos = arvore_binaria.buscar(no_atual.get_valor())
                        if atual_arvore_nos:
                            atual_arvore_nos.insere(no_vizinho.get_valor(), pai=atual_arvore_nos, custo=custo_g[vizinho_valor], altura=no_vizinho.altura)

        passos.append((lista_fechados.copy(), lista_abertos.copy(), custo_f.copy()))
    return None, passos, arvore_binaria


# Função que renderiza a árvore de busca em um gráfico
def renderiza_arvore(arvore_binaria, passos, rotulos, destino):
    grafo = graphviz.Digraph()

    def adicionar_nos_arestas(arvore_nos):
        if arvore_nos is not None:
            pai_valor = rotulos[arvore_nos.pai.valor] if arvore_nos.pai else None
            atual_valor = rotulos[arvore_nos.valor]
            distancia_manhattan = heuristicas(arvore_nos.valor, destino)
            grafo.node(str(arvore_nos.valor), f'{atual_valor}, {pai_valor}, h(n): {distancia_manhattan}')
            if arvore_nos.esquerda:
                grafo.edge(str(arvore_nos.valor), str(arvore_nos.esquerda.valor), label='1') 
                adicionar_nos_arestas(arvore_nos.esquerda)
            if arvore_nos.direita:
                grafo.edge(str(arvore_nos.valor), str(arvore_nos.direita.valor), label='1') 
                adicionar_nos_arestas(arvore_nos.direita)

    def adicionar_nos_mesmo_nivel(passos):
        alturas = {}
        for i, (_, lista_abertos, _) in enumerate(passos):
            for no in lista_abertos:
                atual_arvore_nos = arvore_binaria.buscar(no)
                if atual_arvore_nos is not None:
                    altura_no = atual_arvore_nos.altura
                    if altura_no not in alturas:
                        alturas[altura_no] = []
                    alturas[altura_no].append(no)

        for altura in alturas:
            with grafo.subgraph() as s:
                s.attr(rank='igual')    
                for no in alturas[altura]:
                    s.node(str(no))

    adicionar_nos_arestas(arvore_binaria)
    adicionar_nos_mesmo_nivel(passos)
    return grafo


# Função que cria e plota o grafo do ambiente
def criar_e_plotar_grafo(ax, planta, rotulos, objetivos, pos):
    linhas, colunas = planta.shape
    G = nx.Graph()

    for linha in range(linhas):
        for coluna in range(colunas):
            if planta[linha, coluna] in [0, 2]:
                G.add_node(rotulos[(linha, coluna)])

                if linha > 0 and planta[linha - 1, coluna] in [0, 2]:
                    G.add_edge(rotulos[(linha, coluna)],
                               rotulos[(linha - 1, coluna)])
                if linha < linhas - 1 and planta[linha + 1, coluna] in [0, 2]:
                    G.add_edge(rotulos[(linha, coluna)],
                               rotulos[(linha + 1, coluna)])
                if coluna > 0 and planta[linha, coluna - 1] in [0, 2]:
                    G.add_edge(rotulos[(linha, coluna)],
                               rotulos[(linha, coluna - 1)])
                if coluna < colunas - 1 and planta[linha, coluna + 1] in [0, 2]:
                    G.add_edge(rotulos[(linha, coluna)],
                               rotulos[(linha, coluna + 1)])

    # Adicionar as arestas entre os nós de objetivo e os nós adjacentes
    for objetivo in objetivos.values():
        linha, coluna = objetivo
        no_rotulo = rotulos[(linha, coluna)]
        if planta[linha, coluna] == 1:  # Apenas para objetivos (código 1)
            if linha > 0 and planta[linha - 1, coluna] in [0, 2]:
                G.add_edge(no_rotulo, rotulos[(linha - 1, coluna)])
            if linha < linhas - 1 and planta[linha + 1, coluna] in [0, 2]:
                G.add_edge(no_rotulo, rotulos[(linha + 1, coluna)])
            if coluna > 0 and planta[linha, coluna - 1] in [0, 2]:
                G.add_edge(no_rotulo, rotulos[(linha, coluna - 1)])
            if coluna < colunas - 1 and planta[linha, coluna + 1] in [0, 2]:
                G.add_edge(no_rotulo, rotulos[(linha, coluna + 1)])

    ax.clear()
    # Desenhar todos os nós
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=300,
                           node_color='lightblue',
                           ax=ax)
    # Desenhar todas as arestas em uma cor neutra
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', width=2)

    # Desenhar rótulos
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title('Grafo dos Caminhos')
    ax.axis('off') 
    plt.tight_layout()

    return G


# Função para ir para o destino selecionado
def ir_para_destino():
    resetar_variaveis() 
    ponto_inicial = combo_inicial.get()
    destino = combo_destinos.get()
    opcao = combo_opcoes.get()
    print("\n-------------------------------------------------------------------------------------")
    print("Ponto inicial selecionado:", ponto_inicial)
    print("Destino selecionado:", destino)
    print("")

    inicio_coords = None
    destino_coords = None

    # Encontrar as coordenadas do ponto inicial e do destino
    for (i, j), rotulo in rotulos.items():
        if rotulo == ponto_inicial:
            inicio_coords = (i, j)
            print("Coord. inicial selecionado:", inicio_coords)
        if rotulo == destino:
            destino_coords = (i, j)
            print("Coord. destino selecionado:", destino_coords)

    # Executar o algoritmo A* para encontrar o caminho
    caminho, passos, arvore_binaria = a_estrela(planta, inicio_coords, destino_coords)

    if caminho:
        caminho_rotulado = [rotulos[coord] for coord in caminho]
        print("Caminho encontrado:", caminho_rotulado)
        
        plotar_planta(ax1, planta, rotulos, caminho, destino_coords, opcao)

        tabela_data = []
        for _, (fechado, aberto_, _) in enumerate(passos):
            rotulos_abertos = [rotulos[n] for n in aberto_]
            rotulos_fechados = [rotulos[n] for n in fechado]

            rotulos_abertos = sorted(set(rotulos_abertos), key=rotulos_abertos.index)
            rotulos_fechados = sorted(set(rotulos_fechados), key=rotulos_fechados.index)

            aberto_str = ', '.join(rotulos_abertos) if rotulos_abertos else '-'
            fechado_str = ', '.join(rotulos_fechados) if rotulos_fechados else '-'

            tabela_data.append((aberto_str, fechado_str))

        for linha in tabela.get_children():
            tabela.delete(linha)
        for aberto_str, fechado_str in tabela_data:
            tabela.insert('', 'end', values=(aberto_str, fechado_str))

        ax2.clear()
        arvore_binaria.imprime_arvore(rotulos=rotulos)
        graficos_arvore = renderiza_arvore(arvore_binaria, passos, rotulos, destino_coords)
        graficos_arvore.render('arvore_binaria', format='png')

        img = Image.open("arvore_binaria.png")
        largura, altura = img.size
        proporcao_imagem = altura / largura
        aspecto_figura = ax2.get_window_extent().height / ax2.get_window_extent().width
        if proporcao_imagem > aspecto_figura:
            ax2.set_ylim(altura, 0)
            ax2.set_xlim(0, altura / aspecto_figura)
        else:
            ax2.set_xlim(0, largura)
            ax2.set_ylim(altura, 0)
        
        ax2.imshow(img, aspect='auto') 
        ax2.axis('off')
        
        criar_e_plotar_grafo(ax3, planta, rotulos, objetivos, pos)

        canvas.draw()

        animar_robo(caminho)
        
    else:
        print("Nenhum caminho encontrado.")


# Função para animar o robô percorrendo o caminho encontrado
def animar_robo(caminho):
    img_robo = mpimg.imread('robo.png')
    img_escolhida = combo_opcoes.get()
    
    if img_escolhida == "Água":
        img_final = mpimg.imread('garrafa-de-agua.png')
    elif img_escolhida == "Chocolate":
        img_final = mpimg.imread('chocolate.png')
    elif img_escolhida == "Café":
        img_final = mpimg.imread('xicara-de-cafe.png')
    else:
        img_final = img_robo
    
    imagem = ax1.imshow(img_robo, extent=(caminho[0][1] - 0.5, caminho[0][1] + 0.5, caminho[0][0] - 0.5, caminho[0][0] + 0.5))

    def atualiza(frame):
        if frame == len(caminho) - 1:
            imagem.set_data(img_final)
        imagem.set_extent((caminho[frame][1] - 0.5, caminho[frame][1] + 0.5, caminho[frame][0] - 0.5, caminho[frame][0] + 0.5))
        return imagem,

    ani = animacao.FuncAnimation(fig, atualiza, frames=len(caminho), interval=500, blit=True, repeat=False)
    ani._start()
    canvas.draw() 


def fechar_janela():
    root.quit()  # Fecha o loop principal
    root.destroy()  # Destroi a janela


# Função para criar a planta do ambiente
def criar_planta(linhas, colunas, objetivos):
    planta = np.full((linhas, colunas), 0)

    for rotulo, pos in objetivos.items():
        planta[pos] = 1

    paredes = [(0, 1), (0, 3), (0, 5), (0, 7), (1, 3), (1, 5), (1, 7), (2, 1),
               (2, 2), (2, 3), (2, 7), (3, 5), (4, 1), (4, 3), (4, 4), (4, 5),
               (4, 6), (4, 7), (5, 1), (5, 3)]

    for parede in paredes:
        planta[parede] = 3

    return planta


# Função para plotar a planta do ambiente
def plotar_planta(ax, planta, rotulos, caminho=[], destino=None, opcao=""):
    cmap = mcolors.ListedColormap(['white', 'blue', 'green', 'black'])
    intervalos = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(intervalos, cmap.N)

    ax.imshow(planta, cmap=cmap, norm=norm, interpolation='none')

    ax.set_xticks(np.arange(-.5, planta.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, planta.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    ax.tick_params(which='both',
                   bottom=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)
    ax.set_title('Grid da Empresa')

    for (i, j), rotulo in rotulos.items():
        ax.text(j, i, rotulo, ha='center', va='center', color='black')

    for (i, j) in caminho:
        ax.add_patch(
            plt.Rectangle((j - 0.5, i - 0.5),
                          1,
                          1,
                          fill=None,
                          edgecolor='red',
                          linewidth=2))

    ax.set_xlim(-0.5, planta.shape[1] - 0.5)
    ax.set_ylim(planta.shape[0] - 0.5, -0.5)


# Função para resetar as variáveis globais e limpar gráficos e tabelas
def resetar_variaveis():
    global caminho, passos, custo_g, arvore_binaria, tabela

    caminho = []
    passos = []
    custo_g = {}
    arvore_binaria = None

    # Limpar as tabelas e gráficos
    for linha in tabela.get_children():
        tabela.delete(linha)

    ax1.clear()
    plotar_planta(ax1, planta, rotulos)
    
    ax2.clear()
    ax2.axis('off')
    ax2.set_title('Grafo da Árvore de Busca')

    ax3.clear()
    ax3.axis('off')
    ax3.set_title('Grafo dos Caminhos')

    canvas.draw()


# Configuração inicial da planta do ambiente e dos objetivos
linhas = 6
colunas = 9

objetivos = {
    'RH': (0, 0),
    'MKT': (0, 2),
    'TEC': (0, 4),
    'ADM': (0, 8),
    'FIN': (5, 0),
    'LOG': (5, 4)
}

planta = criar_planta(linhas, colunas, objetivos)

coordenadas_numeros = {
    (0, 6): 0,    (1, 0): 1,    (1, 1): 2,    (1, 2): 3,    (1, 4): 4,    (1, 6): 5,    
    (1, 8): 6,    (2, 0): 7,    (2, 4): 8,    (2, 5): 9,    (2, 6): 10,    (2, 8): 11,    
    (3, 0): 12,    (3, 1): 13,    (3, 2): 14,    (3, 3): 15,    (3, 4): 16,    (3, 6): 17,    
    (3, 7): 18,    (3, 8): 19,    (4, 0): 20,    (4, 2): 21,     (4, 8): 22,    (5, 2): 23,    
    (5, 5): 24,    (5, 6): 25,    (5, 7): 26,    (5, 8): 27
}

# Inicializando o dicionário rótulos com coordenadas numéricas
rotulos = {pos: str(num) for pos, num in coordenadas_numeros.items()}

# Adicionando os objetivos ao dicionário rótulos
rotulos.update({pos: nome for nome, pos in objetivos.items()})

pos = {rotulos[(i, j)]: (j, -i) for i in range(linhas) for j in range(colunas) if (i, j) in rotulos}

# Configuração da interface gráfica 
root = tk.Tk()
root.title("Entrega em escritório")

rotulos_dos_objetivos = {
    nome: nome
    for nome in objetivos.keys()
}

root.geometry("1900x1200")  

quadro_superior = tk.Frame(root, width=1900, height=100)
quadro_superior.pack_propagate(False) 
quadro_superior.pack(pady=10)

combo_inicial = ttk.Combobox(quadro_superior, values=list(rotulos_dos_objetivos.values()), state="readonly")
combo_inicial.set('Selecione o ponto inicial')
combo_inicial.pack(side=tk.LEFT, padx=5)

combo_destinos = ttk.Combobox(quadro_superior, values=list(rotulos_dos_objetivos.values()), state="readonly")
combo_destinos.set('Selecione o destino')
combo_destinos.pack(side=tk.LEFT, padx=5)

combo_opcoes = ttk.Combobox(quadro_superior, values=list(["Água", "Café", "Chocolate"]), state="readonly")
combo_opcoes.set('O que deseja levar?')
combo_opcoes.pack(side=tk.LEFT, padx=5)

heuristica_nao_admissivel = tk.BooleanVar()
chk_exibir_caminho = ttk.Checkbutton(quadro_superior, text="Heurística Não Admissível", variable=heuristica_nao_admissivel)
chk_exibir_caminho.pack(side=tk.LEFT, padx=5)

btn_confirmar = ttk.Button(quadro_superior, text="Confirmar Destino", command=ir_para_destino)
btn_confirmar.pack(side=tk.LEFT, padx=5)

quadro_graficos = tk.Frame(root, width=1300, height=650)  
quadro_graficos.pack_propagate(False)
quadro_graficos.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

quadro_tabela = tk.Frame(root, width=600, height=550) 
quadro_tabela.pack_propagate(False)
quadro_tabela.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20)

fig = Figure(figsize=(14, 9))  
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

gs.update(wspace=0.1, hspace=0.1, left=0.05, right=0.95, top=0.95, bottom=0.05)

ax1 = fig.add_subplot(gs[0, 0])  # Primeiro eixo para o grid do galpão (primeira linha, primeira coluna)
ax2 = fig.add_subplot(gs[:, 1])  # Segundo eixo para a árvore de busca (ocupa as duas linhas na segunda coluna)
ax3 = fig.add_subplot(gs[1, 0])  # Terceiro eixo para o grafo dos caminhos (segunda linha, primeira coluna)

canvas = FigureCanvasTkAgg(fig, master=quadro_graficos)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

plotar_planta(ax1, planta, rotulos)

ax2.axis('off')
ax2.set_title('Grafo da Árvore de Busca')

ax3.axis('off')
ax3.set_title('Grafo dos Caminhos')

# Criar a tabela
tabela = ttk.Treeview(quadro_tabela, columns=('Nós Abertos', 'Nós Fechados'), show='headings')
tabela.heading('Nós Abertos', text='Nós Abertos')
tabela.heading('Nós Fechados', text='Nós Fechados')

# Definir a largura das colunas
tabela.column('Nós Abertos', width=100)
tabela.column('Nós Fechados', width=500)

tabela.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

plt.tight_layout()

# Adicionar manipulador de fechamento de janela
root.protocol("WM_DELETE_WINDOW", fechar_janela)

root.mainloop()
