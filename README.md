# 👗 VestAI – Análise Visual de Padrões de Vestuário com Visão Computacional

![Banner](https://i.postimg.cc/k51tW12Y/Banner.png)

> Projeto desenvolvido no MBA em Ciência de Dados e Inteligência Artificial – Faculdade SENAC/PE  
> Disciplina: Visão Computacional • Professor: Heuryk Wilk  
> 📅 Junho de 2025

---

## 🎯 Objetivo

A aplicação **VestAI** utiliza técnicas de **visão computacional** e **deep learning** para identificar, por meio da leitura de imagens ou vídeos, padrões de vestuário entre pessoas — como peças **lisas** ou **estampadas**.

Essa análise tem como foco auxiliar o setor varejista na compreensão de **preferências visuais** dos consumidores, fornecendo dados estratégicos para:

- Campanhas de marketing segmentadas
- Otimização de atendimento
- Planejamento de estoques e produção

---

## 🔭 Visão de Futuro

Como evolução natural da aplicação, há intenção de:

- **Classificar o tipo da peça** (ex: saia, blusa, vestido, calça)
- **Relacionar padrões culturais e étnicos** com escolhas de vestuário
- Expandir a análise para séries temporais (ex: mudanças sazonais)

---

## 🧠 Tecnologias Utilizadas

| Biblioteca/Ferramenta | Descrição |
|------------------------|-----------|
| **Streamlit** | Interface web interativa para o usuário |
| **OpenCV** | Manipulação de imagens e captura de vídeo |
| **Pillow (PIL)** | Processamento de imagens no formato Python |
| **YOLOv8 (Ultralytics)** | Detecção de pessoas em imagens e vídeos |
| **PyTorch** | Framework de redes neurais |
| **torchvision** | Transformações de imagem e modelos pré-treinados |
| **MobileNetV2** | Backbone leve para classificação de roupas |
| **NumPy** | Manipulação de arrays |
| **Pandas** | Organização de dados tabulares |
| **tempfile / os** | Manipulação de arquivos temporários |

---

## 🧪 Como Funciona

1. **Detecção com YOLOv11**:
   - Pessoas são detectadas em imagens, vídeos ou pela webcam.
   - A imagem da pessoa é dividida entre "parte de cima" e "parte de baixo".

2. **Classificação com MobileNetV2**:
   - Dois modelos distintos classificam cada parte da vestimenta como:
     - `Lisas`
     - `Estampadas`

3. **Resultado**:
   - O app exibe, em tempo real, o número de pessoas detectadas e a contagem de peças por tipo.

---

## 🖼️ Interface da Aplicação

- **Webcam**, **Vídeo** ou **Imagem** como entrada
- Exibição da imagem com as caixas delimitadoras
- Métricas resumidas (pessoas, peças lisas, peças estampadas)

---

## 🚀 Como Executar Localmente

streamlit run app.py

---

## 📁 Estrutura de Arquivos

- `app.py`: Código principal da aplicação Streamlit
- `yolo11n.pt`: Modelo YOLO para detecção de pessoas
- `clothing_top_classifier.pt`: Classificador para parte de cima das roupas
- `clothing_bottom_classifier.pt`: Classificador para parte de baixo das roupas
- `requirements.txt`: Lista de dependências do projeto
- `README.md`: Documentação do projeto


---

## 👤 Autores

- **Ivan Filho, José Gentil, Anízio Neto e Vinicius de Souza**
- MBA em Ciência de Dados e IA – Faculdade SENAC/PE

---

# ⭐ Agradecimentos

Agradeço ao Prof. **Heuryk Wilk** pela orientação no desenvolvimento do projeto.

---
