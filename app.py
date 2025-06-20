import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import tempfile
import os
import pandas as pd

# Configuração da página (layout wide para usar toda a largura, depois limitamos via CSS)
st.set_page_config(page_title="VestAI", layout="wide")

# CSS para centralizar e limitar largura do container principal do Streamlit
st.markdown("""
    <style>
        /* Centraliza o container principal */
        .main .block-container {
            max-width: 710px;
            margin-left: auto;
            margin-right: auto;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        /* Para título e textos brancos */
        h3, h4, h5, p, ul {
            color: #FFFFFF;
        }
        /* Remove espaçamento extra do hr padrão */
        hr {
            margin: 1rem 0;
            border-color: #666666;
        }
        /* Garante que o container pai centralize o conteúdo */
        .css-18e3th9 {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Cabeçalho e banner
st.write("<h3 style='color: #434442;'>Faculdade SENAC/ PE - MBA Ciência de Dados e Inteligência Artificial</h3>", unsafe_allow_html=True)
st.write("<h5 style='color: #434442;'>Projeto: Visão Computacional (Prof. Heuryk Wilk) * Junho 2025</h5>", unsafe_allow_html=True)

st.write("---")

st.markdown("<h4>Objetivo</h4>", unsafe_allow_html=True)
st.markdown("""
<p style='color: #C8C8C9; font-size: 12px'>
A aplicação tem como objetivo, por meio da leitura de imagens, identificar padrões de vestuário entre potenciais clientes do varejo de roupas. Inicialmente, busca-se segmentar e compreender o comportamento dessas pessoas em relação à escolha das tonalidades das roupas que utilizam, classificando-as em categorias como tons mais sóbrios e estampas. Com base nessa segmentação, é possível direcionar ações de marketing mais focadas, melhorar o posicionamento no atendimento, ajustar o planejamento de lotes de produção, entre outras oportunidades estratégicas.
</p>
""", unsafe_allow_html=True)

st.markdown("<h4>Visão de Futuro</h4>", unsafe_allow_html=True)
st.markdown("""
<p style='color: #C8C8C9; font-size: 12px'>
Como desdobramento futuro, pretende-se aprofundar essa análise, incluindo a identificação do tipo de peça de roupa, como: saias, calças, blusas e vestidos. Além de estabelecer possíveis relações com fatores étnicos e culturais, ampliando a compreensão sobre os hábitos de consumo.
</p>
""", unsafe_allow_html=True)

st.markdown("<h4>Tecnologias Utilizadas</h4>", unsafe_allow_html=True)
st.markdown("""
<ul style='color: #C8C8C9; font-size: 12px'>
  <li><b>Streamlit</b>: Utilizado para construção da interface web da aplicação, facilitando a interação com o usuário.</li>
  <li><b>OpenCV (cv2)</b>: Responsável pela leitura, captura e manipulação de imagens e vídeos.</li>
  <li><b>Pillow (PIL)</b>: Usado para abrir e converter imagens em formatos compatíveis com outras bibliotecas.</li>
  <li><b>YOLO (Ultralytics)</b>: Empregado para detecção de objetos nas imagens, identificando vestimentas e regiões de interesse.</li>
  <li><b>PyTorch (torch)</b>: Base para o uso de modelos de aprendizado profundo e redes neurais.</li>
  <li><b>torch.nn</b>: Permite a construção e personalização das camadas de redes neurais.</li>
  <li><b>Torchvision</b>: Auxilia nas transformações das imagens (como redimensionamento e normalização) e no uso de modelos pré-treinados.</li>
  <li><b>NumPy</b>: Utilizado para operações numéricas e manipulação de arrays de imagem.</li>
  <li><b>Pandas</b>: Responsável pela organização e análise de dados estruturados.</li>
  <li><b>os</b>: Manipula caminhos e arquivos no sistema operacional.</li>
  <li><b>tempfile</b>: Cria arquivos e diretórios temporários durante a execução da aplicação.</li>
</ul>
""", unsafe_allow_html=True)

st.markdown("<h4>Aplicação</h4>", unsafe_allow_html=True)

image_url = "https://i.postimg.cc/k51tW12Y/Banner.png"
st.image(image_url, width=710)


# Estilo para os cards
st.markdown("""
<style>
.metric-box {
    background-color: #f0f2f6;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 10px rgba(0,0,0,0.05);
    margin-bottom: 10px;
}
.metric-box h2 {
    margin: 0;
    font-size: 28px;
    color: black;
}
.metric-box span {
    font-size: 20px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# Carrega modelo YOLOv11
detector = YOLO("yolo11n.pt")

# Carrega dois classificadores: parte de cima e parte de baixo

def load_classifier(path):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


classifier_top = load_classifier("clothing_top_classifier.pt")
classifier_bottom = load_classifier("clothing_bottom_classifier.pt")

# Transformações para classificação
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.7, 0.7, 0.7], [0.7, 0.7, 0.7])
])

class_names = ['Estampadas', 'Lisas']

# Classifica um recorte com o modelo fornecido

def classify_crop(crop, model):
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        return class_names[pred.item()]

# Processa o frame e realiza detecção e classificação

def process_frame(frame):
    results = detector(frame)[0]
    registros = []

    for box in results.boxes:
        if int(box.cls[0].item()) == 0:  # class 0 = pessoa
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h = y2 - y1
            mid_y = y1 + h // 2

            top_crop = frame[y1:mid_y, x1:x2]
            bottom_crop = frame[mid_y:y2, x1:x2]

            top_label = classify_crop(top_crop, classifier_top)
            bottom_label = classify_crop(bottom_crop, classifier_bottom)

            registros.append({
                "Parte de Cima": top_label,
                "Parte de Baixo": bottom_label
            })

            # Desenha a caixa sem texto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)

    return frame, registros

st.markdown("""
    <style>
        /* Cor e tamanho do label (título da pergunta) */
        .stRadio > label {
            color: #FFFFFF !important;
            font-size: 12px !important;
        }

        /* Cor e tamanho das opções */
        .stRadio div div {
            color: #C8C8C9 !important;
            font-size: 12px !important;
        }

        /* Estilo do label do file_uploader também */
        .stFileUploader label {
            font-size: 12px !important;
            color: #C8C8C9 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Interface Streamlit
fonte = st.radio("Escolha a fonte de entrada:", ["Webcam", "Vídeo", "Imagem"])
arquivo = None
if fonte in ["Vídeo", "Imagem"]:
    arquivo = st.file_uploader("Envie o arquivo", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

frame_placeholder = st.empty()
tabela_placeholder = st.empty()

# Exibe resultado por pessoa detectada

def mostrar_registros(registros):
    total_pessoas = len(registros)
    total_lisas = 0
    total_estampadas = 0

    for r in registros:
        if r['Parte de Cima'] == 'Lisas':
            total_lisas += 1
        else:
            total_estampadas += 1
        if r['Parte de Baixo'] == 'Lisas':
            total_lisas += 1
        else:
            total_estampadas += 1

    st.markdown(f"""
    <div class='metric-box'>
        <h2>{total_pessoas}</h2>
        <span>Indivíduos identificados</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-box'>
            <h2>{total_lisas}</h2>
            <span>Peças Lisas</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-box'>
            <h2>{total_estampadas}</h2>
            <span>Peças Estampadas</span>
        </div>
        """, unsafe_allow_html=True)

# Entrada Webcam
if fonte == "Webcam":
    if st.button("▶️ Iniciar Webcam"):
        cap = cv2.VideoCapture(0)
        stop = st.button("⏹️ Parar")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break
            frame = cv2.resize(frame, (720, 480))
            processed, registros = process_frame(frame)
            frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            if registros:
                mostrar_registros(registros)
        cap.release()
        cv2.destroyAllWindows()

# Entrada Vídeo
elif fonte == "Vídeo" and arquivo is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(arquivo.read())
    cap = cv2.VideoCapture(tfile.name)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (720, 480))
        processed, registros = process_frame(frame)
        frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        if registros:
            mostrar_registros(registros)
    cap.release()
    cv2.destroyAllWindows()
    os.remove(tfile.name)

# Entrada Imagem
elif fonte == "Imagem" and arquivo is not None:
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame = cv2.resize(frame, (720, 480))
    processed, registros = process_frame(frame)
    frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")
    if registros:
        mostrar_registros(registros)

st.write("---")

st.markdown("""
<p style='text-align: left; font-size: 12px; color: #C8C8C9;'>
🔗 Acesse o repositório no GitHub e veja mais detalhes no <a href="https://github.com/ivan-bezerra-filho/VestAI/blob/main/README.md" target="_blank" style="color: #FFFFFF; text-decoration: underline;">README do projeto</a>.
</p>
""", unsafe_allow_html=True)
