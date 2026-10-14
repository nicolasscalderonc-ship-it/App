import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import io

# Configuración de la página (usamos 'wide' para aprovechar todo el ancho de la pantalla)
st.set_page_config(
    page_title="Grabador de audio - MATLAB a Python",
    page_icon="🎙️",
    layout="wide"
)

# --- ESTILOS Y TÍTULO PRINCIPAL ---
st.title("Analizador de Señales de Voz y FFT")
st.markdown("Interfaz desarrollada para el análisis en el **Dominio del Tiempo** y la **Transformada de Fourier (FFT)** de vocales y consonantes.")
st.markdown("---")

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header(" Panel de Control")
st.sidebar.write("Configura tu análisis de audio:")

# Selector de fonema
fonema = st.sidebar.selectbox(
    "Selecciona el fonema a analizar:",
    ("A", "E", "I", "O", "U", "F", "S")
)

# Método de entrada
metodo_entrada = st.sidebar.radio(
    "Método de captura:",
    ("Grabar con el micrófono", "Subir archivo WAV")
)

st.sidebar.markdown("---")
st.sidebar.info(" **Consejo:** Pronuncia el fonema de forma sostenida durante unos segundos para obtener un mejor espectro en la FFT." \
"Jeison Steban Grajales Lopez, 121584"\
"Nicolás Javier Fonseca Alvarado, 125662")

# --- ÁREA PRINCIPAL ---
senal = None
fs = None

# Contenedor según el método elegido
if metodo_entrada == "Grabar con el micrófono":
    st.subheader(f"1. Grabación de la letra: **'{fonema}'**")
    audio_bytes = st.audio_input("Haz clic en el micrófono para grabar:")
   
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        fs, senal = wavfile.read(io.BytesIO(audio_bytes.read()))
       
else:
    st.subheader(f"1. Cargar Archivo para el Fonema: **'{fonema}'**")
    uploaded_file = st.file_uploader("Sube tu archivo de audio en formato .wav", type=["wav"])
    if uploaded_file is not None:
        fs, senal = wavfile.read(uploaded_file)
        st.audio(uploaded_file, format="audio/wav")

# --- PROCESAMIENTO MATEMÁTICO Y GRÁFICAS ---
if senal is not None:
    st.markdown("---")
    st.subheader("2. Analisis de señales")
   
    # Procesamiento mono y normalización
    if len(senal.shape) > 1:
        senal = senal[:, 0]
    senal = senal.astype(float)
   
    N = len(senal)
   
    if N > 0:
        # Cálculo de la FFT (idéntico a MATLAB)
        Y = np.fft.fft(senal)
        P2 = np.abs(Y / N)
        P1 = P2[:N // 2 + 1]
        P1[1:-1] = 2 * P1[1:-1]
        f = fs * np.arange(N // 2 + 1) / N
        t = np.arange(N) / fs

        # Dividimos la pantalla principal en dos columnas para las gráficas
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Dominio del Tiempo")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(t, senal, color='#2b5c8f', linewidth=1.2)
            ax1.set_xlabel('Tiempo (s)', fontsize=10)
            ax1.set_ylabel('Amplitud', fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.set_title(f'Señal Temporal - "{fonema}"', fontsize=11)
            st.pyplot(fig1)

        with col2:
            st.markdown("### Dominio de la Frecuencia (FFT)")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(f, P1, color='#d95f02', linewidth=1.2)
            ax2.set_xlabel('Frecuencia (Hz)', fontsize=10)
            ax2.set_ylabel('|X(f)|', fontsize=10)
            ax2.set_xlim(0, 5000)
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.set_title(f'Espectro de Fourier - "{fonema}"', fontsize=11)
            st.pyplot(fig2)
           
    else:
        st.warning("El audio capturado está vacío. Por favor, intenta grabar de nuevo.")
