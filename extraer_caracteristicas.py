import cv2
import mediapipe as mp
import time
import csv
from pathlib import Path
import numpy as np


# =========================================================
# CONFIGURACIÓN
# =========================================================

RUTA_MODELO = Path("modelos/face_landmarker.task")
RUTA_DATOS = Path("datos/muestra_rostro.csv")

if not RUTA_MODELO.exists():
    print("ERROR: No se encontró el modelo:")
    print(RUTA_MODELO)
    exit()


# =========================================================
# FUNCIÓN PARA NORMALIZAR LANDMARKS
# =========================================================

def normalizar_landmarks(landmarks):
    """
    Convierte los landmarks originales en un vector
    normalizado para que sea menos sensible a la posición
    y al tamaño del rostro dentro de la imagen.
    """

    puntos = np.array(
        [[p.x, p.y, p.z] for p in landmarks],
        dtype=np.float32
    )

    # -----------------------------------------------------
    # 1. Centrar los puntos
    # -----------------------------------------------------

    centro = np.mean(puntos, axis=0)

    puntos_centrados = puntos - centro

    # -----------------------------------------------------
    # 2. Calcular escala del rostro
    # -----------------------------------------------------

    distancias = np.linalg.norm(
        puntos_centrados,
        axis=1
    )

    escala = np.mean(distancias)

    if escala < 1e-8:
        raise ValueError("La escala del rostro es demasiado pequeña.")

    # -----------------------------------------------------
    # 3. Normalizar
    # -----------------------------------------------------

    puntos_normalizados = puntos_centrados / escala

    # -----------------------------------------------------
    # 4. Convertir a vector
    # -----------------------------------------------------

    vector = puntos_normalizados.flatten()

    return vector


# =========================================================
# CONFIGURAR MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


opciones = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(RUTA_MODELO)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# ABRIR CÁMARA
# =========================================================

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    exit()


print("--------------------------------------------")
print(" EXTRACCION DE CARACTERISTICAS FACIALES")
print("--------------------------------------------")
print()
print("Colócate frente a la cámara.")
print("Cuando detecte tu rostro:")
print()
print("   ESPACIO = guardar muestra")
print("   ESC     = salir")
print()


# =========================================================
# INICIAR LANDMARKER
# =========================================================

with FaceLandmarker.create_from_options(opciones) as landmarker:

    while True:

        ret, frame = camara.read()

        if not ret:
            print("ERROR: No se pudo obtener el frame.")
            break

        # Vista tipo espejo
        frame = cv2.flip(frame, 1)

        # BGR → RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convertir a imagen MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = time.monotonic_ns() // 1_000_000

        resultado = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # =================================================
        # ROSTRO DETECTADO
        # =================================================

        if resultado.face_landmarks:

            landmarks = resultado.face_landmarks[0]

            cv2.putText(
                frame,
                "ROSTRO DETECTADO",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "ESPACIO = guardar muestra",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Dibujar algunos puntos
            for punto in landmarks:

                x = int(punto.x * frame.shape[1])
                y = int(punto.y * frame.shape[0])

                if (
                    0 <= x < frame.shape[1]
                    and 0 <= y < frame.shape[0]
                ):
                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            # =============================================
            # ESPACIO = GUARDAR
            # =============================================

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 32:

                try:

                    vector = normalizar_landmarks(
                        landmarks
                    )

                    # Crear carpeta si no existe
                    RUTA_DATOS.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    # Guardar CSV
                    with open(
                        RUTA_DATOS,
                        "w",
                        newline=""
                    ) as archivo:

                        escritor = csv.writer(archivo)

                        escritor.writerow(
                            [
                                f"caracteristica_{i}"
                                for i in range(len(vector))
                            ]
                        )

                        escritor.writerow(vector)

                    print()
                    print("====================================")
                    print("MUESTRA GUARDADA CORRECTAMENTE")
                    print("====================================")
                    print(
                        f"Cantidad de características: {len(vector)}"
                    )
                    print(
                        f"Archivo: {RUTA_DATOS}"
                    )
                    print()

                    # Salimos después de guardar
                    break

                except Exception as error:

                    print(
                        f"ERROR al guardar la muestra: {error}"
                    )

            elif tecla == 27:
                break

        else:

            cv2.putText(
                frame,
                "NO SE DETECTA NINGUN ROSTRO",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                break

        cv2.imshow(
            "Extraccion de caracteristicas",
            frame
        )


# =========================================================
# FINALIZAR
# =========================================================

camara.release()
cv2.destroyAllWindows()

print("Programa finalizado.")