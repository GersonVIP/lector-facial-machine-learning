import cv2
import mediapipe as mp
import time
from pathlib import Path


# ---------------------------------------------------------
# 1. Buscar el modelo
# ---------------------------------------------------------

RUTA_MODELO = Path("modelos/face_landmarker.task")

if not RUTA_MODELO.exists():
    print("ERROR: No se encontró el modelo:")
    print(RUTA_MODELO)
    exit()


# ---------------------------------------------------------
# 2. Configurar MediaPipe Face Landmarker
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

opciones = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(RUTA_MODELO)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ---------------------------------------------------------
# 3. Abrir cámara
# ---------------------------------------------------------

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada correctamente.")
print("MediaPipe Face Landmarker iniciado.")
print("Presiona ESC para salir.")


# ---------------------------------------------------------
# 4. Crear Face Landmarker
# ---------------------------------------------------------

with FaceLandmarker.create_from_options(opciones) as landmarker:

    while True:

        # Obtener imagen de la cámara
        ret, frame = camara.read()

        if not ret:
            print("ERROR: No se pudo obtener el frame.")
            break

        # Voltear horizontalmente para una vista tipo espejo
        frame = cv2.flip(frame, 1)

        # OpenCV utiliza BGR
        # MediaPipe necesita RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convertir a imagen de MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Timestamp en milisegundos
        timestamp_ms = time.monotonic_ns() // 1_000_000

        # Detectar landmarks
        resultado = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # -------------------------------------------------
        # 5. Dibujar landmarks
        # -------------------------------------------------

        if resultado.face_landmarks:

            # Tomamos el primer rostro
            landmarks = resultado.face_landmarks[0]

            # Dibujar todos los puntos
            for landmark in landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Evitar puntos fuera de la imagen
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            # Mostrar cantidad de landmarks
            texto = f"Landmarks detectados: {len(landmarks)}"

            cv2.putText(
                frame,
                texto,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "No se detecta ningun rostro",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # Mostrar imagen
        cv2.imshow(
            "MediaPipe - Face Landmarks",
            frame
        )

        # ESC para salir
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            break


# ---------------------------------------------------------
# 6. Liberar cámara
# ---------------------------------------------------------

camara.release()
cv2.destroyAllWindows()

print("Programa finalizado.")