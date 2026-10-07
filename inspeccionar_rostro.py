import cv2
import mediapipe as mp
import time
from pathlib import Path


# ---------------------------------------------------------
# 1. Ruta del modelo
# ---------------------------------------------------------

RUTA_MODELO = Path("modelos/face_landmarker.task")

if not RUTA_MODELO.exists():
    print("ERROR: No se encontró el modelo.")
    exit()


# ---------------------------------------------------------
# 2. Configuración de Face Landmarker
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 3. Abrir cámara
# ---------------------------------------------------------

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada.")
print("Presiona ESC para salir.")


# ---------------------------------------------------------
# 4. Crear detector
# ---------------------------------------------------------

with FaceLandmarker.create_from_options(opciones) as landmarker:

    datos_mostrados = False

    while True:

        ret, frame = camara.read()

        if not ret:
            print("ERROR: No se pudo obtener la imagen.")
            break

        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = time.monotonic_ns() // 1_000_000

        resultado = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # -------------------------------------------------
        # 5. Mostrar información del rostro
        # -------------------------------------------------

        if resultado.face_landmarks:

            landmarks = resultado.face_landmarks[0]

            cv2.putText(
                frame,
                f"Landmarks: {len(landmarks)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Mostrar datos solamente una vez
            if not datos_mostrados:

                print("\n==============================")
                print("INFORMACION DEL ROSTRO")
                print("==============================")

                print(f"Cantidad de landmarks: {len(landmarks)}")

                print("\nPrimeros 5 landmarks:\n")

                for i in range(min(5, len(landmarks))):

                    punto = landmarks[i]

                    print(
                        f"Landmark {i}: "
                        f"x={punto.x:.4f}, "
                        f"y={punto.y:.4f}, "
                        f"z={punto.z:.4f}"
                    )

                print("\n==============================\n")

                datos_mostrados = True

        else:

            cv2.putText(
                frame,
                "No se detecta rostro",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Inspeccion de landmarks",
            frame
        )

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            break


camara.release()
cv2.destroyAllWindows()

print("Programa finalizado.")