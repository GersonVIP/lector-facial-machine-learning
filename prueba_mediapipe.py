import cv2
import mediapipe as mp
import time

# Clases de MediaPipe que vamos a utilizar
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

# Configuración del detector
options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path="modelos/blaze_face_short_range.tflite"
    ),
    running_mode=RunningMode.VIDEO,
    min_detection_confidence=0.5
)

# Abrir cámara
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("Error: no se pudo abrir la cámara.")
    exit()

print("Cámara iniciada.")
print("MediaPipe iniciando...")
print("Presiona ESC para salir.")

# Crear detector
with FaceDetector.create_from_options(options) as detector:

    frame_count = 0

    while True:

        ret, frame = camara.read()

        if not ret:
            print("Error: no se pudo obtener la imagen.")
            break

        # OpenCV trabaja normalmente en BGR.
        # MediaPipe espera una imagen RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convertir la imagen a formato MediaPipe
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Timestamp creciente para el modo VIDEO
        timestamp_ms = frame_count * 33
        frame_count += 1

        # Ejecutar detección
        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Si encontramos rostros
        if result.detections:

            for detection in result.detections:

                # Obtener recuadro del rostro
                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                w = bbox.width
                h = bbox.height

                # Dibujar rectángulo
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Obtener confianza
                confianza = detection.categories[0].score

                texto = f"Rostro detectado: {confianza:.2f}"

                cv2.putText(
                    frame,
                    texto,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        else:

            cv2.putText(
                frame,
                "No se detecta ningun rostro",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Mostrar resultado
        cv2.imshow("Detector facial - MediaPipe", frame)

        # ESC para salir
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == 27:
            break

camara.release()
cv2.destroyAllWindows()

print("Programa finalizado.")