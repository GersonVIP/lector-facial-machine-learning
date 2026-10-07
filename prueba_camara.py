import cv2

# Abrir la cámara predeterminada
camara = cv2.VideoCapture(0)

# Verificar que la cámara pudo abrirse
if not camara.isOpened():
    print("Error: no se pudo abrir la cámara.")
    exit()

print("Cámara iniciada correctamente.")
print("Presiona la tecla ESC para salir.")

while True:
    # Capturar un frame
    ret, frame = camara.read()

    if not ret:
        print("Error: no se pudo obtener la imagen.")
        break

    # Mostrar imagen
    cv2.imshow("Prueba de camara", frame)

    # Salir al presionar ESC
    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
        break

# Liberar recursos
camara.release()
cv2.destroyAllWindows()