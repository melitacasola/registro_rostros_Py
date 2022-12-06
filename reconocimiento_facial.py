import cv2
import face_recognition as fr


#cargar images
foto_control = fr.load_image_file('foto4.jpeg')
foto_prueba = fr.load_image_file('foto3.jpg')

#formato de fotos (RGB)
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

#localizar rostro control
lugar_rostro_A = fr.face_locations(foto_control)[0]
rostro_codificado_A = fr.face_encodings(foto_control)[0]

lugar_rostro_B = fr.face_locations(foto_prueba)[0]
rostro_codificado_B = fr.face_encodings(foto_prueba)[0]

# mostrar rectangulos
cv2.rectangle(foto_prueba,
              (lugar_rostro_B[3], lugar_rostro_B[0]),
              (lugar_rostro_B[1], lugar_rostro_B[2]),
              (0, 255, 0),
              2)

cv2.rectangle(foto_control,
              (lugar_rostro_A[3], lugar_rostro_A[0]),
              (lugar_rostro_A[1], lugar_rostro_A[2]),
              (0, 255, 0),
              2)

#realizar comparacion
resultado = fr.compare_faces([rostro_codificado_A], rostro_codificado_B) ##con coma, agrego un valor mas en 0.6 -->
# es la distancia subir o bajar valores para generar mayor o menor coincidencia


#medida de la distancia
distancia = fr.face_distance([rostro_codificado_A], rostro_codificado_B)


# mostrar resultado
cv2.putText(foto_prueba,
            f'{resultado} {distancia.round(2)}',
            (50, 50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2) #(50,50) es la ubi del texto - 1 escala de texto, 2 grosor


#mostrar img
cv2.imshow('foto control', foto_control)
cv2.imshow('foto prueba', foto_prueba)


#mantener prog abierto
cv2.waitKey(0)