import cv2
import face_recognition as fr
import os
import numpy
import datetime



dia = datetime.date.today()
dia_semana = dia.weekday()
calendario = {0: 'lunes',
              1: 'martes',
              2: 'miércoles',
              3: 'jueves',
              4: 'viérnes',
              5: 'sábado',
              6: 'domingo'}


hora = datetime.datetime.now()
momento = ''
if hora.hour < 6 or hora.hour > 20:
    momento = 'Buenas Noches!'
elif 6 <= hora.hour <= 12:
    momento = 'Buenos Dias!'
else:
    momento = 'Buenas Tardes'



# crear data base
ruta = 'empleados'
mis_img = []
trabajadores = []
lista_trab = os.listdir(ruta)

for nombre in lista_trab:
    img_actual = cv2.imread(f'{ruta}/{nombre}')
    mis_img.append(img_actual)
    trabajadores.append(os.path.splitext(nombre)[0])

print(trabajadores)

# cod imgs
def codificar(imgs):

    #crear una list nueva
    lista_codif = []

    #pasar todas las img a rgb
    for img in imgs:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #codif rostros:
        codificado = fr.face_encodings(img)[0]

        #agregar a la list
        lista_codif.append(codificado)

    #devolver lista codif
    return lista_codif


#registrar los ingresos
def reg_ingreso(persona):
    f = open('regis')


lista_trab_codif = codificar(mis_img)
#print(len(lista_trab_codif))


# Tomar img de cam web
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# leer img de la cam
exito, img = captura.read()

if not exito:
    print('No se ha podido tomar captura')
else:
    #reconocer rostro en captura
    rostro_captura = fr.face_locations(img)

    #codf rostro capturado
    rostro_captura_codif = fr.face_encodings(img, rostro_captura)

    # buscar coincidencias
    for rostrocod, rostroubic in zip(rostro_captura_codif, rostro_captura):
        coincidencias = fr.compare_faces(lista_trab_codif, rostrocod)
        distancias = fr.face_distance(lista_trab_codif, rostrocod)

        print(distancias)

        indice_coincidencia = numpy.argmin(distancias)

        #mostrar coincidencias
        if distancias[indice_coincidencia] > 0.6:
            print('No coincide con ninguno de nuestros empleados')
        else:
            # nombre del trabajador encontrado
            nombre = trabajadores[indice_coincidencia]
            y1, x2, y2, x1 = rostroubic
            cv2.rectangle(img,
                          (x1, y1), (x2, y2),
                          (0, 255, 0),
                          2)

            cv2.rectangle(img,
                          (x1, y2 - 35),
                          (x2, y2),
                          (0, 255, 0),
                          cv2.FILLED
                          )
            cv2.putText(img,
                        f'{momento} {nombre}, Feliz {calendario[dia_semana]}!',
                        (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        (255, 255, 255),
                        1)

            print(f'{momento} BIENVENID@ {nombre} AL TRABAJO!! - Mooy! Feliz {calendario[dia_semana]} !!')

            #mostrar imagen obtenida
            cv2.imshow('Imagen Web', img)

            #mantener ventana
            cv2.waitKey(0)