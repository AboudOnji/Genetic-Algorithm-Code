# Script de Gnuplot para graficar convergencia

# Configurar el delimitador de campos a coma (para CSV)
set datafile separator ","

# Título del gráfico
set title "Convergencia del Algoritmo Genético"

# Etiquetas de los ejes
set xlabel "Generación"
set ylabel "Aptitud Máxima"

# Activar la cuadrícula
set grid

# Definir el estilo de la línea (opcional)
set style line 1 lc rgb "blue" lw 2 pt 7 ps 1 # pt 7 es círculo, ps 1 es tamaño

# Nombre del archivo CSV (¡REEMPLAZA ESTO!)
csv_file = "GA_Convergencia_20250516_154905.csv"

# Comando para graficar: usa la columna 1 para X, columna 2 para Y
# 'skip 1' es para saltar la fila de encabezado si tu CSV la tiene (el nuestro la tiene)
plot csv_file using 1:2 with linespoints linestyle 1 title "Aptitud Máxima"

# Para mantener la ventana abierta en algunos sistemas (opcional):
# pause -1 "Presiona cualquier tecla para salir"

# Para guardar en un archivo PNG (descomenta y ajusta):
# set terminal pngcairo size 800,600 enhanced font "arial,10"
# set output "convergencia_ag.png"
# replot # Vuelve a dibujar con la nueva terminal y salida
# set output # Restaura la salida a la pantalla (si es necesario)
# set terminal X11 # O wxt, qt, aqua, dependiendo de tu sistema