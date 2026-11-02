# Actividad 3: simulación del modelo de Kuramoto

Requisitos: Python 3.10 o posterior, NumPy y Matplotlib.

```bash
pip install numpy matplotlib
python actividad3_kuramoto.py
```

Se abren una comparación de resultados y una animación circular. Seleccione K con los botones de la derecha; cada selección vuelve al tiempo inicial. Los controles permiten pausar y reiniciar. La animación reproduce trayectorias ya calculadas y acelera el tiempo simulado.

Para generar resultados sin interfaz gráfica:

```bash
python actividad3_kuramoto.py --no-show
```

Para exportar además una animación que puede abrir en el navegador sin conexión (K=3):

```bash
python actividad3_kuramoto.py --no-show --html
```

Los resultados se guardan en `resultados/`: gráfica PNG, indicadores CSV y comprobación del paso temporal; con --html se añade la animación. Los controles de selección de K pertenecen a la ventana Python; el HTML tiene controles de reproducción para un solo K.

## Google Colab o Jupyter
Abra `Actividad3_Kuramoto.ipynb`. Contiene el código integrado, las gráficas y una celda que presenta la animación mediante controles de reproducción HTML. Cambie `K_animacion` para animar otro caso.

## Parámetros y modelo
N=100, sigma=1 rad/s, semilla=42, T=80 s, dt=0.02 s y K=(0, 0.5, 1.5, 3). Puede modificarlos en la llamada a `experimento`. Cada K usa exactamente las mismas frecuencias y fases iniciales.

Modelo: d(theta_i)/dt = omega_i + (K/N) sum_j sin(theta_j - theta_i).
Integración: Runge–Kutta de orden 4, con actualización simultánea. Las fases se conservan sin envolver. El modelo se expresa en un marco con frecuencia media poblacional cero; valores negativos indican ritmos más lentos que esa referencia.

## Qué observar
- r(t): concentración de fases, entre 0 y 1.
- Frecuencias observadas frente a naturales: una banda horizontal indica osciladores con frecuencia media similar.
- Cinco señales sin(theta) durante los últimos diez segundos, para el mayor K.
- Media y desviación temporal de r en la segunda mitad. Las barras NO son intervalos de confianza ni variabilidad entre semillas.
- Verificación de dt frente a dt/2 para el mayor K, usando las mismas condiciones iniciales.

r alto no demuestra por sí solo bloqueo de todos los osciladores; r bajo puede ocultar grupos en oposición. La ventana temporal de medida debe ampliarse si persisten transitorios.

## Trabajo del estudiante
Interprete las gráficas, examine los indicadores, compare el paso temporal y justifique qué casos muestran coherencia de fase y/o bloqueo de frecuencia. Las cinco señales mostradas son los primeros cinco osciladores, sin seleccionar previamente los más sincronizados.
