"""Actividad 3: Kuramoto global. Ejecutar: python actividad3_kuramoto.py."""
from pathlib import Path
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button


def simular(omega, theta0, K, T=80.0, dt=0.02):
    """RK4 simultáneo; devuelve tiempo, fases no envueltas, r y frecuencias medias."""
    omega = np.asarray(omega, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)
    if omega.ndim != 1 or omega.size < 2 or omega.shape != theta0.shape:
        raise ValueError('omega y theta0 deben ser vectores iguales, con N >= 2.')
    if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(theta0)):
        raise ValueError('Las entradas deben ser finitas.')
    if not all(np.isfinite(x) for x in (K, T, dt)) or K < 0 or T <= 0 or dt <= 0:
        raise ValueError('Se requiere K >= 0, T > 0 y dt > 0, todos finitos.')
    pasos = int(round(T / dt))
    if pasos < 2:
        raise ValueError('Se requieren al menos dos pasos temporales.')
    t = np.arange(pasos + 1) * dt
    theta = np.empty((pasos + 1, omega.size))
    theta[0] = theta0

    def derivada(fases):
        c, s = np.mean(np.cos(fases)), np.mean(np.sin(fases))
        return omega + K * (s * np.cos(fases) - c * np.sin(fases))

    for n in range(pasos):
        x = theta[n]
        a = derivada(x)
        b = derivada(x + dt*a/2)
        c = derivada(x + dt*b/2)
        d = derivada(x + dt*c)
        theta[n+1] = x + dt*(a + 2*b + 2*c + d)/6
    r = np.abs(np.mean(np.exp(1j*theta), axis=1))
    inicio = pasos // 2
    omega_obs = (theta[-1] - theta[inicio]) / (t[-1] - t[inicio])
    return t, theta, r, omega_obs


def experimento(N=100, sigma=1.0, semilla=42, T=80.0, dt=0.02,
                valores_K=(0.0, 0.5, 1.5, 3.0), carpeta='resultados'):
    if N < 5 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Se requiere N >= 5 y sigma >= 0 finito.')
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    omega = rng.normal(0, sigma, N)
    theta0 = rng.uniform(0, 2*np.pi, N)
    # La media poblacional de referencia es cero; no se fuerza la media muestral.
    resultados = {float(K): simular(omega, theta0, K, T, dt) for K in valores_K}
    fig, ejes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    filas = []
    for K, (t, theta, r, obs) in resultados.items():
        inicio = len(t) // 2
        filas.append([K, r[inicio:].mean(), r[inicio:].std(), obs.std()])
        ejes[0, 0].plot(t, r, label=f'K={K:g}')
        ejes[0, 1].scatter(omega, obs, s=15, alpha=.65, label=f'K={K:g}')
    ejes[0, 0].set(title='Evolución de la coherencia', xlabel='Tiempo (s)',
                   ylabel='r(t)', ylim=(0, 1.02))
    ejes[0, 0].axvspan(T/2, T, alpha=.08, color='gray', label='Ventana de medida')
    ejes[0, 1].set(title='Bloqueo de frecuencias', xlabel='Frecuencia natural (rad/s)',
                   ylabel='Frecuencia media observada (rad/s)')
    Kmax = max(resultados)
    t, theta, r, _ = resultados[Kmax]
    final = t >= max(0, t[-1]-10)
    for j in range(5):
        ejes[1, 0].plot(t[final], np.sin(theta[final, j]), label=f'Oscilador {j+1}')
    ejes[1, 0].set(title=f'Cinco señales: K={Kmax:g}', xlabel='Tiempo (s)',
                   ylabel='sin(fase)')
    datos = np.asarray(filas)
    ejes[1, 1].errorbar(datos[:, 0], datos[:, 1], yerr=datos[:, 2],
                       marker='o', capsize=4, label='Media ± desviación temporal')
    ejes[1, 1].set(title='Resumen tras el transitorio', xlabel='K (s⁻¹)',
                   ylabel='Coherencia r', ylim=(0, 1.05))
    for ax in ejes.flat:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(carpeta/'comparacion_kuramoto.png', dpi=160)
    with (carpeta/'indicadores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['K', 'r_media', 'r_desviacion_temporal', 'dispersion_frecuencias_observadas'])
        writer.writerows(filas)
    print('\n K      r media    desv. temporal r   dispersión frecuencias')
    for K, media, desv, disp in filas:
        print(f'{K:4.1f}    {media:8.5f}       {desv:8.5f}           {disp:8.5f}')

    # Verificación numérica del mismo experimento con medio paso.
    _, _, rf, of = simular(omega, theta0, Kmax, T, dt/2)
    error_r = np.max(np.abs(r - rf[::2]))
    error_o = np.max(np.abs(resultados[Kmax][3] - of))
    texto = (f'K={Kmax:g}; dt={dt:g} frente a dt/2={dt/2:g}\n'
             f'Máxima diferencia de r en tiempos comunes: {error_r:.8g}\n'
             f'Máxima diferencia de frecuencias medias: {error_o:.8g} rad/s\n'
             'Las barras muestran variación temporal, no incertidumbre entre repeticiones.\n')
    (carpeta/'verificacion_paso.txt').write_text(texto, encoding='utf-8')
    print('\n'+texto)
    return omega, resultados, fig


def animar(omega, resultados, controles=True, max_cuadros=320):
    """Animación de trayectorias precomputadas; seleccionar K reinicia la vista."""
    claves = list(resultados)
    fig, (ax, ar) = plt.subplots(1, 2, figsize=(11, 5))
    fig.subplots_adjust(bottom=.18, right=.83 if controles else .97, wspace=.35)
    angulo = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(angulo), np.sin(angulo), color='.7')
    ax.set(aspect='equal', xlim=(-1.3, 1.3), ylim=(-1.3, 1.3),
           xlabel='cos(fase)', ylabel='sin(fase)', title='Fases de los osciladores')
    puntos = ax.scatter([], [], s=35)
    puntos.set_array(omega)
    puntos.set_cmap('viridis')
    minimo, maximo = omega.min(), omega.max()
    puntos.set_clim(minimo, maximo if maximo > minimo else minimo+1)
    fig.colorbar(puntos, ax=ax, shrink=.65, label='Frecuencia natural (rad/s)')
    vector, = ax.plot([], [], color='#e76f51', lw=3, label='Vector medio')
    ax.legend(loc='lower left', fontsize=8)
    linea, = ar.plot([], [], color='#087f8c', lw=2)
    marca, = ar.plot([], [], 'o', color='#e76f51')
    ar.set(xlabel='Tiempo (s)', ylabel='r(t)', ylim=(0, 1.02))
    ar.grid(alpha=.3)
    titulo = fig.suptitle('')
    estado = {'K': claves[-1], 'pausa': False, 'cursor': 0}
    cantidad = min(max_cuadros, len(resultados[claves[0]][0]))

    def dibujar(_):
        t, theta, r, _obs = resultados[estado['K']]
        indices = np.linspace(0, len(t)-1, cantidad, dtype=int)
        idx = indices[estado['cursor']]
        fases = theta[idx]
        puntos.set_offsets(np.column_stack((np.cos(fases), np.sin(fases))))
        z = np.mean(np.exp(1j*fases))
        vector.set_data([0, z.real], [0, z.imag])
        linea.set_data(t[:idx+1], r[:idx+1])
        marca.set_data([t[idx]], [r[idx]])
        ar.set_xlim(0, t[-1])
        titulo.set_text(f"Kuramoto | K={estado['K']:g} | t={t[idx]:.2f} s | r={r[idx]:.3f}")
        if not estado['pausa']:
            estado['cursor'] = (estado['cursor']+1) % cantidad
        return puntos, vector, linea, marca, titulo

    # init_func evita consumir un cuadro antes de empezar.
    animacion = FuncAnimation(fig, dibujar, frames=cantidad, interval=50,
                             init_func=lambda: (), blit=False, repeat=True)
    if controles:
        radio = RadioButtons(fig.add_axes([.85, .47, .13, .27]),
                             [f'{k:g}' for k in claves], active=len(claves)-1)
        radio.ax.set_title('Acoplamiento K', fontsize=10)
        boton = Button(fig.add_axes([.85, .30, .13, .08]), 'Pausa / seguir')
        reinicio = Button(fig.add_axes([.85, .18, .13, .08]), 'Reiniciar')
        def seleccionar(etiqueta):
            estado.update(K=float(etiqueta), cursor=0)
            fig.canvas.draw_idle()
        def pausar(_):
            estado['pausa'] = not estado['pausa']
        def volver(_):
            estado['cursor'] = 0
        radio.on_clicked(seleccionar)
        boton.on_clicked(pausar)
        reinicio.on_clicked(volver)
        fig._controles_kuramoto = (radio, boton, reinicio)
    fig.text(.03, .035, 'Reproducción acelerada. La longitud del vector medio es r. '
             'El color identifica la frecuencia natural.', fontsize=9)
    return fig, animacion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true', help='Solo exportar resultados estáticos')
    parser.add_argument('--html', action='store_true', help='Exportar animación autónoma HTML para K=3')
    parser.add_argument('--salida', default='resultados')
    args = parser.parse_args()
    omega, resultados, fig = experimento(carpeta=args.salida)
    animacion = None  # Mantener una referencia durante plt.show().
    if args.html:
        fh, ah = animar(omega, resultados, controles=False, max_cuadros=160)
        Path(args.salida, 'animacion_kuramoto.html').write_text(ah.to_jshtml(), encoding='utf-8')
        plt.close(fh)
    if not args.no_show:
        fa, animacion = animar(omega, resultados)
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
