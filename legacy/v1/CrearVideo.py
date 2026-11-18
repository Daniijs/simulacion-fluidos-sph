import os
import subprocess
import glob
import re

def crear_video():
    print("====================================================")
    print("    GENERADOR DE VIDEO - SIMULACION DE FLUIDOS      ")
    print("====================================================")
    
    # 1. Configurar rutas
    directorio_frames = "POV_Frames"
    patron_frames = os.path.join(directorio_frames, "render_*.png")
    video_salida = "simulacion_fluidos.mp4"
    
    # 2. Buscar y contar frames png
    frames = glob.glob(patron_frames)
    if not frames:
        print(f"[ERROR] No se encontraron imagenes PNG en la carpeta '{directorio_frames}'.")
        print("Asegurate de que POV-Ray haya renderizado los archivos y guardado los .png alli.")
        return
        
    print(f"[OK] Se encontraron {len(frames)} imagenes PNG en '{directorio_frames}'.")
    
    # Encontrar el frame maximo
    numeros = []
    for f in frames:
        match = re.search(r'render_(\d+)\.png', os.path.basename(f))
        if match:
            numeros.append(int(match.group(1)))
            
    if numeros:
        max_frame = max(numeros)
        min_frame = min(numeros)
        print(f"[OK] Rango de frames detectados: render_{min_frame:04d}.png hasta render_{max_frame:04d}.png")
    
    # 3. Configurar FPS (Frames por segundo)
    fps = 30
    print(f"[OK] Framerate configurado: {fps} FPS (~{len(frames)/fps:.2f} segundos de video)")
    
    # 4. Comando FFmpeg optimizado
    comando = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(directorio_frames, "render_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        video_salida
    ]
    
    print("\nEjecutando FFmpeg para renderizar el video...")
    print(f"Comando: {' '.join(comando)}\n")
    
    try:
        resultado = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if resultado.returncode == 0:
            print("====================================================")
            print("         ¡VIDEO GENERADO CON EXITO!")
            print(f"Video guardado como: '{video_salida}'")
            print(f"Ubicacion: {os.path.abspath(video_salida)}")
            print("====================================================")
        else:
            print("[ERROR] Error durante la generacion del video:")
            print(resultado.stderr)
    except FileNotFoundError:
        print("[ERROR] No se encontro 'ffmpeg' en las variables de entorno.")
        print("Por favor, asegurate de que FFmpeg este correctamente instalado.")

if __name__ == "__main__":
    crear_video()
