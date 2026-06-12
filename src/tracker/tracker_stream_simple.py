import openvr
import time
import sys

def iniciar_tracking():
    # 1. Inicializar SteamVR en modo aplicación de fondo (Background)
    try:
        vr_system = openvr.init(openvr.VRApplication_Background)
        print("=" * 50)
        print("¡Conexión exitosa con SteamVR!")
        print("=" * 50)
    except openvr.OTFError as e:
        print(f"Error crítico al conectar con SteamVR: {e}")
        print("Asegúrate de que SteamVR esté abierto y en verde.")
        sys.exit(1)

    print("Buscando rastreadores activos... Mueve el tracker para activarlo.\n")

    try:
        while True:
            # 2. Capturar las matrices de pose de todos los dispositivos posibles
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )
            
            # 3. Recorrer los slots de dispositivos de SteamVR
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                device_class = vr_system.getTrackedDeviceClass(i)
                
                # Filtrar solo dispositivos que actúen como Trackers o Controles
                if device_class in [openvr.TrackedDeviceClass_Controller, openvr.TrackedDeviceClass_GenericTracker]:
                    pose = poses[i]
                    
                    # Solo leemos datos si el tracking es válido (Estado 4 en tus logs)
                    if pose.bPoseIsValid:
                        # Extraer la matriz de transformación de 3x4
                        matrix = pose.mDeviceToAbsoluteTracking
                        
                        # Coordenadas espaciales en metros (X: Izquierda/Derecha, Y: Altura, Z: Profundidad)
                        x = matrix[0][3]
                        y = matrix[1][3]
                        z = matrix[2][3]
                        
                        # --- AGREGADO: Obtener porcentaje de batería ---
                        try:
                            bateria_float = vr_system.getFloatTrackedDeviceProperty(
                                i, openvr.Prop_DeviceBatteryPercentage_Float
                            )
                            bateria = f"{int(bateria_float * 100)}%"
                        except Exception:
                            bateria = "N/A"
                        
                        # Limpiar la línea e imprimir en tiempo real en la misma fila (incluyendo batería)
                        print(f"[ID {i}] -> X: {x: .3f}m | Y (Altura): {y: .3f}m | Z: {z: .3f}m | Batería: {bateria}", end="\r")
            
            # Frecuencia de muestreo (0.02s = ~50 Hz, ideal para análisis cinemático inicial)
            time.sleep(0.02)

    except KeyboardInterrupt:
        # Cerrar la conexión limpiamente al presionar Ctrl + C
        openvr.shutdown()
        print("\n" + "=" * 50)
        print("Streaming finalizado correctamente.")
        print("=" * 50)

if __name__ == "__main__":
    iniciar_tracking()