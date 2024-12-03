import cv2
import mediapipe as mp
import os
from tqdm import tqdm
from deepface import DeepFace

def detect_actions_and_emotions(video_path, output_path):
    # Inicializar MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Inicializar contadores de ações e emoções
    action_counts = {
        "waving": 0,
        "writing": 0,
        "using_cellphone": 0,
        "dancing": 0,
        "grimace": 0,
        "walking": 0,
        "raising_arm": 0,
        "greeting": 0,
        "lying_down": 0
    }

    # Inicializar contadores de emoções
    emotion_counts = {
        "angry": 0,
        "disgust": 0,
        "fear": 0,
        "happy": 0,
        "sad": 0,
        "surprise": 0,
        "neutral": 0
    }

    # Variáveis para rastreamento de estado
    previous_states = {action: False for action in action_counts}

    # Processar vídeo com barra de progresso
    for _ in tqdm(range(total_frames), desc="Processando o vídeo"):
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detecção de ações com MediaPipe
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            # Funções de detecção de ações
            def is_waving():
                return landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y or \
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            def is_raising_arm():
                return landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y or \
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            def is_walking():
                return abs(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y - landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y) > 0.05

            def is_using_cellphone():
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
                return abs(right_hand.x - right_ear.x) < 0.1 and abs(right_hand.y - right_ear.y) < 0.1

            def is_lying_down():
                shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
                knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
                return abs(shoulder_y - hip_y) < 0.05 and abs(hip_y - knee_y) < 0.05
            
            def is_dancing():
                return abs(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y - landmarks[mp_pose.PoseLandmark.LEFT_HIP].y) > 0.1 or \
                       abs(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y - landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) > 0.1

            def is_grimacing():
                nose = landmarks[mp_pose.PoseLandmark.NOSE]
                left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
                return abs(nose.x - left_ear.x) < 0.05 or abs(nose.x - right_ear.x) < 0.05

            def is_greeting():
                left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
                return (abs(left_hand.x - left_ear.x) < 0.1 and abs(left_hand.y - left_ear.y) < 0.1) or \
                       (abs(right_hand.x - right_ear.x) < 0.1 and abs(right_hand.y - right_ear.y) < 0.1)

            def is_writing():
                right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                return abs(right_hand.y - right_hip.y) < 0.1

            actions = {
                "waving": is_waving,
                "raising_arm": is_raising_arm,
                "walking": is_walking,
                "using_cellphone": is_using_cellphone,
                "lying_down": is_lying_down,
                "dancing": is_dancing,
                "grimace": is_grimacing,
                "greeting": is_greeting,
                "writing": is_writing
            }

            for action, detector in actions.items():
                if detector() and not previous_states[action]:
                    action_counts[action] += 1
                    previous_states[action] = True
                elif not detector():
                    previous_states[action] = False

        
        # Analisar o frame para detectar faces e expressões
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Iterar sobre cada face detectada pelo DeepFace
        for face in result:
            # Obter a caixa delimitadora da face
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            
            # Obter a emoção dominante
            dominant_emotion = face['dominant_emotion']
            emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
            
            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Escrever a emoção dominante acima da face
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            print(f"Emoções: {dominant_emotion}")          

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    cap.release()
    out.release()

    # Exibir contagem final de ações e emoções
    print("Ações detectadas:")
    for action, count in action_counts.items():
        print(f"{action}: {count}")

    print("\nEmoções detectadas:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count}")

    # Salvar relatório
    report_path = os.path.join(os.path.dirname(output_path), 'action_and_emotion_report.txt')
    with open(report_path, 'w') as report_file:
        report_file.write("Ações detectadas no vídeo:\n")
        for action, count in action_counts.items():
            report_file.write(f"{action}: {count}\n")

        report_file.write("\nEmoções detectadas no vídeo:\n")
        for emotion, count in emotion_counts.items():
            report_file.write(f"{emotion}: {count}\n")
    print(f"Relatório salvo em: {report_path}")

# Caminho do vídeo de entrada e saída
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')
output_video_path = os.path.join(script_dir, 'output_video.mp4')

# Executar a função
detect_actions_and_emotions(input_video_path, output_video_path)