import cv2
import mediapipe as mp
import rtmidi

# ============================================================
#                 PERFORMANCE OPTIONS (TOGGLES)
# ============================================================

USE_MODEL_COMPLEXITY_0 = True
USE_FRAME_SKIP = True
FRAME_SKIP_RATE = 2

USE_MJPEG = True
USE_DRAWING = True
USE_TEXT = True

# CC1 smoothing
USE_CC_SMOOTHING = True
ALPHA = 0.3
smoothed_cc1 = 0

# ============================================================
#                 MEDIAPIPE + MIDI SETUP
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---------------- MIDI SETUP ----------------
midi_out = rtmidi.MidiOut()
ports = midi_out.get_ports()

print("\nAvailable MIDI output ports:\n")
for i, name in enumerate(ports):
    print(f"  {i}: {name}")

print("\nSelect a port number, or press ENTER to create a virtual port.")
choice = input("> ").strip()

if choice == "":
    print("Creating virtual port: HandController")
    midi_out.open_virtual_port("HandController")
else:
    try:
        index = int(choice)
        midi_out.open_port(index)
        print(f"Opened MIDI port: {ports[index]}")
    except:
        print("Invalid selection. Creating virtual port instead.")
        midi_out.open_virtual_port("HandController")

def send_cc(cc, value):
    midi_out.send_message([0xB0, cc, value])

def send_note_on(note):
    midi_out.send_message([0x90, note, 100])

def send_note_off(note):
    midi_out.send_message([0x80, note, 0])

# Chord dictionary (right hand)
CHORDS = {
    1: [60, 64, 67],   # C major
    2: [64, 67, 71],   # E minor
    3: [65, 69, 72],   # F major
    4: [67, 71, 74],   # G major
    5: [69, 72, 76],   # A minor
}

# Bass notes (left hand) two octaves below chord roots
BASS_NOTES = {
    1: 60 - 24,   # C
    2: 64 - 24,   # E
    3: 65 - 24,   # F
    4: 67 - 24,   # G
    5: 69 - 24,   # A
}

current_chord = None
current_bass = None

# ============================================================
#                 CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

if USE_MJPEG:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(1 if hand.landmark[tips[0]].x <
                       hand.landmark[tips[0]-1].x else 0)

    # Other fingers
    for tip in tips[1:]:
        fingers.append(1 if hand.landmark[tip].y <
                           hand.landmark[tip-2].y else 0)

    return sum(fingers)

# ============================================================
#                 MEDIAPIPE HANDS INITIALIZATION
# ============================================================

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0 if USE_MODEL_COMPLEXITY_0 else 1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

# ============================================================
#                 MAIN LOOP
# ============================================================

frame_counter = 0

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    # Frame skipping
    frame_counter += 1
    if USE_FRAME_SKIP and (frame_counter % FRAME_SKIP_RATE != 0):
        continue

    # Downscale
    small = cv2.resize(frame, (160, 120))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    right_fingers = None
    right_height = None
    left_height = None

    if results.multi_hand_landmarks:
        for handedness, handLms in zip(results.multi_handedness,
                                       results.multi_hand_landmarks):

            label = handedness.classification[0].label

            if label == "Right":
                right_fingers = fingers_up(handLms)
                right_height = handLms.landmark[0].y
                if USE_TEXT:
                    cv2.putText(small, f"R: {right_fingers}", (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if label == "Left":
                left_height = handLms.landmark[0].y
                if USE_TEXT:
                    cv2.putText(small, f"L Y: {left_height:.2f}", (5, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            if USE_DRAWING:
                mp_draw.draw_landmarks(small, handLms, mp_hands.HAND_CONNECTIONS)

    # ============================================================
    #                 RIGHT HAND → CC1 (Y position)
    # ============================================================

    if right_height is None:
        target_cc = 0
    else:
        target_cc = int((1.0 - right_height) * 127)
        target_cc = max(0, min(127, target_cc))

    if USE_CC_SMOOTHING:
        smoothed_cc1 = int(smoothed_cc1 + ALPHA * (target_cc - smoothed_cc1))
        send_cc(1, smoothed_cc1)
    else:
        send_cc(1, target_cc)

    # ============================================================
    #                 RIGHT HAND → CHORDS
    # ============================================================

    if right_fingers is None or right_fingers == 0:
        if current_chord in CHORDS:
            for n in CHORDS[current_chord]:
                send_note_off(n)
            current_chord = None
    elif right_fingers in CHORDS:
        if current_chord != right_fingers:
            if current_chord in CHORDS:
                for n in CHORDS[current_chord]:
                    send_note_off(n)
            for n in CHORDS[right_fingers]:
                send_note_on(n)
            current_chord = right_fingers

    # ============================================================
    #                 LEFT HAND → BASS NOTES
    # ============================================================

    if left_height is None:
        if current_bass is not None:
            send_note_off(current_bass)
            current_bass = None
    else:
        if current_chord in BASS_NOTES:
            bass_note = BASS_NOTES[current_chord]
            if current_bass != bass_note:
                if current_bass is not None:
                    send_note_off(current_bass)
                send_note_on(bass_note)
                current_bass = bass_note

    cv2.imshow("Hands", small)
    if cv2.waitKey(1) == 27:
        break

# turn off notes on exit
if current_chord in CHORDS:
    for n in CHORDS[current_chord]:
        send_note_off(n)

if current_bass is not None:
    send_note_off(current_bass)
