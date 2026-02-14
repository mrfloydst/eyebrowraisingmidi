import cv2
import mediapipe as mp
import rtmidi
import numpy as np

LEFT_EYE_LIDS  = [159, 145, 33, 133]
RIGHT_EYE_LIDS = [386, 374, 362, 263]

BLINK_THRESHOLD = 0.20
BLINK_COOLDOWN = 10   # frames
blink_timer = 0

def eye_aspect_ratio(pts, eye_indices):
    # eye_indices: [upper_lid, lower_lid, left_corner, right_corner]
    upper = pts[eye_indices[0]]
    lower = pts[eye_indices[1]]
    left  = pts[eye_indices[2]]
    right = pts[eye_indices[3]]

    vertical = abs(upper[1] - lower[1])
    horizontal = abs(left[0] - right[0])

    if horizontal == 0:
        return 1.0

    return vertical / horizontal


# ============================================================
#                 MIDI SETUP
# ============================================================

midi_out = rtmidi.MidiOut()
ports = midi_out.get_ports()

print("\nAvailable MIDI output ports:\n")
for i, name in enumerate(ports):
    print(f"  {i}: {name}")

print("\nSelect a port number, or press ENTER to create a virtual port.")
choice = input("> ").strip()

if choice == "":
    print("Creating virtual port: EyebrowController")
    midi_out.open_virtual_port("EyebrowController")
else:
    try:
        index = int(choice)
        midi_out.open_port(index)
        print(f"Opened MIDI port: {ports[index]}")
    except:
        print("Invalid selection. Creating virtual port instead.")
        midi_out.open_virtual_port("EyebrowController")

def send_cc(cc, value):
    midi_out.send_message([0xB0, cc, value])

def send_pitchbend(value):
    # value: 0–16383
    lsb = value & 0x7F
    msb = (value >> 7) & 0x7F
    midi_out.send_message([0xE0, lsb, msb])

# ============================================================
#                 MEDIAPIPE FACE MESH
# ============================================================

mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
#                 CAMERA
# ============================================================

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# ============================================================
#                 EYEBROW LANDMARK INDICES
# ============================================================

LEFT_EYEBROW = [52, 65, 55]
LEFT_EYE = [159, 145]

RIGHT_EYEBROW = [282, 295, 285]
RIGHT_EYE = [386, 374]

# ============================================================
#                 SMOOTHING
# ============================================================

ALPHA = 0.3
smooth_mod = 0
smooth_bend = 8192

# ============================================================
#                 MAIN LOOP
# ============================================================

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    small = cv2.resize(frame, (320, 240))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        # Convert to numpy for convenience
        pts = np.array([(lm.x, lm.y) for lm in face.landmark])

        # LEFT EYEBROW HEIGHT
        left_brow_y = np.mean(pts[LEFT_EYEBROW][:,1])
        left_eye_y = np.mean(pts[LEFT_EYE][:,1])
        left_raise = (left_eye_y - left_brow_y) * 1000  # scale

        # RIGHT EYEBROW HEIGHT
        right_brow_y = np.mean(pts[RIGHT_EYEBROW][:,1])
        right_eye_y = np.mean(pts[RIGHT_EYE][:,1])
        right_raise = (right_eye_y - right_brow_y) * 1000

        # --- BLINK DETECTION ---
        left_ear = eye_aspect_ratio(pts, LEFT_EYE_LIDS)
        right_ear = eye_aspect_ratio(pts, RIGHT_EYE_LIDS)

        ear = (left_ear + right_ear) / 2

        if blink_timer > 0:
            blink_timer -= 1

        # Trigger note when eyes close
        if ear < BLINK_THRESHOLD and blink_timer == 0:
            # MIDI channel 2 = 0x91 (note on, channel 2)
            midi_out.send_message([0x91, 60, 100])   # Note on, middle C
            midi_out.send_message([0x81, 60, 0])     # Note off
            blink_timer = BLINK_COOLDOWN

        # Normalize to MIDI
        mod_target = int(np.clip(left_raise * 2, 0, 127))
        bend_target = int(np.clip(8192 + (right_raise - 20) * 200, 0, 16383))

        # Smoothing
        smooth_mod = int(smooth_mod + ALPHA * (mod_target - smooth_mod))
        smooth_bend = int(smooth_bend + ALPHA * (bend_target - smooth_bend))

        # Send MIDI
        send_cc(1, smooth_mod)
        send_pitchbend(smooth_bend)

        # Debug text
        cv2.putText(small, f"Mod: {smooth_mod}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(small, f"Bend: {smooth_bend}", (5, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Eyebrow MIDI", small)
    if cv2.waitKey(1) == 27:
        break
