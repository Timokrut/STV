# STEP 1
# python -m pip install opencv-python numpy

# STEP 2
import cv2

cap = cv2.VideoCapture(0) # 0 - iPhone camera || 1 - Built in camera 
print('Камера открыта:', cap.isOpened())
cap.release()

# STEP 3
cap = cv2.VideoCapture(0)
for h, w, f in [(640, 480, 30), (1280, 720, 30), (1920, 1080, 30)]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, h)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, w)
    cap.set(cv2.CAP_PROP_FPS, f)

    print('Width:', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('Height:', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('FPS:', cap.get(cv2.CAP_PROP_FPS))

print('Brightness:', cap.get(cv2.CAP_PROP_BRIGHTNESS))
print('Contrast:', cap.get(cv2.CAP_PROP_CONTRAST))
print('Exposure:', cap.get(cv2.CAP_PROP_EXPOSURE))
print('Focus:', cap.get(cv2.CAP_PROP_FOCUS))
print('IOS EXPOSURE:', cap.get(cv2.CAP_PROP_IOS_DEVICE_EXPOSURE))
print('IOS FOCUS:', cap.get(cv2.CAP_PROP_IOS_DEVICE_FOCUS))
print('IOS WHITEBALANCE:', cap.get(cv2.CAP_PROP_IOS_DEVICE_WHITEBALANCE))
print('FPS:', cap.get(cv2.CAP_PROP_FPS))

cap.set(cv2.CAP_PROP_FPS, 120)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)

print('FPS:', cap.get(cv2.CAP_PROP_FPS))



# STEP 4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('capture.mp4', fourcc, 30.0, (1920, 1080))
while True:
    ok, frame = cap.read()
    if not ok:
        break

    out.write(frame)
    
    cv2.imshow('Preview', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# STEP 6
cap.release()
cv2.destroyAllWindows()