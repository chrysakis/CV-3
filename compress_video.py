import cv2

# Parameters
frames_per_sec = 16
ratio = 0.60
resolution = (int(960 * ratio), int(720 * ratio))

# Open input and output streams
original = cv2.VideoCapture('../misc/sample.wmv')
fourcc = cv2.VideoWriter_fourcc(*'MP42')
out = cv2.VideoWriter('../misc/final_video.mp4v', fourcc, frames_per_sec,
                      resolution)

# Subsample video
interval = 128
frames_per_sec = 30
for i in range(interval * frames_per_sec):
    ret, frame = original.read()
    assert ret
    if i % 2 == 0:
        frame = cv2.resize(frame, None, fx=ratio, fy=ratio,
                           interpolation=cv2.INTER_CUBIC)
        out.write(frame)

# Close input and output streams
print(resolution)
original.release()
out.release()
