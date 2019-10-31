import cv2

path = "/Users/Melanie/Movies/Moth2/rotated/"
imgs = [cv2.imread(path + str(index).zfill(8) + "_r.png") for index in range(28119, 28411)]

vid_writer = cv2.VideoWriter('rotated.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (imgs[0].shape[1],imgs[0].shape[0]))
for img in imgs:
    vid_writer.write(img)
vid_writer.release()
