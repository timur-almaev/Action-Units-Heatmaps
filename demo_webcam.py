
# TODO: print short help on startup, i.e. CTRL+C to kill safely, if verbose it
# can be done via pressing Q with the figure in focus or simply by closing
# the figure.

import sys, os, time, platform, cv2, dlib, numpy as np, matplotlib.pyplot as plt
import AUmaps

keep = True
def handle_close(evt):
	global keep
	keep = False

def main():
	verbose = True
	if len(sys.argv) > 1 and sys.argv[1] == '-q':
		verbose = False

	print(' ** Loading model ... ')
	AUdetector = AUmaps.AUdetector('shape_predictor_68_face_landmarks.dat', enable_cuda=True)
	cam = cv2.VideoCapture(0)

	fig = plt.figure(figsize=plt.figaspect(.5))
	fig.canvas.mpl_connect('close_event', handle_close)
	axs = fig.subplots(5, 2)

	# Init subplots and images within
	implots = []
	for ax in axs.reshape(-1):
	    ax.axis('off')
	    implots.append(ax.imshow(np.zeros((256, 256))))

	clearscr = True
	try:
		global keep
		tstart_time = time.time()
		nframes = 0
		while keep:
			start_time = time.time()
			_, img = cam.read()

			# Downscale webcam image 2x to speed things up
			img = cv2.resize(img, None, fx = 0.5, fy = 0.5)

			# Optionally flip webcam image, probably not relevant
			# img = cv2.flip(img, 1)

			# Optionally display webcam image with opencv
			# cv2.imshow('Action Unit Heatmaps - Press Q to exit!', img)
			# if cv2.waitKey(1) & 0xFF == ord('q'):
			# 	break

			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			nframes += 1

			pred,map,img = AUdetector.detectAU(img)

			if len(pred) == 0:
				sys.stdout.write("\r |   FALSE    |   --   |   --   |   --   |   --   |   --   |     --      |")
				sys.stdout.flush()
				continue

			if clearscr:
				clearscr = False
				if platform.system() is 'Windows':
					os.system('cls')
				else:
					os.system('clear')
				# Print table hat
				sys.stdout.write("  _______________________________________________________________________ \n")
				sys.stdout.write(" | Face Found |  AU06  |  AU10  |  AU12  |  AU14  |  AU17  | FPS Elapsed |\n")

			if verbose:
				for j in range(0,5):
					resized_map = dlib.resize_image(map[j,:,:].cpu().data.numpy(),rows=256,cols=256)

			        # Update face image subplot
					implots[2*j].set_data(img)

			        # Update heatmap subplot
					implots[2*j+1].set_data(resized_map)

					# Set correct heatmap limits
					resized_map_flat = resized_map.flatten()
					implots[2*j+1].set_clim(min(resized_map_flat), max(resized_map_flat))

			        # To plot heatmaps the original way - looks identical to the new way!
			        # ax = fig.add_subplot(5,2,2*j+1)
			        # ax.imshow(resized_map)
			        # ax.axis('off')

				plt.pause(0.001)
				# plt.show(block=False)
				plt.draw()

			elapsed_time = 1.0 / (time.time() - start_time)
			sys.stdout.write("\r |    TRUE    | %6.3f | %6.3f | %6.3f | %6.3f | %6.3f |   %7.3f   |"
			 % (pred[0], pred[1], pred[2], pred[3], pred[4], elapsed_time))
			sys.stdout.flush()

	except KeyboardInterrupt:
	    pass

	# Close camera
	cam.release()

	# If webcam images shown with opencv, close window
	# cv2.destroyAllWindows()

	telapsed_time = time.time() - tstart_time
	print('\n ** Mean FPS Elapsed: {0:.3f} \n'.format(1.0 / (telapsed_time / nframes)))

if __name__ == "__main__":
    main()
