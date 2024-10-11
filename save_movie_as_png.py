import cv2
import os

def save_movie_as_png(video_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Extract the base name of the video file (without extension)
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        exit()

    # Frame counter
    frame_count = 0

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()

        # If the frame is read correctly, ret will be True
        if not ret:
            break

        # Save the current frame as PNG
        frame_filename = os.path.join(output_dir, f'{video_name}_frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)

        print(f'Saved {frame_filename}')
        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f'Extracted {frame_count} frames and saved them as PNG files.')


# Specify the path to your video file
#video_file = r'.\movies\neutral\neutro_kiko.mp4'
#video_file = r'.\movies\laughing\leve_kiko.mp4'
#video_file = r'.\movies\happy\sorriso_kiko.mp4'
video_file = r'.\paiva_videos\smiling\paiva_gargalhada.mkv'

# Create a directory to store the frames
#output_dir = r'.\data2\neutral'
#output_dir = r'.\data2\laughing'
#output_dir = r'.\data2\happy'
output_dir = r'.\datadb2\laughing'

save_movie_as_png(video_file, output_dir)