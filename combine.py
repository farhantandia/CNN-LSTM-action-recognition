from moviepy.editor import VideoFileClip, concatenate_videoclips
video_1 = VideoFileClip("back.avi")
video_2 = VideoFileClip("free.avi")
video_3 = VideoFileClip("breast.avi")


final_video= concatenate_videoclips([video_1, video_2, video_3])
final_video.write_videofile("final_video.mp4")