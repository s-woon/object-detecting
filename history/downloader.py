import pytube

yt = pytube.YouTube("https://www.youtube.com/watch?v=QQEoMLCtmgg&ab_channel=SPOTV")

yt.streams.filter(adaptive=True, file_extension='mp4', only_video=True).order_by('resolution').desc().first().download('./video')