from pytube import Search
from pytube import YouTube

def download_top_video(search_string):
    # Perform the search
    s = Search(search_string)
    for video in s.results:
        # Get the video's duration in seconds
        yt = YouTube(video.watch_url)
        if yt.length < 600:  # 10 minutes = 600 seconds
            print(f"Downloading: {yt.title}")
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
            print("Download completed!")
            break
    else:
        print("No suitable video found.")

if __name__ == "__main__":
    search_string = input("Enter the search string: ")
    download_top_video(search_string)

