import kagglehub

def download():
    path = kagglehub.competition_download('march-machine-learning-mania-2026')

    print("Path to files:", path)
    return path