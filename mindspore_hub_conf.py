"""hub config"""
from src.skipgram import SkipGram

def skipgram(*args, **kwargs):
    return SkipGram(*args, **kwargs)


def create_network(name, *args, **kwargs):
    if name == "skipgram":
        return SkipGram(*args, **kwargs)
    raise NotImplementedError(f"{name} is not implemented in the repo")
