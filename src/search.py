from pathlib import Path

import numpy as np
import faiss


class FSearch:
    def __init__(self, feat_path, img_path, d = 4096, debug = False, v2 = False):
        self.d = d  #dimension of feature vectors
        self.feat_path = feat_path
        self.img_path = img_path
        self.img_paths = list(sorted(img_path.glob("*")))
        self.index = self.load_index() if not v2 else self.load_index_v2()
        self.debug = debug
    
    def load_index(self):
        index = faiss.IndexFlatL2(self.d)   # build the index
        for feature_path in sorted(self.feat_path.glob("*.npy")):
            index.add(np.load(feature_path).reshape(1,-1))     # add vectors to the index

        assert index.ntotal == len(self.img_paths) - 1, "Index not created" # -1 for an extra .keep file
        return index
    
    def query(self, query, top_k = 5):
        assert query.shape[1] == self.d, f"Query vector does not match feature vector shape {query.shape[1]}!={self.d}"
        D, I = self.index.search(query, top_k)
        if self.debug:
            return D, I
        return None, I
    
    def load_index_v2(self):
        nlist= 3  #no. of clusters
        quantiser = faiss.IndexFlatL2(self.d)   # build the index
        index = faiss.IndexIVFFlat(quantiser, self.d, nlist, faiss.METRIC_L2)
        data = []
        for feature_path in sorted(self.feat_path.glob("*.npy")):
            data.append(np.load(feature_path))           # add feature vectors to the index
        data = np.array(data)
        assert not index.is_trained
        index.train(data)
        assert index.is_trained
        index.add(data)
        index.nprobe = 2  #how many clusters needed in result
        assert index.ntotal == len(self.img_paths) - 1, "Index not created" # -1 for an extra .keep file
        return index


if __name__ == "__main__":
    from src.extractor import FeatureExtractor

    IMG_PATH = Path("./static/imgs")
    FEATURE_PATH = Path("./static/features")
    QUERY_PATH = Path("./static/uploaded/5549f026.jpg")

    fe=FeatureExtractor()
    img_feats = fe.extract(QUERY_PATH).numpy().reshape(1, -1)

    fs = FSearch(FEATURE_PATH, IMG_PATH)
    response = fs.query(img_feats)
    print(response)