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
        self.feat_paths = []
        for feature_path in sorted(self.feat_path.glob("*.npy")):
            self.feat_paths.append(feature_path)
            index.add(np.load(feature_path).reshape(1,-1))     # add vectors to the index

        assert index.ntotal == len(self.img_paths) , "Index features and total images size does not match" # -1 for an extra .keep file
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
        assert index.ntotal == len(self.img_paths) , "Index features and total images size does not match" # -1 for an extra .keep file

        return index


if __name__ == "__main__":
    import cv2
    
    from extractor import FeatureExtractor
    from utils import *

    IMG_PATH = Path("../static/imgs")
    FEATURE_PATH = Path("../static/features")
    QUERY_PATH = Path("../static/imgs/5ab96e94cd72e62700584a06_rmr-blog-bird-oasis.jpg")

    fe=FeatureExtractor()
    img_feats = fe.extract(QUERY_PATH).numpy().reshape(1, -1)

    fs = FSearch(FEATURE_PATH, IMG_PATH, debug=True)
    D, response = fs.query(img_feats, 10)
    match_img_paths = [fs.img_paths[id] for id in response[0]]
    response = stack_images_side_by_side(QUERY_PATH, match_img_paths)
    print(D)
    cv2.imshow("response", response)
    cv2.waitKey(0)
    cv2.destroyAllWindows()