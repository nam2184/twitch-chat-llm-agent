from enum import Enum

class SearchType(Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"

class SearchKwargs(Enum):
    K = "k"
    FETCH_K = "fetch_k"
    LAMBDA_MULT = "lambda_mult"

