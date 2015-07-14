from scipy.stats import rankdata


def ensemble_px(pxx, wx):
    finalRank = 0
    assert(len(pxx)==len(wx))
    for px, w in zip(pxx, wx):
        finalRank = finalRank + rankdata(px, method='ordinal') * w
    finalRank = finalRank / (max(finalRank) + 1.0)
    return finalRank
