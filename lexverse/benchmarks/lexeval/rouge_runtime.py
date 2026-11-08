"""Keep plain rouge's LCS semantics while avoiding its recursive backtrace."""


def _iterative_reconstruction(x, y, exclusive=True):
    from rouge import rouge_score
    table = rouge_score._lcs(x, y)
    i, j = len(x), len(y)
    words = []
    while i and j:
        if x[i - 1] == y[j - 1]:
            words.append(x[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1, j] > table[i, j - 1]:
            i -= 1
        else:
            j -= 1
    words.reverse()
    return rouge_score.Ngrams(words, exclusive=exclusive)


def install_iterative_backtrace():
    # Assignment is idempotent. Concurrent scorers use the same pure implementation.
    from rouge import rouge_score
    rouge_score._recon_lcs = _iterative_reconstruction
