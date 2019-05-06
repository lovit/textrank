```python
import numpy as np

z = np.random.random_sample((5,3))
z
```

```
array([[0.04656175, 0.20145282, 0.67823318],
       [0.33197532, 0.48793961, 0.75195233],
       [0.51498774, 0.58938071, 0.98792026],
       [0.45006027, 0.70745067, 0.15536108],
       [0.42664766, 0.66697394, 0.94806522]])
```

```python
rows = np.asarray([0, 4, 1])
cols = np.asarray([1, 2, 0])
z[rows, cols]
```

```
array([0.20145282, 0.94806522, 0.33197532])
```

Pairwise distance matrix 는 numpy.ndarray 의 형태이며, numpy.where 를 이용하여 특정 조건을 만족하는 값들의 rows, cols 를 가져온 뒤, 이 값을 각각 row 와 column 위치에 입력하면 해당 값이 slice 된다.

pairwise distances 함수 결과는 n by n 크기의 행렬이기 때문에, 데이터가 클 경우에는 부분적으로 나눠서 행렬을 만든 뒤, vstack 을 이용하여 이를 합치는게 더 좋다.

```python
def cosine_similarity_matrix(x, min_sim=0.3, batch_size=1000):
    n_rows = x.shape[0]
    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx+1) * batch_size))
        psim = 1 - pairwise_distances(x[b:e], x, metric='cosine')
        rows, cols = np.where(psim >= min_sim)
        data = psim[rows, cols]
        mat.append(csr_matrix((data, (rows, cols)), shape=(e-b, n_rows)))
    mat = sp.sparse.vstack(mat)
    return mat
```

collections 의 Counter 를 이용하여 구현한 Cosine sentence similarity 함수를 이용한 경우 [(commit)](https://github.com/lovit/textrank/blob/c4fb13a0070167a55bef2c89dec219fd0867c2c8/textrank/sentence.py#L89) 약 21 분 23 초의 계산이 걸리던 작업이 numpy 를 이용한 경우에는 3.31 초로 개선되었다.

```python
def textrank_similarity_matrix(x, min_sim=0.3, batch_size=1000):
    n_rows, n_cols = x.shape

    # Boolean matrix
    rows, cols = x.nonzero()
    data = np.ones(rows.shape[0])
    z = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    # Inverse sentence length
    size = np.asarray(x.sum(axis=1)).reshape(-1)
    size[np.where(size <= min_length)] = 10000
    size = np.log(size)

    mat = []
    for bidx in range(math.ceil(n_rows / batch_size)):

        # slicing
        b = int(bidx * batch_size)
        e = min(n_rows, int((bidx+1) * batch_size))

        # dot product
        inner = z[b:e,:] * z.transpose()

        # sentence len[i,j] = size[i] + size[j]
        norm = size[b:e].reshape(-1,1) + size.reshape(1,-1)
        norm = norm ** (-1)
        norm[np.where(norm == np.inf)] = 0

        # normalize
        sim = inner.multiply(norm).tocsr()
        rows, cols = (sim >= min_sim).nonzero()
        data = np.asarray(sim[rows, cols]).reshape(-1)

        # append
        mat.append(csr_matrix((data, (rows, cols)), shape=(e-b, n_rows)))

    return sp.sparse.vstack(mat)
```

두 문장에 공통으로 등장한 단어의 개수는 term frequency matrix 를 Boolean matrix 로 변환한 뒤, 이를 내적하면 알 수 있다. 하지만 두 문장 길이의 로그값의 합으로 normalize 하는 부분을 구현하는 부분은 행렬의 곲으로는 구현되지 않는데, norm(i,j) = size(i) + size(j) 형식이기 때문이다. norm 을 계산한 뒤, Boolean matrix 간의 inner product 와 element-wise muptiplication 을 한 뒤, scipy.sparse 의 where 를 실행하여 min_sim 이상의 값만으로 이뤄진 새로운 sparse matrix 를 만든다. sparse matrix 를 stack 에 쌓은 뒤, 병합하여 return 한다.

Python 의 set 을 이용하여 구현한 TextRank sentence similarity 함수를 이용한 경우 [(commit)](https://github.com/lovit/textrank/blob/c4fb13a0070167a55bef2c89dec219fd0867c2c8/textrank/sentence.py#L69) 약 3분 20 초의 계산이 걸리던 작업이 numpy 를 이용한 경우에는 24 초로 개선되었다.

Cosine 의 경우에는 numpy 에서 모든 계산이 끝나는 것과 비교하여, TextRank similarity 는 n by n 의 행렬을 메모리에 올리지 않기 위해서는 Python 과 numpy 를 왔다갔다 하는 작업을 몇 번 거쳐야 하기 때문에 계산 속도의 감소 폭이 적었다.
