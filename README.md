## TextRank implementation

To summarize La La Land user comments by keyword extraction with part-of-speech tagged documents

```python
from textrank import KeywordSummarizer

docs = ['list of str form', 'sentence list']

keyword_extractor = KeywordSummarizer(
    tokenize = lambda x:x.split(),      # YOUR TOKENIZER
    window = -1,
    verbose = False
)

keywords = keyword_extractor.summarize(sents, topk=30)
for word, rank in keywords:
    # do something
```


You can specify word cooccurrence graph with arguments.

```python
from textrank import KeywordSummarizer

keyword_extractor = KeywordSummarizer(
    tokenize = lambda x:x.split()
    min_count=2,
    window=-1,                     # cooccurrence within a sentence
    min_cooccurrence=2,
    vocab_to_idx=None,             # you can specify vocabulary to build word graph
    df=0.85,                       # PageRank damping factor
    max_iter=30,                   # PageRank maximum iteration
    bias=None,                     # PageRank initial ranking
    verbose=False
)
```

To summarize La La Land user comments by key-sentence extraction, 


```python
from textrank import KeysentenceSummarizer

summarizer = KeysentenceSummarizer(
    tokenize = YOUR_TOKENIZER,
    min_sim = 0.5,
    verbose = True
)

keysents = summarizer.summarize(sents, topk=5)
for sent_idx, rank, sent in keysents:
    # do something
```

```
시사회 보고 왔어요 꿈과 사랑에 관한 이야기인데 뭔가 진한 여운이 남는 영화예요
시사회 갔다왔어요 제가 라이언고슬링팬이라서 하는말이아니고 너무 재밌어요 꿈과 현실이 잘 보여지는영화 사랑스런 영화 전 개봉하면 또 볼생각입니당
시사회에서 보고왔는데 여운쩔었다 엠마스톤과 라이언 고슬링의 케미가 도입부의 강렬한음악좋았고 예고편에 나왓던 오디션 노래 감동적이어서 눈물나왔다ㅠ 이영화는 위플래쉬처럼 꼭 영화관에봐야함 색감 노래 배우 환상적인 영화
방금 시사회로 봤는데 인생영화 하나 또 탄생했네 롱테이크 촬영이 예술 영상이 넘나 아름답고 라이언고슬링의 멋진 피아노 연주 엠마스톤과의 춤과 노래 눈과 귀가 호강한다 재미를 기대하면 약간 실망할수도 있지만 충분히 훌륭한 영화
황홀하고 따뜻한 꿈이었어요 imax로 또 보려합니다 좋은 영화 시사해주셔서 감사해요
```

You can also use KoNLPy as your tokenizer

```python
from konlpy.tag import Komoran

komoran = Komoran()
def komoran_tokenizer(sent):
    words = komoran.pos(sent, join=True)
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]
    return words

summarizer = KeysentenceSummarizer(
    tokenize = komoran_tokenizer,
    min_sim = 0.3,
    verbose = False
)

keysents = summarizer.summarize(sents, topk=3)
```