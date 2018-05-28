## TextRank implementation

For keyword selections, 

    from pprint import pprint
    from textrank import summarize_as_keywords

    sents = ['sent is list of str', 'this is example']
    keywords = summarize_as_keywords(sents)

You can specify word graph with arguments of summarize_as_keywords functions. If you use debug mode, it return not only keywords but ranks, vocabulary index, and word graph.

    keywords = summarize_as_keywords(
        sents,
        topk=50,
        tokenizer=lambda s:s.split(),
        min_count=10,
        min_cooccurrence=3,
        verbose=True,
        debug=True
    )

Keywords, returned variable is list of tuple form. Each tuple contains sentence and its score

    [('재배포', 0.538140850221495),
     ('무단', 0.46747526750507146),
     ('금지', 0.3797005381404317),
     ('뉴시스', 0.1889406802065108),
     ('공감', 0.10557622894724966),
     ('저작권자', 0.07848823486967427),
     ('영상', 0.050533308260353224),
     ('사진', 0.050099116749948054),
     ('기자', 0.04488792619512985),
     ('서울', 0.03798698215834762),
     ('매일경제', 0.03568735179483187),
     ('서울경제', 0.035317492192730525),
     ('머니투데이', 0.0343692938880199),
     ('독자', 0.0323318142332849),
     ('헤럴드경제', 0.02936172956817344),
     ('제보', 0.028581613186882144),
     ('뉴스', 0.028187297136703055),
     ('한국경제', 0.024748167201201682),
     ...
     ]

For extracting key-sentences, 

    keysentences = summarize_as_keysentences(
        sents,
        vocab2idx=vocab2idx,
        topk=3,
        verbose=False
    )

Keysentences, returned variable is also list of tuple form. Each tuple contains sentence and its score
