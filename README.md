---
pipeline_tag: feature-extraction
tags:
- feature-extraction
- transformers
license: apache-2.0
language:
- id
metrics:
- accuracy
- f1
- precision
- recall
datasets:
- squad_v2
---
### indo-dpr-question_encoder-single-squad-base
<p style="font-size:16px">Indonesian Dense Passage Retrieval trained on translated SQuADv2.0 dataset in DPR format.</p>


### Evaluation 

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| hard_negative | 0.9963 | 0.9963 | 0.9963 | 183090 |
| positive | 0.8849 | 0.8849 | 0.8849 | 5910 |

| Metric | Value |
|--------|-------|
| Accuracy | 0.9928 |
| Macro Average | 0.9406 |
| Weighted Average | 0.9928 |

<p style="font-size:16px">Note: This report is for evaluation on the dev set, after 12000 batches.</p>

### Usage

```python
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('firqaaa/indo-dpr-question_encoder-single-squad-base')
model = DPRQuestionEncoder.from_pretrained('firqaaa/indo-dpr-question_encoder-single-squad-base')
input_ids = tokenizer("Ibukota Indonesia terletak dimana?", return_tensors='pt')["input_ids"]
embeddings = model(input_ids).pooler_output
```

We can use it using `haystack` as follows:

```
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore

retriever = DensePassageRetriever(document_store=InMemoryDocumentStore(),
                                  query_embedding_model="firqaaa/indo-dpr-question_encoder-single-squad-base",
                                  passage_embedding_model="firqaaa/indo-dpr-question_encoder-single-squad-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=256,
                                  batch_size=16,
                                  use_gpu=True,
                                  embed_title=True,
                                  use_fast_tokenizers=True)
```
