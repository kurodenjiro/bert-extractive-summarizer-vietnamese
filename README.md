# Bert Extractive Summarizer For Vietnamese

This repo is the generalization of the lecture-summarizer repo. This tool utilizes the HuggingFace Pytorch transformers library
to run extractive summarizations. This works by first embedding the sentences, then running a clustering algorithm, finding 
the sentences that are closest to the cluster's centroids. This library also uses coreference techniques, utilizing the 
https://github.com/huggingface/neuralcoref library to resolve words in summaries that need more context. The greedyness of 
the neuralcoref library can be tweaked in the SingleModel class.

Paper: https://arxiv.org/abs/1906.04165

## Install

#### NOTE: You will need spacy 2.1.3 installed. There is currently an issue with Spacy 2.1.4 that produces segmentation faults. 

With that in mind, the setup.py should install 2.1.3 by default.
```bash
pip install spacy==2.1.3
pip install transformers
```

## How to Use

#### Simple Example
```python
from summarizer import Summarizer

body = 'Text body that you want to summarize with BERT'
body2 = 'Something else you want to summarize with BERT'
model = Summarizer()
model(body)
model(body2)
```

#### Large Example

```python
from summarizer import Summarizer

body = '''
Mèo, chính xác hơn là mèo nhà để phân biệt với các loài trong họ Mèo khác, là động vật có vú nhỏ và ăn thịt, sống chung với loài người, được nuôi để săn vật gây hại hoặc làm thú nuôi. Người ta tin rằng tổ tiên trung gian gần nhất trước khi được thuần hóa của chúng là mèo rừng châu Phi (Felis silvestris lybica). Mèo nhà đã sống gần gũi với loài người ít nhất 9.500 năm, và hiện nay chúng là con vật cưng phổ biến nhất trên thế giới.
Có rất nhiều các giống mèo khác nhau, một số không có lông hoặc không có đuôi, và chúng tồn tại với rất nhiều màu lông. Mèo là những con vật có kỹ năng của thú săn mồi và được biết đến với khả năng săn bắt hàng nghìn loại sinh vật để làm thức ăn. Chúng đồng thời là những sinh vật thông minh, và có thể được dạy hay tự học cách sử dụng các công cụ đơn giản như mở tay nắm cửa hay giật nước trong nhà vệ sinh.
Mèo giao tiếp bằng cách kêu meo, gừ-gừ, rít, gầm gừ và ngôn ngữ cơ thể. Mèo trong các bầy đàn sử dụng cả âm thanh lẫn ngôn ngữ cơ thể để giao tiếp với nhau.
Giống như một số động vật đã thuần hóa khác (như ngựa), mèo vẫn có thể sống tốt trong môi trường hoang dã như mèo hoang. Trái với quan niệm thông thường của mọi người rằng mèo là loài động vật cô độc, chúng thường tạo nên các đàn nhỏ trong môi trường hoang dã.
Sự kết hợp giữa con người và loài mèo dẫn tới việc nó thường được khắc họa trong các truyền thuyết và thần thoại tại nhiều nền văn hoá, gồm truyền thuyết và thần thoại Ai Cập cổ, Trung Quốc cổ, Na Uy cổ, và vị Vua xứ Wales thời Trung Cổ, Hywel Dda (người Tử tế) đã thông qua bộ luật bảo vệ động vật đầu tiên trên thế giới bằng cách đặt ra ngoài vòng pháp luật hành động giết hại hay làm tổn hại tới mèo, với những hình phạt nặng nề cho những kẻ vi phạm. Tuy nhiên, mèo thỉnh thoảng bị coi là ma quỷ, ví dụ như nó không mang lại may mắn hay thường đi liền với những mụ phù thuỷ trong nhiều nền văn hoá Trung cổ.
Cho đến gần đây, mèo được cho rằng đã được thuần hóa trong thời kỳ Ai Cập cổ đại, nơi chúng được thờ cúng.[6] Một nghiên cứu năm 2007 chỉ ra rằng tất cả mèo nhà có thể xuất phát từ Mèo hoang châu Phi tự thuần hóa (Felis silvestris lybica) vào khoảng 8000 TCN, tại Cận Đông.[3] Bằng chứng gần đây chỉ ra sự thuần hóa mèo là thi thể một con mèo con được chôn với chủ của nó cách đây 9.500 năm tại Síp.
Mèo là một trong mười hai con giáp tại Việt Nam, thường gọi là "Mão" hay "Mẹo".
'''

model = Summarizer()
result = model(body, min_length=60)
full = ''.join(result)
print(full)
"""
Mèo , chính_xác hơn là mèo nhà để phân_biệt với các loài trong họ Mèo khác , là động_vật có vú nhỏ và ăn thịt , sống chung với loài_người , được nuôi để săn vật gây hại hoặc làm thú nuôi . Trái với quan_niệm thông_thường của mọi người rằng mèo là loài động_vật cô_độc , chúng thường tạo nên các đàn nhỏ trong môi_trường hoang_dã .
"""
```

## Summarizer Options

```
model = Summarizer(
    model: str #This gets used by the hugging face bert library to load the model, you can supply a custom trained model here
    hidden: int # Needs to be negative, but allows you to pick which layer you want the embeddings to come from.
    reduce_option: str # It can be 'mean', 'median', or 'max'. This reduces the embedding layer for pooling.
    greedyness: float # number between 0 and 1. It is used for the coreference model. Anywhere from 0.35 to 0.45 seems to work well.
)

model(
    body: str # The string body that you want to summarize
    ratio: float # The ratio of sentences that you want for the final summary
    min_length: int # Parameter to specify to remove sentences that are less than 40 characters
    max_length: int # Parameter to specify to remove sentences greater than the max length
)
```

## Running the Service

There is a provided flask service and corresponding Dockerfile. Running the service is simple, and can be done though 
the Makefile with the two commands:

```
make docker-service-build
make docker-service-run
```

This will use the Bert-base-uncased model, which has a small representation. The docker run also accepts a variety of 
arguments for custom and different models. This can be done through a command such as:

```
docker build -t summary-service -f Dockerfile.service ./
docker run --rm -it -p 5000:5000 summary-service:latest -model bert-large-uncased
```

Other arguments can also be passed to the server. Below includes the list of available arguments.

* -greediness: Float parameter that determines how greedy nueralcoref should be
* -reduce: Determines the reduction statistic of the encoding layer (mean, median, max).
* -hidden: Determines the hidden layer to use for embeddings (default is -2)
* -port: Determines the port to use.
* -host: Determines the host to use.

Once the service is running, you can make a summarization command at the `http://localhost:5000/summarize` endpoint. 
This endpoint accepts a text/plain input which represents the text that you want to summarize. Parameters can also be 
passed as request arguments. The accepted arguments are:

* ratio: Ratio of sentences to summarize to from the original body. (default to 0.2)
* min_length: The minimum length to accept as a sentence. (default to 25)
* max_length: The maximum length to accept as a sentence. (default to 500)

An example of a request is the following:

```
POST http://localhost:5000/summarize?ratio=0.1

Content-type: text/plain

Body:
Mèo, chính xác hơn là mèo nhà để phân biệt với các loài trong họ Mèo khác, là động vật có vú nhỏ và ăn thịt, sống chung với loài người, được nuôi để săn vật gây hại hoặc làm thú nuôi. Người ta tin rằng tổ tiên trung gian gần nhất trước khi được thuần hóa của chúng là mèo rừng châu Phi (Felis silvestris lybica). Mèo nhà đã sống gần gũi với loài người ít nhất 9.500 năm, và hiện nay chúng là con vật cưng phổ biến nhất trên thế giới.
Có rất nhiều các giống mèo khác nhau, một số không có lông hoặc không có đuôi, và chúng tồn tại với rất nhiều màu lông. Mèo là những con vật có kỹ năng của thú săn mồi và được biết đến với khả năng săn bắt hàng nghìn loại sinh vật để làm thức ăn. Chúng đồng thời là những sinh vật thông minh, và có thể được dạy hay tự học cách sử dụng các công cụ đơn giản như mở tay nắm cửa hay giật nước trong nhà vệ sinh.
Mèo giao tiếp bằng cách kêu meo, gừ-gừ, rít, gầm gừ và ngôn ngữ cơ thể. Mèo trong các bầy đàn sử dụng cả âm thanh lẫn ngôn ngữ cơ thể để giao tiếp với nhau.
Giống như một số động vật đã thuần hóa khác (như ngựa), mèo vẫn có thể sống tốt trong môi trường hoang dã như mèo hoang. Trái với quan niệm thông thường của mọi người rằng mèo là loài động vật cô độc, chúng thường tạo nên các đàn nhỏ trong môi trường hoang dã.
Sự kết hợp giữa con người và loài mèo dẫn tới việc nó thường được khắc họa trong các truyền thuyết và thần thoại tại nhiều nền văn hoá, gồm truyền thuyết và thần thoại Ai Cập cổ, Trung Quốc cổ, Na Uy cổ, và vị Vua xứ Wales thời Trung Cổ, Hywel Dda (người Tử tế) đã thông qua bộ luật bảo vệ động vật đầu tiên trên thế giới bằng cách đặt ra ngoài vòng pháp luật hành động giết hại hay làm tổn hại tới mèo, với những hình phạt nặng nề cho những kẻ vi phạm. Tuy nhiên, mèo thỉnh thoảng bị coi là ma quỷ, ví dụ như nó không mang lại may mắn hay thường đi liền với những mụ phù thuỷ trong nhiều nền văn hoá Trung cổ.
Cho đến gần đây, mèo được cho rằng đã được thuần hóa trong thời kỳ Ai Cập cổ đại, nơi chúng được thờ cúng.[6] Một nghiên cứu năm 2007 chỉ ra rằng tất cả mèo nhà có thể xuất phát từ Mèo hoang châu Phi tự thuần hóa (Felis silvestris lybica) vào khoảng 8000 TCN, tại Cận Đông.[3] Bằng chứng gần đây chỉ ra sự thuần hóa mèo là thi thể một con mèo con được chôn với chủ của nó cách đây 9.500 năm tại Síp.
Mèo là một trong mười hai con giáp tại Việt Nam, thường gọi là "Mão" hay "Mẹo".

Response:

{
    "summary": "Mèo , chính_xác hơn là mèo nhà để phân_biệt với các loài trong họ Mèo khác , là động_vật có vú nhỏ và ăn thịt , sống chung với loài_người , được nuôi để săn vật gây hại hoặc làm thú nuôi . Trái với quan_niệm thông_thường của mọi người rằng mèo là loài động_vật cô_độc , chúng thường tạo nên các đàn nhỏ trong môi_trường hoang_dã ."
}
```


