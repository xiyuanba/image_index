from jina import Flow, Executor, requests, DocumentArray, Document, Client
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BartTokenizer, BartModel, AutoTokenizer, pipeline, \
    AutoModelWithLMHead
from text2vec import SentenceModel, EncoderType
from tqdm import tqdm
from pprint import pprint


class MyEncoder(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    @requests
    def caption(self, docs: DocumentArray, **kwargs):
        for doc in tqdm(docs):
            img_path = doc.uri
            print(img_path)
            raw_image = Image.open(img_path).convert('RGB')
            # unconditional image captioning
            inputs = self.processor(raw_image, return_tensors="pt")

            out = self.model.generate(**inputs)
            print(self.processor.decode(out[0], skip_special_tokens=True))
            doc.text = self.processor.decode(out[0], skip_special_tokens=True)

    @requests(on="/search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs


class EnglishToChineseTranslator(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mode_name = 'liam168/trans-opus-mt-en-zh'
        model = AutoModelWithLMHead.from_pretrained(mode_name)
        self.tokenizer = AutoTokenizer.from_pretrained(mode_name)
        self.translation = pipeline("translation_en_to_zh", model=model, tokenizer=self.tokenizer)

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            # 执行翻译并将结果添加到文档
            translated_text = self.translation(doc.text, max_length=400)[0]['translation_text']
            doc.text = translated_text
            print(doc.summary())
            print(doc.text)

    @requests(on="/search")
    def search(self, docs: DocumentArray, **kwargs):
        return docs

class TextEncoder(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = SentenceModel("shibing624/text2vec-base-chinese", encoder_type=EncoderType.FIRST_LAST_AVG,
                                    device='cpu')
        self._da = DocumentArray(
            storage='sqlite', config={'connection': 'example.db', 'table_name': 'images'}
        )

    @requests
    def bar(self, docs: DocumentArray, **kwargs):
        print('start to index')
        print(f"Length of da_encode is {len(self._da)}")
        for doc in tqdm(docs):
            doc.embedding = self._model.encode(doc.text)
            print(doc.summary())
            with self._da:
                self._da.append(doc)
                self._da.sync()
                print(f"Length of da_encode is {len(self._da)}")
        print(f"Length of da_encode is {len(self._da)}")

    @requests(on='/search')
    def search(self, docs: DocumentArray, **kwargs):
        self._da.sync()  # Call the sync method
        print(f"Length of da_search is {len(self._da)}")
        for doc in docs:
            doc.embedding = self._model.encode(doc.text)
            print(doc.text)
            print(doc.summary())
            doc.match(self._da, limit=6, exclude_self=True, metric='cos', use_scipy=True)
            pprint(doc.matches[:, ('text', 'uri', 'scores__cos')])


class Search(Executor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = SentenceModel("shibing624/text2vec-base-chinese", encoder_type=EncoderType.FIRST_LAST_AVG,
                                    device='cpu')
        self._da = DocumentArray(
            storage='sqlite', config={'connection': 'example.db', 'table_name': 'images'}
        )
    @requests
    def search(self, docs: DocumentArray, **kwargs):
        self._da.sync()  # Call the sync method
        print(f"Length of da_search is {len(self._da)}")
        for doc in docs:
            doc.embedding = self._model.encode(doc.text)
            print(doc.text)
            print(doc.summary())
            doc.match(self._da, limit=6, exclude_self=True, metric='cos', use_scipy=True)
            pprint(doc.matches[:, ('text', 'uri', 'scores__cos')])



f_index = Flow().config_gateway(protocol='http', port=12345) \
    .add(name='caption', uses=MyEncoder) \
    .add(name='translate', uses=EnglishToChineseTranslator, needs='caption') \
    .add(name='text_encoder', uses=TextEncoder, needs='translate')
f_search = Flow().config_gateway(protocol='http', port=12346) \
    .add(name='search', uses=Search)
with f_index, f_search:
    f_index.block(), f_search()
