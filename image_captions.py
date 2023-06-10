from docarray import DocumentArray, Document
from jina import Client

client = Client(port=12345, protocol='http')
# docs = DocumentArray()
# doc = Document(uri='/home/yingtie/PycharmProjects/image_index/images/*.jpg')
# docs.append(doc)
docs = DocumentArray.from_files('/home/yingtie/PycharmProjects/image_index/images/*.jpg')
da = client.post(on='/', inputs=docs, show_progress=True, timeout=36000)
print(da.texts)
da.summary()