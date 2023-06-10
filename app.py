
import streamlit as st
from PIL import Image
from jina import Client, DocumentArray, Document
from pprint import pprint
import os
import shutil
import tempfile

# 上传文件并保存到指定目录中
def save_uploaded_file(uploaded_file, target_dir):
    # 如果没有上传文件，则直接返回 None
    if uploaded_file is None:
        return None

    # 创建临时文件，并将上传的二进制数据写入文件中
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.close()

        # 移动临时文件到指定目录中
        shutil.move(tmp_file.name, os.path.join(target_dir, uploaded_file.name))

    # 返回保存的文件路径
    return os.path.join(target_dir, uploaded_file.name)
# 创建上传组件
uploaded_file = st.file_uploader("图片", type=["png", "jpg", "jpeg", "gif"])
# text = uploaded_file.read().decode('utf-8')
# 添加上传按钮
if st.button('上传'):
    # 如果已经选择了文件，则进行处理
    if uploaded_file is not None:
        saved_file_path = save_uploaded_file(uploaded_file, '/home/yingtie/PycharmProjects/image_index/images/')
        st.success(f"图片已成功保存到 {saved_file_path}")


index_url = st.text_input('请输入创建索引的flow-url')

if st.button('为上传图片建立索引'):
    # 从指定目录读取上传的文件内容
    image_uri = '/home/yingtie/PycharmProjects/image_index/images/'+ uploaded_file.name
    doc = Document(uri=image_uri)
    docs = DocumentArray()
    docs.append(doc)
    # 发送 POST 请求并获取响应数据
    url = "http://172.66.1.189:12345"
    c = Client(host=url)
    da = c.post(on='/', inputs=docs, show_progress=True, timeout=3600)
    # text = uploaded_file.read().decode('utf-8')
    img = Image.open(image_uri)
    # 缩放图像大小为原来的一半
    width, height = img.size
    new_width, new_height = int(width / 2), int(height / 2)
    img = img.resize((new_width, new_height))
    da.summary()
    for d in da:
        st.image(img, caption=uploaded_file.name)
        st.write("图片描述为：", d.text)


query_keyword = st.text_input('请输入关键词')
if st.button('搜索') and query_keyword is not None:
    url = "http://172.66.1.189:12345"
    c = Client(host=url)
    da_search = DocumentArray()
    # t1 = Document(text=query_keyword)
    t1 = Document(
        mime_type='',
        text=query_keyword,
        uri=''
    )
    da_search.append(t1)
    print(da_search)
    matches = c.post('/search', inputs=da_search, limit=6, show_progress=True)
    # 使用切片操作获取所有元素的 uri 属性

    col1, col2, col3, col4, col5 = st.columns(5)
    uri_values = matches[0].matches[:, 'uri']
    text_values = matches[0].matches[:, 'text']
    score_values = matches[0].matches[:, 'scores__cos']
    nums = [0, 1, 2, 3, 4]
    for i, uri, text, score in zip(range(5), uri_values, text_values, score_values):
        print(f"URI: {uri}, Text: {text}")
        img = Image.open(uri)
        if i == 0:
            with col1:
                st.image(img, caption=text)
                st.write("余弦相似度: ", score)
        elif i == 1:
            with col2:
                st.image(img, caption=text)
                st.write("余弦相似度: ", score)
        elif i == 2:
            with col3:
                st.image(img, caption=text)
                st.write("余弦相似度: ", score)
        elif i == 3:
            with col4:
                st.image(img, caption=text)
                st.write("余弦相似度: ", score)
        else:
            with col5:
                st.image(img, caption=text)
                st.write("余弦相似度: ", score)
    st.write("响应数据：", matches[0].matches[:, ('uri', 'text', 'scores__cos')])


