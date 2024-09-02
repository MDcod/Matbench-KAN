FROM us-docker.pkg.dev/colab-images/public/runtime

WORKDIR /code
# COPY . .

RUN pip install pykan
RUN pip install requests==2.31.0
RUN pip install git+https://github.com/materialsproject/matbench.git