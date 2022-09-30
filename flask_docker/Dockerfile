#FROM python:3.8.0-alpine
FROM python:3.8

WORKDIR usr/src/flask_app
COPY requirements.txt .
RUN pip install --upgrade pip
#RUN apk update
RUN pip install setuptools==41.2.0
#RUN apk add make automake gcc g++ subversion python3-dev
#RUN ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install --upgrade numpy
RUN pip install --upgrade pip
#RUN pip install opencv-python-headless
#RUN apt-get -y install default-jdk
#RUN apt-get install openjdk-8-jdk
#RUN apt install default-jdk -y
#RUN add-apt-repository ppa:openjdk-r/ppa
#RUN apt-get update
#RUN apt-get install openjdk-8-jre
#RUN apt-get update
#RUN apt-get install -y openjdk-8-jdk --no-cache
RUN pip install opencv-python
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install libglib2.0-0
RUN pip install numpy
RUN pip install wheel
RUN pip install scipy
RUN pip install Cython==0.29.14
#RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
#RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
#RUN pip3 install torch>=1.7.0,!=1.12.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -r requirements.txt #-r 옵션은 파일(requirement.txt)에 적혀있는 라이브러리를 설치하라는 의미
#RUN apt-get install default-jdk -y
COPY ./flask_app .
